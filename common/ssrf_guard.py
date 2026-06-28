#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Shared SSRF-guard utilities.

Uses only the standard library so it can be imported from both ``api/`` and
``common/`` without pulling in any heavyweight dependencies.
"""

import ipaddress
import logging
import socket
import threading
from contextlib import contextmanager
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNS pinning — closes the TOCTOU / rebinding window between SSRF validation
# and the actual TCP connection.  The monkey-patch is a no-op for any host
# that has no active pin, so it cannot affect unrelated code.
# ---------------------------------------------------------------------------

_tl = threading.local()
_global_dns_pins: dict[str, str] = {}
_global_pin_lock = threading.Lock()
_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_with_pins(host, port, *args, **kwargs):
    # Thread-local pins (synchronous callers: requests.get in the same thread)
    local_pins: dict = getattr(_tl, "dns_pins", {})
    if host in local_pins:
        ip = local_pins[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
    # Process-global pins (async callers whose DNS resolves in executor threads)
    with _global_pin_lock:
        ip = _global_dns_pins.get(host)
    if ip is not None:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _getaddrinfo_with_pins


@contextmanager
def pin_dns(hostname: str, ip: str):
    """Pin *hostname* → *ip* in the current thread for the duration of this context.

    Use for synchronous ``requests.get()`` callers to prevent DNS rebinding
    between SSRF validation and the actual TCP connection.
    """
    pins = _tl.__dict__.setdefault("dns_pins", {})
    pins[hostname] = ip
    try:
        yield
    finally:
        pins.pop(hostname, None)


@contextmanager
def pin_dns_global(hostname: str, ip: str):
    """Pin *hostname* → *ip* across all threads for the duration of this context.

    Use for async callers (e.g. asyncio-based crawlers) where DNS resolution
    may happen in thread-pool executor threads rather than the calling thread.
    """
    with _global_pin_lock:
        _global_dns_pins[hostname] = ip
    try:
        yield
    finally:
        with _global_pin_lock:
            _global_dns_pins.pop(hostname, None)


_DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _effective_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return the IPv4 equivalent for IPv4-mapped IPv6 addresses, unchanged otherwise.

    Without this normalization ``::ffff:127.0.0.1`` would pass ``is_global``
    as an IPv6Address in some Python versions, bypassing the loopback check.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            return mapped
    return ip


def assert_url_is_safe(
    url: str,
    *,
    allowed_schemes: frozenset[str] = _DEFAULT_ALLOWED_SCHEMES,
) -> tuple[str, str]:
    """Raise ``ValueError`` if *url* is not safe to fetch (SSRF guard).

    Checks performed in order:

    1. Scheme is in *allowed_schemes*.
    2. Hostname is present.
    3. **Every** address returned by ``getaddrinfo`` is globally routable
       (``ip.is_global``).  This is an allowlist approach: it catches private,
       loopback, link-local, reserved, multicast, and all other
       special-purpose ranges rather than individual deny-list flags.
       IPv4-mapped IPv6 addresses (e.g. ``::ffff:127.0.0.1``) are normalised
       to their IPv4 form via :func:`_effective_ip` before the check.

    Returns ``(hostname, resolved_ip)`` — the first validated public IP string
    — so the caller can **pin** that address in its HTTP client and prevent
    DNS-rebinding attacks (the hostname is resolved exactly once).

    SSRF（服务器端请求伪造）防护的关键实现。它的核心目的是：在服务器向外发起HTTP请求前，对目标URL进行严格的安全检查，防止攻击者利用服务器去访问内网资源或进行端口扫描。

    """
    # 协议（Scheme）白名单检查
    #
    # 目的：只允许特定的协议，如 http 和 https，防止使用 file://、gopher:// 等危险协议。
    # 实现：检查URL的scheme是否在预定义的 _DEFAULT_ALLOWED_SCHEMES 集合中。如果不在，直接抛出异常。
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme not in allowed_schemes:
        logger.warning(
            "SSRF guard blocked URL with disallowed scheme: scheme=%r url=%r",
            scheme,
            url,
        )
        raise ValueError(f"Disallowed URL scheme: {scheme!r}. Only {sorted(allowed_schemes)} are allowed.")

    # 主机名（Hostname）存在性检查
    #
    # 目的：确保URL包含有效的主机名，防止解析异常。
    # 实现：解析URL，如果 hostname 为空，则拒绝请求。
    hostname = parsed.hostname
    if not hostname:
        logger.warning("SSRF guard blocked URL with missing host: url=%r", url)
        raise ValueError("URL is missing a host.")

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        logger.warning("SSRF guard could not resolve hostname=%r reason=%s", hostname, exc)
        raise ValueError(f"Could not resolve hostname {hostname!r}: {exc}") from exc

    # IP地址合法性检查（核心防护）
    #
    # 目的：这是最核心的防线，确保解析出的IP地址是公网可路由的，从而阻止对内网IP、本地回环地址（127.0.0.1）、链路本地地址等的访问。
    # 实现：
    # 调用 socket.getaddrinfo 解析主机名，获取所有可能的IP地址。
    # 对每个解析出的IP，使用 _effective_ip 函数进行处理（该函数主要用于将IPv4映射的IPv6地址，如 ::ffff:192.168.1.1，归一化为IPv4格式，以便进行统一的检查）。
    # 使用 ip_address.is_global 方法进行检查。这是一个白名单方法，它只会放行全局公网IP，而拦截所有私有、保留、多播等非公网地址。
    #
    #
    # 防止DNS重绑定攻击
    #
    # 目的：解决攻击者将域名在“解析为公网IP”和“解析为内网IP”之间来回切换的问题。
    # 实现：函数在通过所有检查后，会返回一个确切的IP地址字符串（返回值的第二个元素 resolved_ip）。代码的文档注释中明确说明，调用者应使用这个IP地址去创建HTTP连接，而不是使用原始的域名，这样就确保了连接IP是经过安全检查的唯一地址。
    resolved_ip: str | None = None
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        raw_ip = ipaddress.ip_address(sockaddr[0])
        eff_ip = _effective_ip(raw_ip)
        if not eff_ip.is_global:
            logger.warning(
                "SSRF guard blocked URL: hostname=%r resolved to non-public address=%s",
                hostname,
                raw_ip,
            )
            raise ValueError(f"URL resolves to a non-public address ({raw_ip}), which is not allowed.")
        if resolved_ip is None:
            resolved_ip = str(raw_ip)

    if resolved_ip is None:
        logger.warning("SSRF guard blocked URL: hostname=%r resolved to no addresses", hostname)
        raise ValueError(f"Hostname {hostname!r} resolved to no addresses.")

    return hostname, resolved_ip


def assert_host_is_safe(host: str) -> str:
    """Raise ``ValueError`` if *host* resolves to a non-public IP (SSRF guard for raw host/port connections).

    This is the host-level counterpart of :func:`assert_url_is_safe`, intended
    for callers that connect via database drivers or other non-HTTP protocols
    where there is no URL to parse.

    Returns the first validated public IP string so the caller can pin it if needed.
    """
    if not host:
        raise ValueError("Host must not be empty.")

    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        logger.warning("SSRF guard could not resolve host=%r reason=%s", host, exc)
        raise ValueError(f"Could not resolve host {host!r}: {exc}") from exc

    resolved_ip: str | None = None
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        raw_ip = ipaddress.ip_address(sockaddr[0])
        eff_ip = _effective_ip(raw_ip)
        if not eff_ip.is_global:
            logger.warning(
                "SSRF guard blocked host: host=%r resolved to non-public address=%s",
                host,
                raw_ip,
            )
            raise ValueError(f"Host resolves to a non-public address ({raw_ip}), which is not allowed.")
        if resolved_ip is None:
            resolved_ip = str(raw_ip)

    if resolved_ip is None:
        logger.warning("SSRF guard blocked host: host=%r resolved to no addresses", host)
        raise ValueError(f"Host {host!r} resolved to no addresses.")

    return resolved_ip
