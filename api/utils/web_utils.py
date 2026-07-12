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

"""
Web 工具函数模块 —— 提供邮件发送、HTML→PDF 转换、文件响应安全、OTP 验证码等通用 Web 工具。

本模块包含：
- 文件类型/扩展名的 Content-Type 映射
- 浏览器下载安全策略（强制附件下载 + nosniff）
- 基于 Selenium + Chrome Headless 的 HTML → PDF 转换
- URL 安全校验
- 异步邮件发送（SMTP）
- OTP（一次性密码）验证相关的键名生成和哈希
"""

import base64  # Base64 编解码（PDF 二进制数据处理）
import json  # JSON 解析
import re  # 正则表达式（URL 格式校验）
import aiosmtplib  # 异步 SMTP 客户端（邮件发送）
from email.mime.text import MIMEText  # 邮件正文 MIME 构造
from email.header import Header  # 邮件头编码处理
from common import settings  # 全局配置（邮件、密钥等）
from quart import render_template_string  # Quart 模板渲染（邮件正文）
from api.utils.email_templates import EMAIL_TEMPLATES  # 邮件模板字典
from selenium import webdriver  # Selenium WebDriver（HTML→PDF 转换）
from selenium.common.exceptions import TimeoutException  # Selenium 超时异常
from selenium.webdriver.chrome.options import Options  # Chrome 浏览器选项
from selenium.webdriver.chrome.service import Service  # ChromeDriver 服务管理
from selenium.webdriver.common.by import By  # 元素定位策略
from selenium.webdriver.support.expected_conditions import staleness_of  # 等待页面元素失效
from selenium.webdriver.support.ui import WebDriverWait  # 显式等待工具
from webdriver_manager.chrome import ChromeDriverManager  # ChromeDriver 自动下载管理


# =============================================================================
# OTP（一次性密码）验证参数
# =============================================================================
OTP_LENGTH = 4                     # 验证码长度（4 位数字）
OTP_TTL_SECONDS = 5 * 60           # 验证码有效期（5 分钟）
ATTEMPT_LIMIT = 5                  # 最大尝试次数
ATTEMPT_LOCK_SECONDS = 30 * 60     # 超过尝试次数后的锁定时间（30 分钟）
RESEND_COOLDOWN_SECONDS = 60       # 两次发送验证码的最小间隔（1 分钟）


# =============================================================================
# 文件扩展名 → MIME Content-Type 映射表
# =============================================================================
CONTENT_TYPE_MAP = {
    # ---- Office 办公文档 ----
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "pdf": "application/pdf",
    "csv": "text/csv",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # ---- 文本/代码文件 ----
    "txt": "text/plain",
    "py": "text/plain",
    "js": "text/plain",
    "java": "text/plain",
    "c": "text/plain",
    "cpp": "text/plain",
    "h": "text/plain",
    "php": "text/plain",
    "go": "text/plain",
    "ts": "text/plain",
    "sh": "text/plain",
    "cs": "text/plain",
    "kt": "text/plain",
    "sql": "text/plain",
    # ---- Web 文档 ----
    "md": "text/markdown",
    "markdown": "text/markdown",
    "mdx": "text/markdown",
    "htm": "text/html",
    "html": "text/html",
    "json": "application/json",
    # ---- 图片格式 ----
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "ico": "image/x-icon",
    "avif": "image/avif",
    "heic": "image/heic",
    # ---- 演示文稿 ----
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


# =============================================================================
# 安全策略：强制作为附件下载（而非浏览器直接渲染）的文件类型
# =============================================================================
# 这些扩展名的文件即使 Content-Type 允许浏览器内预览，也必须强制下载。
# 目的：防止 SVG/HTML/XML 等可包含脚本的文件在浏览器中直接打开造成 XSS 攻击。
FORCE_ATTACHMENT_EXTENSIONS = {
    "htm",
    "html",
    "shtml",
    "xht",
    "xhtml",
    "xml",
    "mhtml",
    "svg",
}


# 这些 Content-Type 对应的响应也必须强制作为附件下载。
FORCE_ATTACHMENT_CONTENT_TYPES = {
    "text/html",
    "image/svg+xml",
    "application/xhtml+xml",
    "text/xml",
    "application/xml",
    "multipart/related",
}


def should_force_attachment(ext: str | None, content_type: str | None = None) -> bool:
    """判断文件是否应强制作为附件下载（而非浏览器内预览）。

    安全策略核心函数 —— 对 HTML/XML/SVG 等可能包含可执行脚本的文件类型，
    必须设置 Content-Disposition: attachment 头，配合 nosniff 防止
    MIME 类型嗅探攻击（XSS via MIME sniffing）。

    Args:
        ext: 文件扩展名（如 "html"、"svg"）。
        content_type: HTTP 响应的 Content-Type 头值。

    Returns:
        bool —— True 表示必须强制作为附件下载。
    """
    normalized_ext = (ext or "").lower().strip(".")
    if normalized_ext in FORCE_ATTACHMENT_EXTENSIONS:
        return True
    normalized_type = (content_type or "").lower()
    return normalized_type in FORCE_ATTACHMENT_CONTENT_TYPES


def apply_safe_file_response_headers(response, content_type: str | None, ext: str | None = None):
    """为文件下载响应设置安全的 HTTP 响应头。

    设置 Content-Type，并对 HTML/SVG/XML 等危险文件类型
    添加 X-Content-Type-Options: nosniff 和 Content-Disposition: attachment，
    禁止浏览器进行 MIME 类型嗅探和内联渲染。

    Args:
        response: Quart Response 对象。
        content_type: 文件的 MIME 类型。
        ext: 文件扩展名。

    Returns:
        修改后的 Response 对象。
    """
    if content_type:
        response.headers.set("Content-Type", content_type)
    force_attachment = should_force_attachment(ext, content_type)
    if force_attachment:
        response.headers.set("X-Content-Type-Options", "nosniff")
        response.headers.set("Content-Disposition", "attachment")
    return response


def html2pdf(
    source: str,
    timeout: int = 2,
    install_driver: bool = True,
    print_options: dict = {},
):
    """将 HTML 源码或 URL 转换为 PDF 字节数据。

    基于 Chrome Headless 模式，通过 Selenium + Chrome DevTools Protocol
    调用 Page.printToPDF 生成 PDF。适用于将 Markdown 渲染后的 HTML、
    报告页面等转换为可下载的 PDF 文件。

    Args:
        source: HTML 源码字符串或本地文件路径（file:// 协议）。
        timeout: 页面加载超时时间（秒），默认 2 秒。
        install_driver: 是否自动下载 ChromeDriver，默认 True。
        print_options: 传递给 Page.printToPDF 的打印选项，如
                       landscape（横向）、printBackground（打印背景色）等。

    Returns:
        bytes —— PDF 文件的二进制数据。
    """
    result = __get_pdf_from_html(source, timeout, install_driver, print_options)
    return result


def __send_devtools(driver, cmd, params={}):
    """向 Chrome DevTools Protocol 发送命令（内部辅助函数）。

    通过 Selenium WebDriver 的底层 HTTP 接口，直接发送 Chrome DevTools
    协议命令。用于执行 Page.printToPDF 等 WebDriver 原生不支持的操作。

    Args:
        driver: Selenium WebDriver 实例。
        cmd: Chrome DevTools 命令名（如 "Page.printToPDF"）。
        params: 命令参数字典。

    Returns:
        dict —— DevTools 命令的返回结果（'value' 字段）。

    Raises:
        Exception: 如果 DevTools 返回错误。
    """
    resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
    url = driver.command_executor._url + resource
    body = json.dumps({"cmd": cmd, "params": params})
    response = driver.command_executor._request("POST", url, body)

    if not response:
        raise Exception(response.get("value"))

    return response.get("value")


def __get_pdf_from_html(path: str, timeout: int, install_driver: bool, print_options: dict):
    """使用 Chrome Headless 将 HTML 页面渲染为 PDF（内部实现）。

    工作流程：
    1. 配置 Chrome Headless 选项（禁用 GPU、沙箱、共享内存等）
    2. 禁用图片加载以减少渲染时间
    3. 启动或自动安装 ChromeDriver
    4. 加载页面并等待 DOM 稳定（staleness_of 原 html 元素）
    5. 调用 Chrome DevTools Page.printToPDF 生成 PDF
    6. Base64 解码返回 PDF 二进制数据
    7. finally 中确保退出 WebDriver

    Args:
        path: HTML 文件路径（file:// 协议）。
        timeout: 页面加载超时时间（秒）。
        install_driver: 是否自动安装 ChromeDriver。
        print_options: Page.printToPDF 的打印选项。

    Returns:
        bytes —— PDF 二进制数据。
    """
    webdriver_options = Options()
    webdriver_prefs = {}
    # Chrome Headless 模式配置
    webdriver_options.add_argument("--headless")
    webdriver_options.add_argument("--disable-gpu")
    webdriver_options.add_argument("--no-sandbox")
    webdriver_options.add_argument("--disable-dev-shm-usage")
    webdriver_options.experimental_options["prefs"] = webdriver_prefs

    # 禁用图片加载以提升渲染速度
    webdriver_prefs["profile.default_content_settings"] = {"images": 2}

    # 自动下载或使用系统安装的 ChromeDriver
    if install_driver:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=webdriver_options)
    else:
        driver = webdriver.Chrome(options=webdriver_options)

    driver.get(path)

    # 等待页面 DOM 稳定：监听原 <html> 元素变为 stale（页面已刷新/替换）
    try:
        WebDriverWait(driver, timeout).until(staleness_of(driver.find_element(by=By.TAG_NAME, value="html")))
    except TimeoutException:
        pass

    try:
        # 构造打印选项并合并调用方传入的自定义选项
        calculated_print_options = {
            "landscape": False,           # 纵向打印
            "displayHeaderFooter": False, # 不显示页眉页脚
            "printBackground": True,      # 打印 CSS 背景色/图
            "preferCSSPageSize": True,    # 优先使用 CSS @page 定义的尺寸
        }
        calculated_print_options.update(print_options)
        result = __send_devtools(driver, "Page.printToPDF", calculated_print_options)
        # Page.printToPDF 返回 Base64 编码的 PDF 数据
        return base64.b64decode(result["data"])
    finally:
        driver.quit()


def is_valid_url(url: str) -> bool:
    """校验 URL 是否合法且安全。

    分两步校验：
    1. 正则表达式匹配 URL 格式（仅允许 http/https 协议）
    2. SSRF 防护检查 —— 禁止访问内网地址（127.0.0.1、10.x、172.16-31.x、192.168.x 等）

    Args:
        url: 待校验的 URL 字符串。

    Returns:
        bool —— 格式合法且通过 SSRF 安全检查则为 True。
    """
    if not re.match(r"(https?)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", url):
        return False
    from common.ssrf_guard import assert_url_is_safe

    try:
        assert_url_is_safe(url)
        return True
    except ValueError:
        return False


def safe_json_parse(data: str | dict) -> dict:
    """安全解析 JSON 字符串，永远返回 dict。

    三种情况处理：
    - 已经是 dict → 直接返回
    - 有效 JSON 字符串 → 解析后返回
    - 解析失败 / 空值 → 返回 {}

    Args:
        data: JSON 字符串或 dict 对象。

    Returns:
        dict —— 解析后的字典，永远不会抛出异常。
    """
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data) if data else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def get_float(req: dict, key: str, default: float | int = 10.0) -> float:
    """从请求字典中安全提取浮点数参数。

    用于解析用户传入的数值型参数（如 LLM 温度 temperature），
    做合法性校验：
    - 值必须 > 0，否则返回默认值
    - 无法转换为 float 时返回默认值

    Args:
        req: 请求参数字典。
        key: 要提取的键名。
        default: 转换失败或值无效时的默认值。

    Returns:
        float —— 有效的正浮点数或默认值。
    """
    try:
        parsed = float(req.get(key, default))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


async def send_email_html(to_email: str, subject: str, template_key: str, **context):
    """异步发送 HTML 邮件。

    使用 aiosmtplib（异步 SMTP 客户端）通过 TLS 加密连接发送邮件。
    邮件正文由 Jinja2 模板渲染生成，模板存储在 EMAIL_TEMPLATES 中。

    工作流程：
    1. 用 Quart 模板引擎渲染邮件正文
    2. 构造 MIMEText 邮件对象
    3. 通过 TLS 连接到 SMTP 服务器
    4. 登录并发送邮件

    Args:
        to_email: 收件人邮箱地址。
        subject: 邮件主题。
        template_key: 邮件模板键名（对应 EMAIL_TEMPLATES 中的 key）。
        **context: 传递给模板的上下文变量。
    """
    body = await render_template_string(EMAIL_TEMPLATES.get(template_key), **context)
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = f"{settings.MAIL_DEFAULT_SENDER[0]} <{settings.MAIL_DEFAULT_SENDER[1]}>"
    msg["To"] = to_email

    smtp = aiosmtplib.SMTP(
        hostname=settings.MAIL_SERVER,
        port=settings.MAIL_PORT,
        use_tls=True,
        timeout=10,
    )

    await smtp.connect()
    await smtp.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
    await smtp.send_message(msg)
    await smtp.quit()


async def send_invite_email(to_email, invite_url, tenant_id, inviter):
    """发送团队邀请邮件。

    复用 send_email_html 通用发送器，使用 'invite' 模板，
    将邀请链接、租户信息、邀请人信息渲染到邮件正文中。

    Args:
        to_email: 被邀请人邮箱。
        invite_url: 邀请链接（包含 token）。
        tenant_id: 租户/团队 ID。
        inviter: 邀请人的显示名称。
    """
    await send_email_html(
        to_email=to_email,
        subject="RAGFlow Invitation",
        template_key="invite",
        email=to_email,
        invite_url=invite_url,
        tenant_id=tenant_id,
        inviter=inviter,
    )


def otp_keys(email: str):
    """生成 OTP 验证码相关的 Redis 键名元组。

    为指定邮箱生成四个 Redis key，用于 OTP 生命周期管理：
    - otp:{email}          —— 存储验证码哈希值
    - otp_attempts:{email} —— 记录错误尝试次数
    - otp_last_sent:{email}—— 记录上次发送时间（防重复发送）
    - otp_lock:{email}     —— 锁定状态（超过尝试次数后锁定）

    Args:
        email: 用户邮箱地址（会做 strip + lower 归一化）。

    Returns:
        tuple —— (otp_key, attempts_key, last_sent_key, lock_key)。
    """
    email = (email or "").strip().lower()
    return (
        f"otp:{email}",
        f"otp_attempts:{email}",
        f"otp_last_sent:{email}",
        f"otp_lock:{email}",
    )


def hash_code(code: str, salt: bytes) -> str:
    """使用 HMAC-SHA256 对验证码进行哈希（带盐值）。

    用于 OTP 验证码的安全存储 —— 不存储明文验证码，
    而是存储其 HMAC-SHA256 哈希值，配合随机 salt 防止彩虹表攻击。

    Args:
        code: 需要哈希的验证码字符串。
        salt: 随机盐值（字节串）。

    Returns:
        str —— 十六进制格式的 HMAC-SHA256 哈希值。
    """
    import hashlib
    import hmac

    return hmac.new(salt, (code or "").encode("utf-8"), hashlib.sha256).hexdigest()


def captcha_key(email: str) -> str:
    """生成图形验证码（Captcha）的 Redis 键名。

    用于人机验证：登录页面输入验证码后，后端通过此 key 从 Redis
    中获取正确的验证码文本进行对比校验。

    Args:
        email: 用户邮箱地址。

    Returns:
        str —— Redis 键名，格式为 "captcha:{email}"。
    """
    return f"captcha:{email}"
