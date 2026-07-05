#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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

from quart import Response, request

from api.apps import current_user, login_required
from api.db.db_models import MCPServer
from api.db.services.mcp_server_service import MCPServerService
from api.db.services.user_service import TenantService
from api.utils.api_utils import get_data_error_result, get_json_result, get_mcp_tools, get_request_json, server_error_response, validate_request
from api.utils.pagination_utils import validate_rest_api_page_size
from api.utils.web_utils import get_float, safe_json_parse
from common.constants import VALID_MCP_SERVER_TYPES
from common.mcp_tool_call_conn import MCPToolCallSession, close_multiple_mcp_toolcall_sessions
from common.misc_utils import get_uuid, thread_pool_exec
from common.ssrf_guard import assert_url_is_safe, pin_dns_global


def _get_mcp_ids_from_args() -> list[str]:
    mcp_ids = request.args.getlist("mcp_ids")
    if mcp_ids:
        return [mcp_id for item in mcp_ids for mcp_id in item.split(",") if mcp_id]
    mcp_ids = request.args.get("mcp_id", "")
    return [mcp_id for mcp_id in mcp_ids.split(",") if mcp_id]


def _export_mcp_servers(mcp_ids: list[str]) -> dict | None:
    exported_servers = {}
    for mcp_id in mcp_ids:
        e, mcp_server = MCPServerService.get_by_id(mcp_id)
        if e and mcp_server.tenant_id == current_user.id:
            server_key = mcp_server.name
            exported_servers[server_key] = {
                "type": mcp_server.server_type,
                "url": mcp_server.url,
                "name": mcp_server.name,
                "authorization_token": mcp_server.variables.get("authorization_token", ""),
                "tools": mcp_server.variables.get("tools", {}),
            }

    if not exported_servers:
        return None

    return {"mcpServers": exported_servers}


def _assert_mcp_url_is_safe(url, invalid_message: str = "Invalid url.") -> tuple[str, str, str | None]:
    if not isinstance(url, str) or not url:
        return "", "", invalid_message
    try:
        hostname, resolved_ip = assert_url_is_safe(url)
    except ValueError as exc:
        return "", "", str(exc)
    return hostname, resolved_ip, None


@manager.route("/mcp/servers", methods=["GET"])  # noqa: F821
@login_required
async def list_mcp() -> Response:
    """
    获取当前用户可访问的 MCP 服务器列表，支持分页和筛选。
    """
    keywords = request.args.get("keywords", "") # 关键词搜索
    page_number = int(request.args.get("page", 0)) # 页码（从 0 开始）
    items_per_page = validate_rest_api_page_size(int(request.args.get("page_size", 0))) # 每页数量（0 表示全部）
    orderby = request.args.get("orderby", "create_time") # 排序字段
    if request.args.get("desc", "true").lower() == "false": # 是否降序
        desc = False
    else:
        desc = True

    mcp_ids = _get_mcp_ids_from_args() # 指定 MCP 服务器 ID 列表
    try:
        # 查询服务器列表
        servers = MCPServerService.get_servers(current_user.id, mcp_ids, 0, 0, orderby, desc, keywords) or []
        total = len(servers)

        # 分页处理
        # 1.只在指定了分页参数时才分页
        # 2.page=0 时不分页
        if page_number and items_per_page:
            servers = servers[(page_number - 1) * items_per_page : page_number * items_per_page]

        return get_json_result(data={"mcp_servers": servers, "total": total})
    except Exception as e:
        return server_error_response(e)


@manager.route("/mcp/servers/<mcp_id>", methods=["GET"])  # noqa: F821
@login_required
def detail(mcp_id: str) -> Response:
    """
    获取指定 MCP 服务器的完整信息，支持普通查看和导出两种模式。

    用场景
    场景	    模式	    说明
    查看配置	普通	    查看服务器的详细配置
    编辑配置	普通	    加载配置到编辑表单
    备份/迁移 导出	导出配置用于备份或导入到其他租户
    共享配置	导出	    导出配置分享给其他用户
    """
    try:
        # 导出模式
        # 当 mode=download 时，导出服务器配置
        if request.args.get("mode") == "download":
            # 调用 _export_mcp_servers 处理导出逻辑
            # 导出格式可能为 JSON 或其他可导入格式
            exported_servers = _export_mcp_servers([mcp_id])
            if exported_servers is None:
                return get_data_error_result(message=f"Cannot find MCP server {mcp_id} for user {current_user.id}")
            return get_json_result(data=exported_servers)

        # 普通详情模式
        # 1.查询指定 ID 的 MCP 服务器
        # 2.必须属于当前用户（tenant_id=current_user.id）
        # 3.返回服务器的完整配置
        mcp_server = MCPServerService.get_or_none(id=mcp_id, tenant_id=current_user.id)

        if mcp_server is None:
            return get_data_error_result(message=f"Cannot find MCP server {mcp_id} for user {current_user.id}")

        return get_json_result(data=mcp_server.to_dict())
    except Exception as e:
        return server_error_response(e)


@manager.route("/mcp/servers", methods=["POST"])  # noqa: F821
@login_required
@validate_request("name", "url", "server_type")
async def create() -> Response:
    """
    创建 MCP 服务器配置，验证连接并获取服务器提供的工具列表。
    """
    # 参数验证
    # 1.验证服务器类型是否合法
    # 2.验证名称长度不超过 255 字节
    # 3.检查名称是否已存在
    req = await get_request_json()

    server_type = req.get("server_type", "")
    if server_type not in VALID_MCP_SERVER_TYPES:
        return get_data_error_result(message="Unsupported MCP server type.")

    server_name = req.get("name", "")
    if not server_name or len(server_name.encode("utf-8")) > 255:
        return get_data_error_result(message=f"Invalid MCP name or length is {len(server_name)} which is large than 255.")

    e, _ = MCPServerService.get_by_name_and_tenant(name=server_name, tenant_id=current_user.id)
    if e:
        return get_data_error_result(message="Duplicated MCP server name.")

    # URL 安全验证
    # 1.解析 URL 并验证安全性
    # 2.防止 SSRF 攻击
    # 3.获取主机名和解析的 IP
    url = req.get("url", "")
    hostname, resolved_ip, url_error = _assert_mcp_url_is_safe(url)
    if url_error:
        return get_data_error_result(message=url_error)

    headers = safe_json_parse(req.get("headers", {}))
    req["headers"] = headers
    variables = safe_json_parse(req.get("variables", {}))
    variables.pop("tools", None)

    timeout = get_float(req, "timeout", 10)

    try:
        req["id"] = get_uuid()
        req["tenant_id"] = current_user.id

        e, _ = TenantService.get_by_id(current_user.id)
        if not e:
            return get_data_error_result(message="Tenant not found.")

        mcp_server = MCPServer(id=server_name, name=server_name, url=url, server_type=server_type, variables=variables, headers=headers)
        with pin_dns_global(hostname, resolved_ip):
            server_tools, err_message = await thread_pool_exec(get_mcp_tools, [mcp_server], timeout)
        if err_message:
            return get_data_error_result(message=err_message)

        tools = server_tools[server_name]
        tools = {tool["name"]: tool for tool in tools if isinstance(tool, dict) and "name" in tool}
        variables["tools"] = tools
        req["variables"] = variables

        if not MCPServerService.insert(**req):
            return get_data_error_result(message="Failed to create MCP server.")

        return get_json_result(data=req)
    except Exception as e:
        return server_error_response(e)


@manager.route("/mcp/servers/<mcp_id>", methods=["PUT"])  # noqa: F821
@login_required
async def update(mcp_id: str) -> Response:
    """
    更新 MCP 服务器配置，验证新配置的有效性，并重新获取工具列表。
    """
    req = await get_request_json()

    # 获取现有服务器
    # 1.验证服务器存在且属于当前用户
    # 2.不存在或无权限时返回错误
    e, mcp_server = MCPServerService.get_by_id(mcp_id)
    if not e or mcp_server.tenant_id != current_user.id:
        return get_data_error_result(message=f"Cannot find MCP server {mcp_id} for user {current_user.id}")

    # 服务器类型必须在有效列表中
    server_type = req.get("server_type", mcp_server.server_type)
    if server_type and server_type not in VALID_MCP_SERVER_TYPES:
        return get_data_error_result(message="Unsupported MCP server type.")
    # 名称长度不超过 255 字节
    server_name = req.get("name", mcp_server.name)
    if server_name and len(server_name.encode("utf-8")) > 255:
        return get_data_error_result(message=f"Invalid MCP name or length is {len(server_name)} which is large than 255.")
    # URL 必须通过安全检查
    url = req.get("url", mcp_server.url)
    hostname, resolved_ip, url_error = _assert_mcp_url_is_safe(url)
    if url_error:
        return get_data_error_result(message=url_error)

    # 解析 JSON 字段
    # 1.安全解析 headers 和 variables JSON 字段
    # 2.移除 tools（将由系统重新获取）
    headers = safe_json_parse(req.get("headers", mcp_server.headers))
    req["headers"] = headers

    variables = safe_json_parse(req.get("variables", mcp_server.variables))
    variables.pop("tools", None)

    timeout = get_float(req, "timeout", 10)

    try:
        req["tenant_id"] = current_user.id
        req["id"] = mcp_id

        # 测试连接并获取工具
        # 1.创建临时 MCP 服务器对象
        # 2.使用 DNS 固定防止 DNS 重绑定
        # 3.调用 get_mcp_tools 获取工具列表
        # 4.将工具列表存入 variables["tools"]
        mcp_server = MCPServer(id=server_name, name=server_name, url=url, server_type=server_type, variables=variables, headers=headers)
        with pin_dns_global(hostname, resolved_ip):
            server_tools, err_message = await thread_pool_exec(get_mcp_tools, [mcp_server], timeout)
        if err_message:
            return get_data_error_result(message=err_message)

        tools = server_tools[server_name]
        tools = {tool["name"]: tool for tool in tools if isinstance(tool, dict) and "name" in tool}
        variables["tools"] = tools
        req["variables"] = variables

        # 保存更新
        # 1.设置租户 ID 和服务器 ID
        # 2.执行更新操作
        # 3.获取更新后的服务器信息并返回
        if not MCPServerService.filter_update([MCPServer.id == mcp_id, MCPServer.tenant_id == current_user.id], req):
            return get_data_error_result(message="Failed to updated MCP server.")

        e, updated_mcp = MCPServerService.get_by_id(req["id"])
        if not e:
            return get_data_error_result(message="Failed to fetch updated MCP server.")

        return get_json_result(data=updated_mcp.to_dict())
    except Exception as e:
        return server_error_response(e)


@manager.route("/mcp/servers/<mcp_id>", methods=["DELETE"])  # noqa: F821
@login_required
async def rm(mcp_id: str) -> Response:
    """
    删除指定的 MCP 服务器配置。
    """
    try:
        # 服务器存在性校验
        # 1.检查服务器是否存在
        # 2.确保服务器属于当前用户
        # 3.不存在或无权限时返回错误
        e, mcp_server = MCPServerService.get_by_id(mcp_id)
        if not e or mcp_server.tenant_id != current_user.id:
            return get_data_error_result(message=f"Cannot find MCP server {mcp_id} for user {current_user.id}")
        # 执行删除
        if not MCPServerService.delete_by_ids([mcp_id]):
            return get_data_error_result(message=f"Failed to delete MCP servers {[mcp_id]}")

        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/mcp/servers/import", methods=["POST"])  # noqa: F821
@login_required
@validate_request("mcpServers")
async def import_multiple() -> Response:
    """
    批量导入 MCP 服务器配置，自动处理重名，验证连接并获取工具列表。
    """
    req = await get_request_json()
    servers = req.get("mcpServers", {}) # 解析请求中的 mcpServers 字典
    if not servers:
        return get_data_error_result(message="No MCP servers provided.")

    timeout = get_float(req, "timeout", 10) # 获取超时设置（默认 10 秒）

    results = []
    try:
        # 遍历导入
        # 1.遍历每个服务器配置
        # 2.导入不成功时记录错误，继续处理下一个
        for server_name, config in servers.items():
            # 配置验证
            # 必填字段检查
            if not all(key in config for key in {"type", "url"}):
                results.append({"server": server_name, "success": False, "message": "Missing required fields (type or url)"})
                continue

            # 名称长度检查
            if not server_name or len(server_name.encode("utf-8")) > 255:
                results.append({"server": server_name, "success": False, "message": f"Invalid MCP name or length is {len(server_name)} which is large than 255."})
                continue
            # 服务器类型验证
            if config["type"] not in VALID_MCP_SERVER_TYPES:
                results.append({"server": server_name, "success": False, "message": "Unsupported MCP server type."})
                continue
            # URL 安全检查
            hostname, resolved_ip, url_error = _assert_mcp_url_is_safe(config["url"])
            if url_error:
                results.append({"server": server_name, "success": False, "message": url_error})
                continue

            # 名称去重处理
            # 1.检查名称是否已存在
            # 2.存在则追加数字后缀（如 server_1、server_2）
            # 3.直到找到可用名称
            base_name = server_name
            new_name = base_name
            counter = 0

            while True:
                e, _ = MCPServerService.get_by_name_and_tenant(name=new_name, tenant_id=current_user.id)
                if not e:
                    break
                new_name = f"{base_name}_{counter}"
                counter += 1

            # 准备创建数据
            create_data = {
                "id": get_uuid(),
                "tenant_id": current_user.id,
                "name": new_name,
                "url": config["url"],
                "server_type": config["type"],
                "variables": {"authorization_token": config.get("authorization_token", "")},
            }

            headers = {"authorization_token": config["authorization_token"]} if "authorization_token" in config else {}
            variables = {k: v for k, v in config.items() if k not in {"type", "url", "headers"}}
            # 测试连接并获取工具
            mcp_server = MCPServer(id=new_name, name=new_name, url=config["url"], server_type=config["type"], variables=variables, headers=headers)
            with pin_dns_global(hostname, resolved_ip):
                server_tools, err_message = await thread_pool_exec(get_mcp_tools, [mcp_server], timeout)
            if err_message:
                results.append({"server": base_name, "success": False, "message": err_message})
                continue

            tools = server_tools[new_name]
            tools = {tool["name"]: tool for tool in tools if isinstance(tool, dict) and "name" in tool}
            create_data["variables"]["tools"] = tools

            # 保存服务器
            if MCPServerService.insert(**create_data):
                result = {"server": server_name, "success": True, "action": "created", "id": create_data["id"], "new_name": new_name}
                if new_name != base_name:
                    result["message"] = f"Renamed from '{base_name}' to '{new_name}' avoid duplication"
                results.append(result)
            else:
                results.append({"server": server_name, "success": False, "message": "Failed to create MCP server."})

        return get_json_result(data={"results": results})
    except Exception as e:
        return server_error_response(e)


@manager.route("/mcp/servers/<mcp_id>/test", methods=["POST"])  # noqa: F821
@login_required
@validate_request("url", "server_type")
async def test_mcp(mcp_id: str) -> Response:
    req = await get_request_json()

    url = req.get("url", "")
    if not isinstance(url, str) or not url:
        return get_data_error_result(message="Invalid MCP url.")

    server_type = req.get("server_type", "")
    if server_type not in VALID_MCP_SERVER_TYPES:
        return get_data_error_result(message="Unsupported MCP server type.")

    hostname, resolved_ip, url_error = _assert_mcp_url_is_safe(url, "Invalid MCP url.")
    if url_error:
        return get_data_error_result(message=url_error)

    timeout = get_float(req, "timeout", 10)
    headers = safe_json_parse(req.get("headers", {}))
    variables = safe_json_parse(req.get("variables", {}))

    mcp_server = MCPServer(id=mcp_id, server_type=server_type, url=url, headers=headers, variables=variables)

    result = []
    try:
        with pin_dns_global(hostname, resolved_ip):
            tool_call_session = MCPToolCallSession(mcp_server, mcp_server.variables)

            try:
                tools = await thread_pool_exec(tool_call_session.get_tools, timeout)
            except Exception as e:
                return get_data_error_result(message=f"Test MCP error: {e}")
            finally:
                await thread_pool_exec(close_multiple_mcp_toolcall_sessions, [tool_call_session])

        for tool in tools:
            tool_dict = tool.model_dump()
            tool_dict["enabled"] = True
            result.append(tool_dict)

        return get_json_result(data=result)
    except Exception as e:
        return server_error_response(e)
