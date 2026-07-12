#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
外部数据源连接器 RESTful API 端点。

本模块提供连接器（Google Drive、Gmail、Box、REST API 等）的 CRUD 操作，
以及 OAuth 2.0 Web 流程端点，允许用户通过基于浏览器的授权页面
授权 RAGFlow 访问其第三方数据。

路由前缀: ``/api/v1/connectors``（由 ``api/apps/restful_apis/__init__.py`` 注册）。

OAuth 2.0 Web 流程（Google / Box）
---------------------------------
1. POST ``/google/oauth/web/start``（或 ``/box/oauth/web/start``）——
   返回授权 URL，并将中间状态缓存到 Redis。
2. GET ``/google-drive/oauth/web/callback``（或 ``/box/oauth/web/callback``）——
   OAuth 提供方重定向到此端点；处理器用授权码换取 token 并将结果存入 Redis。
3. POST ``/google/oauth/web/result``（或 ``/box/oauth/web/result``）——
   前端轮询此端点以获取已存储的 token。
"""
import asyncio
import json
import logging
import time
import uuid
from html import escape
from typing import Any

from quart import request, make_response
from google_auth_oauthlib.flow import Flow

from api.db import InputType
from api.db.services.connector_service import ConnectorService, SyncLogsService
from api.utils.api_utils import get_data_error_result, get_json_result, get_request_json, validate_request
from api.utils.pagination_utils import validate_rest_api_page_size
from common.constants import RetCode, TaskStatus
from common.data_source.config import GOOGLE_DRIVE_WEB_OAUTH_REDIRECT_URI, GMAIL_WEB_OAUTH_REDIRECT_URI, BOX_WEB_OAUTH_REDIRECT_URI, DocumentSource
from common.data_source.google_util.constant import WEB_OAUTH_POPUP_TEMPLATE, GOOGLE_SCOPES
from common.misc_utils import get_uuid
from rag.utils.redis_conn import REDIS_CONN
from api.apps import login_required, current_user
from box_sdk_gen import BoxOAuth, OAuthConfig, GetAuthorizeUrlOptions


# 模块级日志记录器
LOGGER = logging.getLogger(__name__)


def _connector_auth_error(connector_id: str, user_id: str):
    """返回连接器授权失败响应并记录拒绝日志。"""
    LOGGER.warning("connector access denied: connector_id=%s user_id=%s", connector_id, user_id)
    return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)


@manager.route("/connectors/<connector_id>", methods=["PATCH"])  # noqa: F821
@login_required
async def update_connector(connector_id):
    """更新可访问连接器的轮询配置（刷新频率、剪枝频率、超时等）。"""
    if not ConnectorService.accessible(connector_id, current_user.id):
        return _connector_auth_error(connector_id, current_user.id)

    req = await get_request_json()
    if isinstance(req, dict) and isinstance(req.get("data"), dict):
        req = req["data"]

    e, conn = ConnectorService.get_by_id(connector_id)
    if not e:
        return get_data_error_result(message="Can't find this Connector!")

    should_sleep = False
    if req:
        update_fields = {fld: req[fld] for fld in ["prune_freq", "refresh_freq", "config", "timeout_secs"] if fld in req}
        if update_fields:
            update_fields["id"] = connector_id
            ConnectorService.update_by_id(connector_id, update_fields)
            should_sleep = True

        if req.get("reschedule"):
            ConnectorService.cancel_tasks(connector_id)
            ConnectorService.schedule_tasks(connector_id)
        elif req.get("status") in [TaskStatus.CANCEL, "CANCEL"]:
            ConnectorService.cancel_tasks(connector_id)
        elif req.get("status") in [TaskStatus.SCHEDULE, "SCHEDULE"]:
            ConnectorService.schedule_tasks(connector_id)

    if should_sleep:
        await asyncio.sleep(1)
    e, conn = ConnectorService.get_by_id(connector_id)
    if not e:
        return get_data_error_result(message="Can't find this Connector!")

    return get_json_result(data=conn.to_dict())


@manager.route("/connectors", methods=["POST"])  # noqa: F821
@login_required
async def create_connector():
    """创建属于当前租户的连接器。"""
    req = await get_request_json()
    if req:
        req["id"] = get_uuid()
        conn = {
            "id": req["id"],
            "tenant_id": current_user.id,
            "name": req["name"],
            "source": req["source"],
            "input_type": InputType.POLL,
            "config": req["config"],
            "refresh_freq": int(req.get("refresh_freq", 5)),
            "prune_freq": int(req.get("prune_freq", 5)),
            "timeout_secs": int(req.get("timeout_secs", 60 * 29)),
            "status": TaskStatus.UNSTART,
        }
        ConnectorService.save(**conn)

    await asyncio.sleep(1)
    e, conn = ConnectorService.get_by_id(req["id"])

    return get_json_result(data=conn.to_dict())


@manager.route("/connectors", methods=["GET"])  # noqa: F821
@login_required
def list_connector():
    """列出当前租户拥有的所有连接器。"""
    return get_json_result(data=ConnectorService.list(current_user.id))


@manager.route("/connectors/<connector_id>", methods=["GET"])  # noqa: F821
@login_required
def get_connector(connector_id):
    """当前用户可访问时返回连接器详情。"""
    if not ConnectorService.accessible(connector_id, current_user.id):
        return _connector_auth_error(connector_id, current_user.id)

    e, conn = ConnectorService.get_by_id(connector_id)
    if not e:
        return get_data_error_result(message="Can't find this Connector!")
    return get_json_result(data=conn.to_dict())


@manager.route("/connectors/<connector_id>/logs", methods=["GET"])  # noqa: F821
@login_required
def list_logs(connector_id):
    """列出当前用户可访问连接器的同步日志。"""
    if not ConnectorService.accessible(connector_id, current_user.id):
        return _connector_auth_error(connector_id, current_user.id)

    req = request.args.to_dict(flat=True)
    arr, total = SyncLogsService.list_sync_tasks(
        connector_id,
        int(req.get("page", 1)),
        validate_rest_api_page_size(int(req.get("page_size", 15))),
    )
    return get_json_result(data={"total": total, "logs": arr})


@manager.route("/connectors/<connector_id>/rebuild", methods=["POST"])  # noqa: F821
@login_required
async def rebuild(connector_id):
    """为可访问的连接器和知识库触发重建任务。"""
    if not ConnectorService.accessible(connector_id, current_user.id):
        return _connector_auth_error(connector_id, current_user.id)

    req = await get_request_json()
    if "kb_id" not in req:
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="required argument is missing: kb_id")

    err = ConnectorService.rebuild(req["kb_id"], connector_id, current_user.id)
    if err:
        return get_json_result(data=False, message=err, code=RetCode.SERVER_ERROR)
    return get_json_result(data=True)


@manager.route("/connectors/<connector_id>", methods=["DELETE"])  # noqa: F821
@login_required
def rm_connector(connector_id):
    """先取消同步任务，再删除可访问的连接器。"""
    if not ConnectorService.accessible(connector_id, current_user.id):
        return _connector_auth_error(connector_id, current_user.id)

    ConnectorService.cancel_tasks(connector_id)
    ConnectorService.delete_by_id(connector_id)
    return get_json_result(data=True)


@manager.route("/connectors/<connector_id>/test", methods=["POST"])  # noqa: F821
@login_required
async def test_connector(connector_id):
    """验证连接器配置，不持久化更改也不触发同步。

    对于 REST API 连接器，使用 ``RestAPIConnector.validate_config``
    基于已保存的配置进行验证。
    """
    if not ConnectorService.accessible(connector_id, current_user.id):
        return _connector_auth_error(connector_id, current_user.id)

    from common.data_source.rest_api_connector import RestAPIConnector
    from common.data_source.exceptions import ConnectorMissingCredentialError, ConnectorValidationError

    ok, conn = ConnectorService.get_by_id(connector_id)
    if not ok:
        return get_data_error_result(message="Can't find this Connector!")

    if conn.source != DocumentSource.REST_API:
        return get_json_result(
            code=RetCode.ARGUMENT_ERROR,
            message="Test endpoint currently supports only REST API connectors.",
            data=False,
        )

    config = conn.config or {}
    credentials = config.get("credentials") or {}

    try:
        await asyncio.to_thread(
            RestAPIConnector.validate_config,
            config=config,
            credentials=credentials,
        )
    except (ConnectorValidationError, ConnectorMissingCredentialError) as exc:
        return get_json_result(
            code=RetCode.DATA_ERROR,
            message=str(exc),
            data=False,
        )
    except Exception as exc:
        logging.exception("REST API connector validation failed: %s", exc)
        return get_json_result(
            code=RetCode.SERVER_ERROR,
            message="REST API connector validation failed, please check logs.",
            data=False,
        )

    return get_json_result(data=True)


# OAuth Web 流程状态和结果缓存的 Redis TTL（秒）。
# 超过此时间窗口后，用户必须重新发起授权流程。
WEB_FLOW_TTL_SECS = 15 * 60


def _web_state_cache_key(flow_id: str, source_type: str | None = None) -> str:
    """返回 Web OAuth 状态的 Redis 键名。

    默认前缀保持对 Google Drive 的向后兼容性。
    当 source_type == "gmail" 时，使用不同的前缀，
    以避免 Drive/Gmail 流程在 Redis 中冲突。
    """
    prefix = f"{source_type}_web_flow_state"
    return f"{prefix}:{flow_id}"


def _web_result_cache_key(flow_id: str, source_type: str | None = None) -> str:
    """返回 Web OAuth 结果的 Redis 键名。

    与 _web_state_cache_key 逻辑一致，用于结果存储。
    """
    prefix = f"{source_type}_web_flow_result"
    return f"{prefix}:{flow_id}"


def _load_credentials(payload: str | dict[str, Any]) -> dict[str, Any]:
    """将 JSON 字符串或字典解析为凭据字典。"""
    if isinstance(payload, dict):
        return payload
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("Invalid Google credentials JSON.") from exc


def _get_web_client_config(credentials: dict[str, Any]) -> dict[str, Any]:
    """从凭据中提取 Google OAuth Web 客户端配置，若不存在则抛出 ValueError。"""
    web_section = credentials.get("web")
    if not isinstance(web_section, dict):
        raise ValueError("Google OAuth JSON must include a 'web' client configuration to use browser-based authorization.")
    return {"web": web_section}


def _exchange_google_web_oauth_code(
    client_config: dict[str, Any],
    scopes: list[str],
    redirect_uri: str,
    code: str,
    code_verifier: str | None,
) -> Flow:
    """用 Google 授权码换取 OAuth token，返回已授权的 Flow 对象。"""
    flow = Flow.from_client_config(client_config, scopes=scopes)
    flow.redirect_uri = redirect_uri
    fetch_token_kwargs: dict[str, Any] = {"code": code}
    if code_verifier:
        fetch_token_kwargs["code_verifier"] = code_verifier
    flow.fetch_token(**fetch_token_kwargs)
    return flow


async def _render_web_oauth_popup(flow_id: str, success: bool, message: str, source="drive"):
    """渲染 OAuth Web 流程完成后的弹出页面（成功或失败）。"""
    status = "success" if success else "error"
    auto_close = "window.close();" if success else ""
    escaped_message = escape(message)
    #   Drive: ragflow-google-drive-oauth
    #   Gmail: ragflow-gmail-oauth
    payload_type = f"ragflow-{source}-oauth"
    payload_json = json.dumps(
        {
            "type": payload_type,
            "status": status,
            "flowId": flow_id or "",
            "message": message,
        }
    )
    # TODO(google-oauth): title/heading/message may need to reflect drive/gmail based on cached type
    html = WEB_OAUTH_POPUP_TEMPLATE.format(
        title=f"Google {source.capitalize()} Authorization",
        heading="Authorization complete" if success else "Authorization failed",
        message=escaped_message,
        payload_json=payload_json,
        auto_close=auto_close,
    )
    response = await make_response(html, 200)
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


@manager.route("/connectors/google/oauth/web/start", methods=["POST"])  # noqa: F821
@login_required
@validate_request("credentials")
async def start_google_web_oauth():
    """启动 Google OAuth Web 授权流程，返回授权 URL。

    支持 google-drive 和 gmail 两种类型。
    将 OAuth 中间状态缓存到 Redis，TTL 由 ``WEB_FLOW_TTL_SECS`` 控制。
    """
    source = request.args.get("type", "google-drive")
    if source not in ("google-drive", "gmail"):
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="Invalid Google OAuth type.")

    req = await get_request_json()

    if source == "gmail":
        default_redirect_uri = GMAIL_WEB_OAUTH_REDIRECT_URI
        scopes = GOOGLE_SCOPES[DocumentSource.GMAIL]
    else:
        default_redirect_uri = GOOGLE_DRIVE_WEB_OAUTH_REDIRECT_URI
        scopes = GOOGLE_SCOPES[DocumentSource.GOOGLE_DRIVE]

    redirect_uri = req.get("redirect_uri", default_redirect_uri)
    if isinstance(redirect_uri, str):
        redirect_uri = redirect_uri.strip()

    if not redirect_uri:
        return get_json_result(
            code=RetCode.SERVER_ERROR,
            message="Google OAuth redirect URI is not configured on the server.",
        )

    raw_credentials = req.get("credentials", "")

    try:
        credentials = _load_credentials(raw_credentials)
        print(credentials)
    except ValueError as exc:
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message=str(exc))

    if credentials.get("refresh_token"):
        return get_json_result(
            code=RetCode.ARGUMENT_ERROR,
            message="Uploaded credentials already include a refresh token.",
        )

    try:
        client_config = _get_web_client_config(credentials)
    except ValueError as exc:
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message=str(exc))

    flow_id = str(uuid.uuid4())
    try:
        flow = Flow.from_client_config(client_config, scopes=scopes)
        flow.redirect_uri = redirect_uri
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=flow_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to create Google OAuth flow: %s", exc)
        return get_json_result(
            code=RetCode.SERVER_ERROR,
            message="Failed to initialize Google OAuth flow. Please verify the uploaded client configuration.",
        )

    cache_payload = {
        "user_id": current_user.id,
        "client_config": client_config,
        "redirect_uri": redirect_uri,
        "code_verifier": flow.code_verifier,
        "created_at": int(time.time()),
    }
    REDIS_CONN.set_obj(_web_state_cache_key(flow_id, source), cache_payload, WEB_FLOW_TTL_SECS)

    return get_json_result(
        data={
            "flow_id": flow_id,
            "authorization_url": authorization_url,
            "expires_in": WEB_FLOW_TTL_SECS,
        }
    )


@manager.route("/connectors/gmail/oauth/web/callback", methods=["GET"])  # noqa: F821
async def google_gmail_web_oauth_callback():
    """Google Gmail OAuth 回调端点，用授权码换取 token 并存入 Redis。"""
    state_id = request.args.get("state")
    error = request.args.get("error")
    source = "gmail"

    error_description = request.args.get("error_description") or error

    if not state_id:
        return await _render_web_oauth_popup("", False, "Missing OAuth state parameter.", source)

    state_cache = REDIS_CONN.get(_web_state_cache_key(state_id, source))
    if not state_cache:
        return await _render_web_oauth_popup(state_id, False, "Authorization session expired. Please restart from the main window.", source)

    state_obj = json.loads(state_cache)
    client_config = state_obj.get("client_config")
    redirect_uri = state_obj.get("redirect_uri", GMAIL_WEB_OAUTH_REDIRECT_URI)
    code_verifier = state_obj.get("code_verifier")
    if not client_config:
        REDIS_CONN.delete(_web_state_cache_key(state_id, source))
        return await _render_web_oauth_popup(state_id, False, "Authorization session was invalid. Please retry.", source)

    if error:
        REDIS_CONN.delete(_web_state_cache_key(state_id, source))
        return await _render_web_oauth_popup(state_id, False, error_description or "Authorization was cancelled.", source)

    code = request.args.get("code")
    if not code:
        return await _render_web_oauth_popup(state_id, False, "Missing authorization code from Google.", source)

    try:
        flow = _exchange_google_web_oauth_code(
            client_config=client_config,
            scopes=GOOGLE_SCOPES[DocumentSource.GMAIL],
            redirect_uri=redirect_uri,
            code=code,
            code_verifier=code_verifier,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to exchange Google OAuth code: %s", exc)
        REDIS_CONN.delete(_web_state_cache_key(state_id, source))
        return await _render_web_oauth_popup(state_id, False, "Failed to exchange tokens with Google. Please retry.", source)

    creds_json = flow.credentials.to_json()
    result_payload = {
        "user_id": state_obj.get("user_id"),
        "credentials": creds_json,
    }
    REDIS_CONN.set_obj(_web_result_cache_key(state_id, source), result_payload, WEB_FLOW_TTL_SECS)
    REDIS_CONN.delete(_web_state_cache_key(state_id, source))

    return await _render_web_oauth_popup(state_id, True, "Authorization completed successfully.", source)


@manager.route("/connectors/google-drive/oauth/web/callback", methods=["GET"])  # noqa: F821
async def google_drive_web_oauth_callback():
    """Google Drive OAuth 回调端点，用授权码换取 token 并存入 Redis。"""
    state_id = request.args.get("state")
    error = request.args.get("error")
    source = "google-drive"

    error_description = request.args.get("error_description") or error

    if not state_id:
        return await _render_web_oauth_popup("", False, "Missing OAuth state parameter.", source)

    state_cache = REDIS_CONN.get(_web_state_cache_key(state_id, source))
    if not state_cache:
        return await _render_web_oauth_popup(state_id, False, "Authorization session expired. Please restart from the main window.", source)

    state_obj = json.loads(state_cache)
    client_config = state_obj.get("client_config")
    redirect_uri = state_obj.get("redirect_uri", GOOGLE_DRIVE_WEB_OAUTH_REDIRECT_URI)
    code_verifier = state_obj.get("code_verifier")
    if not client_config:
        REDIS_CONN.delete(_web_state_cache_key(state_id, source))
        return await _render_web_oauth_popup(state_id, False, "Authorization session was invalid. Please retry.", source)

    if error:
        REDIS_CONN.delete(_web_state_cache_key(state_id, source))
        return await _render_web_oauth_popup(state_id, False, error_description or "Authorization was cancelled.", source)

    code = request.args.get("code")
    if not code:
        return await _render_web_oauth_popup(state_id, False, "Missing authorization code from Google.", source)

    try:
        flow = _exchange_google_web_oauth_code(
            client_config=client_config,
            scopes=GOOGLE_SCOPES[DocumentSource.GOOGLE_DRIVE],
            redirect_uri=redirect_uri,
            code=code,
            code_verifier=code_verifier,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to exchange Google OAuth code: %s", exc)
        REDIS_CONN.delete(_web_state_cache_key(state_id, source))
        return await _render_web_oauth_popup(state_id, False, "Failed to exchange tokens with Google. Please retry.", source)

    creds_json = flow.credentials.to_json()
    result_payload = {
        "user_id": state_obj.get("user_id"),
        "credentials": creds_json,
    }
    REDIS_CONN.set_obj(_web_result_cache_key(state_id, source), result_payload, WEB_FLOW_TTL_SECS)
    REDIS_CONN.delete(_web_state_cache_key(state_id, source))

    return await _render_web_oauth_popup(state_id, True, "Authorization completed successfully.", source)

@manager.route("/connectors/google/oauth/web/result", methods=["POST"])  # noqa: F821
@login_required
@validate_request("flow_id")
async def poll_google_web_result():
    """前端轮询此端点，获取 Google OAuth 授权完成后的凭据。

    从 Redis 读取回调阶段写入的结果，校验用户身份后返回凭据并删除缓存。
    """
    req = await request.json or {}
    source = request.args.get("type")
    if source not in ("google-drive", "gmail"):
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="Invalid Google OAuth type.")
    flow_id = req.get("flow_id")
    cache_raw = REDIS_CONN.get(_web_result_cache_key(flow_id, source))
    if not cache_raw:
        return get_json_result(code=RetCode.RUNNING, message="Authorization is still pending.")

    result = json.loads(cache_raw)
    if result.get("user_id") != current_user.id:
        return get_json_result(code=RetCode.PERMISSION_ERROR, message="You are not allowed to access this authorization result.")

    REDIS_CONN.delete(_web_result_cache_key(flow_id, source))
    return get_json_result(data={"credentials": result.get("credentials")})

@manager.route("/connectors/box/oauth/web/start", methods=["POST"])  # noqa: F821
@login_required
async def start_box_web_oauth():
    """启动 Box OAuth Web 授权流程，返回 Box 授权 URL。

    需要客户端提供 client_id 和 client_secret。
    将 OAuth 中间状态缓存到 Redis。
    """
    req = await get_request_json()

    client_id = req.get("client_id")
    client_secret = req.get("client_secret")    
    redirect_uri = req.get("redirect_uri", BOX_WEB_OAUTH_REDIRECT_URI)

    if not client_id or not client_secret:
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="Box client_id and client_secret are required.")

    flow_id = str(uuid.uuid4())

    box_auth = BoxOAuth(
        OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
        )
    )

    auth_url = box_auth.get_authorize_url(
        options=GetAuthorizeUrlOptions(
            redirect_uri=redirect_uri,
            state=flow_id,
        )
    )

    cache_payload = {
        "user_id": current_user.id,
        "auth_url": auth_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "created_at": int(time.time()),
    }
    REDIS_CONN.set_obj(_web_state_cache_key(flow_id, "box"), cache_payload, WEB_FLOW_TTL_SECS)
    return get_json_result(
        data = {
            "flow_id": flow_id,
            "authorization_url": auth_url,
            "expires_in": WEB_FLOW_TTL_SECS,}
    )

@manager.route("/connectors/box/oauth/web/callback", methods=["GET"])  # noqa: F821
async def box_web_oauth_callback():
    """Box OAuth 回调端点，用授权码换取 access_token/refresh_token 并存入 Redis。"""
    flow_id = request.args.get("state")
    if not flow_id:
        return await _render_web_oauth_popup("", False, "Missing OAuth parameters.", "box")
    
    code = request.args.get("code")
    if not code:
        return await _render_web_oauth_popup(flow_id, False, "Missing authorization code from Box.", "box")

    cache_payload = json.loads(REDIS_CONN.get(_web_state_cache_key(flow_id, "box")))
    if not cache_payload:
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="Box OAuth session expired or invalid.")

    error = request.args.get("error")
    error_description = request.args.get("error_description") or error
    if error:
        REDIS_CONN.delete(_web_state_cache_key(flow_id, "box"))
        return await _render_web_oauth_popup(flow_id, False, error_description or "Authorization failed.", "box")
    
    auth = BoxOAuth(
        OAuthConfig(
            client_id=cache_payload.get("client_id"),
            client_secret=cache_payload.get("client_secret"),
        )
    )

    auth.get_tokens_authorization_code_grant(code)
    token = auth.retrieve_token()
    result_payload = {
        "user_id": cache_payload.get("user_id"),
        "client_id": cache_payload.get("client_id"),
        "client_secret": cache_payload.get("client_secret"),
        "access_token": token.access_token,
        "refresh_token": token.refresh_token,
    }

    REDIS_CONN.set_obj(_web_result_cache_key(flow_id, "box"), result_payload, WEB_FLOW_TTL_SECS)
    REDIS_CONN.delete(_web_state_cache_key(flow_id, "box"))

    return await _render_web_oauth_popup(flow_id, True, "Authorization completed successfully.", "box")

@manager.route("/connectors/box/oauth/web/result", methods=["POST"])  # noqa: F821
@login_required
@validate_request("flow_id")
async def poll_box_web_result():
    """前端轮询此端点，获取 Box OAuth 授权完成后的凭据。

    从 Redis 读取回调阶段写入的结果，校验用户身份后返回凭据并删除缓存。
    """
    req = await get_request_json()
    flow_id = req.get("flow_id")

    cache_blob = REDIS_CONN.get(_web_result_cache_key(flow_id, "box"))
    if not cache_blob:
        return get_json_result(code=RetCode.RUNNING, message="Authorization is still pending.")

    cache_raw = json.loads(cache_blob)
    if cache_raw.get("user_id") != current_user.id:
        return get_json_result(code=RetCode.PERMISSION_ERROR, message="You are not allowed to access this authorization result.")
    
    REDIS_CONN.delete(_web_result_cache_key(flow_id, "box"))

    return get_json_result(data={"credentials": cache_raw})
