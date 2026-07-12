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
聊天渠道（Chat Channel）API 模块

提供聊天渠道机器人的 CRUD 接口，支持将 RAGFlow 对话助手对接到外部平台：
- POST: 创建聊天渠道（如 Telegram Bot、Discord Bot 等）
- GET: 列表查询 / 单个查询
- PATCH: 更新渠道配置（名称、状态、关联对话等）
- DELETE: 删除渠道

每个渠道关联一个 dialog_id（对话助手），通过 channel 字段区分平台类型，
config 字段存储平台特定配置（如 Bot Token、Webhook URL 等）。
"""

import logging

from api.apps import current_user, login_required
from api.db.services.chat_channel_service import ChatChannelService
from api.db.services.dialog_service import DialogService
from api.utils.api_utils import get_data_error_result, get_json_result, get_request_json, validate_request
from common.constants import RetCode
from common.misc_utils import get_uuid

LOGGER = logging.getLogger(__name__)


def _chat_channel_auth_error(channel_id: str, user_id: str):
    """聊天渠道授权失败的统一响应，同时记录警告日志。"""
    LOGGER.warning("chat channel access denied: channel_id=%s user_id=%s", channel_id, user_id)
    return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)


@manager.route("/chat_channels", methods=["POST"])  # noqa: F821
@login_required
@validate_request("name", "channel", "config")
async def create_chat_channel():
    """创建聊天渠道机器人。

    必填字段：name（名称）、channel（渠道类型）、config（平台配置）。
    可选字段：dialog_id（关联的对话助手）。
    新创建的渠道默认状态为 "1"（启用）。
    """
    req = await get_request_json()
    channel = {
        "id": get_uuid(),
        "tenant_id": current_user.id,
        "name": req["name"],
        "channel": req["channel"],
        "config": req["config"],
        "dialog_id": req.get("dialog_id") or None,
        "status": "1",
    }
    ChatChannelService.insert(**channel)

    e, conn = ChatChannelService.get_by_id(channel["id"])
    if not e:
        return get_data_error_result(message="Failed to create chat channel!")
    return get_json_result(data=conn.to_dict())


@manager.route("/chat_channels", methods=["GET"])  # noqa: F821
@login_required
def list_chat_channel():
    """列出当前租户所有聊天渠道。"""
    return get_json_result(data=ChatChannelService.list(current_user.id))


@manager.route("/chat_channels/<channel_id>", methods=["GET"])  # noqa: F821
@login_required
def get_chat_channel(channel_id):
    """查看指定聊天渠道的详细信息（需有访问权限）。"""
    if not ChatChannelService.accessible(channel_id, current_user.id):
        return _chat_channel_auth_error(channel_id, current_user.id)

    e, conn = ChatChannelService.get_by_id(channel_id)
    if not e:
        return get_data_error_result(message="Can't find this chat channel!")
    return get_json_result(data=conn.to_dict())


@manager.route("/chat_channels/<channel_id>", methods=["PATCH"])  # noqa: F821
@login_required
async def update_chat_channel(channel_id):
    """更新聊天渠道的 name / config / dialog_id / status。

    安全校验：
    1. 用户必须有权访问该渠道
    2. 若更新 dialog_id，验证目标对话助手属于同一租户
    """
    if not ChatChannelService.accessible(channel_id, current_user.id):
        return _chat_channel_auth_error(channel_id, current_user.id)

    e, conn = ChatChannelService.get_by_id(channel_id)
    if not e:
        return get_data_error_result(message="Can't find this chat channel!")

    req = await get_request_json()
    if isinstance(req, dict) and isinstance(req.get("data"), dict):
        req = req["data"]

    # 若提供了 dialog_id，验证对话助手是否属于当前渠道的租户
    if req.get("dialog_id"):
        e, dia = DialogService.get_by_id(req["dialog_id"])
        if not e:
            return get_data_error_result(message="Can't find this chat assistant!")
        if dia.tenant_id != conn.tenant_id:
            return _chat_channel_auth_error(channel_id, current_user.id)

    # 只更新请求中实际提供的字段
    update_fields = {fld: req[fld] for fld in ["name", "config", "dialog_id", "status"] if fld in req}
    if update_fields:
        ChatChannelService.update_by_id(channel_id, update_fields)

    e, conn = ChatChannelService.get_by_id(channel_id)
    if not e:
        return get_data_error_result(message="Can't find this chat channel!")
    return get_json_result(data=conn.to_dict())


@manager.route("/chat_channels/<channel_id>", methods=["DELETE"])  # noqa: F821
@login_required
def rm_chat_channel(channel_id):
    """删除指定的聊天渠道（需有访问权限）。"""
    if not ChatChannelService.accessible(channel_id, current_user.id):
        return _chat_channel_auth_error(channel_id, current_user.id)

    ChatChannelService.delete_by_id(channel_id)
    return get_json_result(data=True)
