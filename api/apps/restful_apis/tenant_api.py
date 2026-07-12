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
import asyncio
import logging
from typing import Set

from api.apps import current_user, login_required
from api.db import UserTenantRole
from api.db.db_models import UserTenant
from api.db.services.user_service import UserService, UserTenantService
from api.utils.api_utils import (
    get_data_error_result,
    get_json_result,
    get_request_json,
    server_error_response,
    validate_request,
)
from api.utils.web_utils import send_invite_email
from common import settings
from common.constants import RetCode, StatusEnum
from common.misc_utils import get_uuid
from common.time_utils import delta_seconds

# 保存对"即发即忘"（fire-and-forget）异步任务的强引用，防止任务在完成前被垃圾回收。
_background_tasks: Set[asyncio.Task] = set()


@manager.route("/tenants/<tenant_id>/users", methods=["GET"])  # noqa: F821
@login_required
def user_list(tenant_id):
    """获取指定租户下的所有用户列表。

    仅允许租户所有者（tenant_id 等于当前用户 ID）查看。
    返回用户列表时，会为每个用户附加上次更新时间距现在的秒数（delta_seconds）。
    """
    # 权限校验：只有租户所有者才能查看成员列表
    if current_user.id != tenant_id:
        return get_json_result(
            data=False,
            message="No authorization.",
            code=RetCode.AUTHENTICATION_ERROR,
        )

    try:
        users = UserTenantService.get_by_tenant_id(tenant_id)
        for user in users:
            user["delta_seconds"] = delta_seconds(str(user["update_date"]))
        return get_json_result(data=users)
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/tenants/<tenant_id>/users", methods=["POST"])  # noqa: F821
@login_required
@validate_request("email")
async def create(tenant_id):
    """邀请用户加入租户（团队）。

    流程：
    1. 校验当前用户是否为租户所有者。
    2. 根据邮箱查找被邀请用户，检查是否已在团队中。
    3. 在 user_tenant 表中创建 INVITE 状态的记录。
    4. 异步发送邀请邮件（fire-and-forget 模式，不阻塞响应）。
    """
    # 权限校验：只有租户所有者才能邀请
    if current_user.id != tenant_id:
        return get_json_result(
            data=False,
            message="No authorization.",
            code=RetCode.AUTHENTICATION_ERROR,
        )

    req = await get_request_json()
    invite_user_email = req["email"]
    invite_users = UserService.query(email=invite_user_email)
    if not invite_users:
        return get_data_error_result(message="User not found.")

    user_id_to_invite = invite_users[0].id
    # 检查被邀请用户与租户的现有关系
    user_tenants = UserTenantService.query(user_id=user_id_to_invite, tenant_id=tenant_id)
    if user_tenants:
        user_tenant_role = user_tenants[0].role
        if user_tenant_role == UserTenantRole.NORMAL:
            return get_data_error_result(message=f"{invite_user_email} is already in the team.")
        if user_tenant_role == UserTenantRole.OWNER:
            return get_data_error_result(message=f"{invite_user_email} is the owner of the team.")
        return get_data_error_result(
            message=f"{invite_user_email} is in the team, but the role: {user_tenant_role} is invalid."
        )

    # 创建 INVITE 状态的租户-用户关联记录
    UserTenantService.save(
        id=get_uuid(),
        user_id=user_id_to_invite,
        tenant_id=tenant_id,
        invited_by=current_user.id,
        role=UserTenantRole.INVITE,
        status=StatusEnum.VALID.value,
    )

    # 异步发送邀请邮件（fire-and-forget 模式）
    try:
        user_name = ""
        _, user = UserService.get_by_id(current_user.id)
        if user:
            user_name = user.nickname

        def _on_invite_email_done(done_task: asyncio.Task) -> None:
            """邀请邮件任务的完成回调：清理后台任务集合并处理异常。"""
            _background_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                logging.warning("Invite email task cancelled: tenant_id=%s to=%s", tenant_id, invite_user_email)
            except Exception:
                logging.exception("Invite email task failed: tenant_id=%s to=%s", tenant_id, invite_user_email)

        task = asyncio.create_task(
            send_invite_email(
                to_email=invite_user_email,
                invite_url=settings.MAIL_FRONTEND_URL,
                tenant_id=tenant_id,
                inviter=user_name or current_user.email,
            )
        )
        if isinstance(task, asyncio.Task):
            _background_tasks.add(task)
            task.add_done_callback(_on_invite_email_done)
    except Exception as exc:
        logging.exception(f"Failed to send invite email to {invite_user_email}: {exc}")
        return get_json_result(
            data=False,
            message="Failed to send invite email.",
            code=RetCode.SERVER_ERROR,
        )

    # 返回被邀请用户的基本信息
    user = invite_users[0].to_dict()
    user = {k: v for k, v in user.items() if k in ["id", "avatar", "email", "nickname"]}
    return get_json_result(data=user)


@manager.route("/tenants/<tenant_id>/users", methods=["DELETE"])  # noqa: F821
@login_required
@validate_request("user_id")
async def rm(tenant_id):
    """从租户中移除用户。

    允许两种角色操作：
    - 租户所有者（current_user.id == tenant_id）可以移除任何成员
    - 用户本人（current_user.id == user_id）可以主动退出团队
    """
    req = await get_request_json()
    user_id = req["user_id"]
    # 权限校验：租户所有者或用户本人才可移除
    if current_user.id != tenant_id and current_user.id != user_id:
        return get_json_result(
            data=False,
            message="No authorization.",
            code=RetCode.AUTHENTICATION_ERROR,
        )

    try:
        UserTenantService.filter_delete([UserTenant.tenant_id == tenant_id, UserTenant.user_id == user_id])
        return get_json_result(data=True)
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/tenants", methods=["GET"])  # noqa: F821
@login_required
def tenant_list():
    """获取当前用户所属的所有租户列表。

    返回每个租户的基本信息及上次更新时间距现在的秒数（delta_seconds）。
    """
    try:
        users = UserTenantService.get_tenants_by_user_id(current_user.id)
        for user in users:
            user["delta_seconds"] = delta_seconds(str(user["update_date"]))
        return get_json_result(data=users)
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/tenants/<tenant_id>", methods=["PATCH"])  # noqa: F821
@login_required
def agree(tenant_id):
    """当前用户同意加入指定租户。

    将用户在租户中的角色从 INVITE（受邀）变更为 NORMAL（正式成员），
    即完成邀请确认流程。
    """
    try:
        # 将角色从 INVITE 更新为 NORMAL，表示接受邀请
        UserTenantService.filter_update(
            [UserTenant.tenant_id == tenant_id, UserTenant.user_id == current_user.id],
            {"role": UserTenantRole.NORMAL},
        )
        return get_json_result(data=True)
    except Exception as exc:
        return server_error_response(exc)
