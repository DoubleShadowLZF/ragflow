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
Langfuse API 密钥管理模块

提供租户级别的 Langfuse API 密钥 CRUD 接口：
- POST/PUT: 创建或更新 Langfuse 密钥（需验证密钥有效性）
- GET: 查询已配置的 Langfuse 密钥及关联的项目信息
- DELETE: 删除已配置的 Langfuse 密钥

Langfuse 是一个开源 LLM 可观测性平台，RAGFlow 集成它来追踪 Agent 调用链路和 Token 消耗。
"""


from api.apps import current_user, login_required
from langfuse import Langfuse

from api.db.db_models import DB
from api.db.services.langfuse_service import TenantLangfuseService
from api.utils.api_utils import get_error_data_result, get_json_result, get_request_json, server_error_response, validate_request


@manager.route("/langfuse/api-key", methods=["POST", "PUT"])  # noqa: F821
@login_required
@validate_request("secret_key", "public_key", "host")
async def set_api_key():
    """创建或更新当前租户的 Langfuse API 密钥。

    流程：
    1. 校验必填字段（secret_key、public_key、host）
    2. 使用提供的密钥向 Langfuse 发起认证检查
    3. 若认证通过，写入或更新数据库（使用事务保证原子性）
    """
    req = await get_request_json()
    secret_key = req.get("secret_key", "")
    public_key = req.get("public_key", "")
    host = req.get("host", "")
    if not all([secret_key, public_key, host]):
        return get_error_data_result(message="Missing required fields")

    current_user_id = current_user.id
    langfuse_keys = dict(
        tenant_id=current_user_id,
        secret_key=secret_key,
        public_key=public_key,
        host=host,
    )

    # 先向 Langfuse 验证密钥是否有效
    langfuse = Langfuse(public_key=langfuse_keys["public_key"], secret_key=langfuse_keys["secret_key"], host=langfuse_keys["host"])
    if not langfuse.auth_check():
        return get_error_data_result(message="Invalid Langfuse keys")

    # 数据库事务：不存在则创建，已存在则更新
    langfuse_entry = TenantLangfuseService.filter_by_tenant(tenant_id=current_user_id)
    with DB.atomic():
        try:
            if not langfuse_entry:
                TenantLangfuseService.save(**langfuse_keys)
            else:
                TenantLangfuseService.update_by_tenant(tenant_id=current_user_id, langfuse_keys=langfuse_keys)
            return get_json_result(data=langfuse_keys)
        except Exception as e:
            return server_error_response(e)


@manager.route("/langfuse/api-key", methods=["GET"])  # noqa: F821
@login_required
@validate_request()
def get_api_key():
    """查询当前租户已配置的 Langfuse API 密钥。

    不仅返回密钥信息，还会：
    1. 验证密钥是否仍然有效
    2. 查询关联的 Langfuse 项目 ID 和名称
    """
    current_user_id = current_user.id
    langfuse_entry = TenantLangfuseService.filter_by_tenant_with_info(tenant_id=current_user_id)
    if not langfuse_entry:
        return get_json_result(message="Have not record any Langfuse keys.")

    langfuse = Langfuse(public_key=langfuse_entry["public_key"], secret_key=langfuse_entry["secret_key"], host=langfuse_entry["host"])
    try:
        if not langfuse.auth_check():
            return get_error_data_result(message="Invalid Langfuse keys loaded")
    except langfuse.api.core.api_error.ApiError as api_err:
        return get_json_result(message=f"Error from Langfuse: {api_err}")
    except Exception as e:
        return server_error_response(e)

    # 从 Langfuse API 获取关联的项目信息
    langfuse_entry["project_id"] = langfuse.api.projects.get().dict()["data"][0]["id"]
    langfuse_entry["project_name"] = langfuse.api.projects.get().dict()["data"][0]["name"]

    return get_json_result(data=langfuse_entry)


@manager.route("/langfuse/api-key", methods=["DELETE"])  # noqa: F821
@login_required
@validate_request()
def delete_api_key():
    """删除当前租户已配置的 Langfuse API 密钥。"""
    current_user_id = current_user.id
    langfuse_entry = TenantLangfuseService.filter_by_tenant(tenant_id=current_user_id)
    if not langfuse_entry:
        return get_json_result(message="Have not record any Langfuse keys.")

    with DB.atomic():
        try:
            TenantLangfuseService.delete_model(langfuse_entry)
            return get_json_result(data=True)
        except Exception as e:
            return server_error_response(e)
