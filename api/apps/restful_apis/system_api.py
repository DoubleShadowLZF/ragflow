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

"""
系统管理 API 模块

提供系统级别的 RESTful 接口，主要包括以下功能：
- 基础健康检查：ping 心跳、healthz 健康探测、status 组件状态汇总
- 系统配置管理：注册开关、密码登录开关、日志级别动态调整
- API Token 管理：创建、查询、删除用户的 API 访问令牌
- 数据库状态监控：OceanBase 等数据库的健康与性能指标
"""

import json
import logging
from datetime import datetime
from timeit import default_timer as timer

from quart import jsonify

from api.apps import login_required, current_user
from api.utils.api_utils import get_json_result, get_data_error_result, server_error_response, generate_confirmation_token
from api.utils.health_utils import run_health_checks, get_oceanbase_status
from common.versions import get_ragflow_version
from common.time_utils import current_timestamp, datetime_format
from api.db.db_models import APIToken
from api.db.services.api_service import APITokenService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.user_service import UserTenantService
from common.log_utils import get_log_levels, set_log_level
from common import settings
from rag.utils.redis_conn import REDIS_CONN

# =============================================================================
#  基础健康检查接口
# =============================================================================

@manager.route("/system/ping", methods=["GET"])  # noqa: F821
async def ping():
    """心跳检测接口，返回 "pong" 表示 HTTP 服务正常运行"""
    return "pong", 200

@manager.route("/system/version", methods=["GET"])  # noqa: F821
def version():
    """
    获取 RAGFlow 当前版本号。
    ---
    tags:
      - System
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Version retrieved successfully.
        schema:
          type: object
          properties:
            version:
              type: string
              description: Version number.
    """
    return get_json_result(data=get_ragflow_version())


@manager.route("/system/status", methods=["GET"])  # noqa: F821
@login_required
def status():
    """
    获取系统各组件的运行状态，包括文档检索引擎、对象存储、数据库、
    Redis 缓存以及任务执行器心跳信息。
    ---
    tags:
      - System
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: System is operational.
        schema:
          type: object
          properties:
            es:
              type: object
              description: Elasticsearch status.
            storage:
              type: object
              description: Storage status.
            database:
              type: object
              description: Database status.
      503:
        description: Service unavailable.
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message.
    """
    res = {}
    # 检查文档检索引擎（Elasticsearch / Infinity）健康状态
    st = timer()
    try:
        res["doc_engine"] = settings.docStoreConn.health()
        res["doc_engine"]["elapsed"] = "{:.1f}".format((timer() - st) * 1000.0)
    except Exception as e:
        res["doc_engine"] = {
            "type": "unknown",
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    # 检查对象存储（MinIO / S3 等）健康状态
    st = timer()
    try:
        settings.STORAGE_IMPL.health()
        res["storage"] = {
            "storage": settings.STORAGE_IMPL_TYPE.lower(),
            "status": "green",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
        }
    except Exception as e:
        res["storage"] = {
            "storage": settings.STORAGE_IMPL_TYPE.lower(),
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    # 检查数据库（MySQL / PostgreSQL）连接是否正常
    st = timer()
    try:
        KnowledgebaseService.get_by_id("x")
        res["database"] = {
            "database": settings.DATABASE_TYPE.lower(),
            "status": "green",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
        }
    except Exception as e:
        res["database"] = {
            "database": settings.DATABASE_TYPE.lower(),
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    # 检查 Redis 连接是否正常
    st = timer()
    try:
        if not REDIS_CONN.health():
            raise Exception("Lost connection!")
        res["redis"] = {
            "status": "green",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
        }
    except Exception as e:
        res["redis"] = {
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    # 获取任务执行器（Task Executor）的心跳信息
    # 从 Redis 集合中读取所有在线的执行器实例，并查询最近 30 分钟的心跳记录
    task_executor_heartbeats = {}
    try:
        task_executors = REDIS_CONN.smembers("TASKEXE")
        now = datetime.now().timestamp()
        for task_executor_id in task_executors:
            heartbeats = REDIS_CONN.zrangebyscore(task_executor_id, now - 60 * 30, now)
            heartbeats = [json.loads(heartbeat) for heartbeat in heartbeats]
            task_executor_heartbeats[task_executor_id] = heartbeats
    except Exception:
        logging.exception("get task executor heartbeats failed!")
    res["task_executor_heartbeats"] = task_executor_heartbeats

    return get_json_result(data=res)


# =============================================================================
#  数据库状态监控
# =============================================================================

@manager.route("/system/oceanbase/status", methods=["GET"])  # noqa: F821
@login_required
def oceanbase_status():
    """
    获取 OceanBase 数据库的健康状态和性能指标，包括连接状态、响应时间等。
    ---
    tags:
      - System
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: OceanBase status retrieved successfully.
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status (alive/timeout).
            message:
              type: object
              description: Detailed status information including health and performance metrics.
    """
    try:
        status_info = get_oceanbase_status()
        return get_json_result(data=status_info)
    except Exception as e:
        return get_json_result(
            data={
                "status": "error",
                "message": f"Failed to get OceanBase status: {str(e)}"
            },
            code=500
        )


# =============================================================================
#  系统配置接口
# =============================================================================

@manager.route("/system/config", methods=["GET"])  # noqa: F821
def get_config():
    """
    获取系统配置信息，当前返回用户注册开关和密码登录开关的状态。
    ---
    tags:
        - System
    responses:
        200:
            description: Return system configuration
            schema:
                type: object
                properties:
                    registerEnable:
                        type: integer 0 means disabled, 1 means enabled
                        description: Whether user registration is enabled
    """
    return get_json_result(data={
        "registerEnabled": settings.REGISTER_ENABLED,
        "disablePasswordLogin": settings.DISABLE_PASSWORD_LOGIN,
    })

@manager.route("/system/healthz", methods=["GET"])  # noqa: F821
def healthz():
    """Kubernetes 风格的健康探测接口，返回所有组件的健康检查结果。
    若全部通过则返回 200，否则返回 500。"""
    result, all_ok = run_health_checks()
    return jsonify(result), (200 if all_ok else 500)

# =============================================================================
#  API Token 管理接口
# =============================================================================

@manager.route("/system/tokens", methods=["GET"])  # noqa: F821
@login_required
def token_list():
    """
    获取当前用户的所有 API 访问令牌列表。
    ---
    tags:
      - API Tokens
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: List of API tokens.
        schema:
          type: object
          properties:
            tokens:
              type: array
              items:
                type: object
                properties:
                  token:
                    type: string
                    description: The API token.
                  name:
                    type: string
                    description: Name of the token.
                  create_time:
                    type: string
                    description: Token creation time.
    """
    try:
        # 查找当前用户所属的租户，需具备 owner 角色
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")

        tenant_id = [tenant for tenant in tenants if tenant.role == "owner"][0].tenant_id
        objs = APITokenService.query(tenant_id=tenant_id)
        objs = [o.to_dict() for o in objs]
        # 为没有 beta 字段的旧 token 补充生成 beta 值，保证向后兼容
        for o in objs:
            if not o["beta"]:
                o["beta"] = generate_confirmation_token().replace("ragflow-", "")[:32]
                APITokenService.filter_update([APIToken.tenant_id == tenant_id, APIToken.token == o["token"]], o)
        return get_json_result(data=objs)
    except Exception as e:
        return server_error_response(e)


@manager.route("/system/tokens", methods=["POST"])  # noqa: F821
@login_required
def new_token():
    """
    为当前用户生成一个新的 API 访问令牌。令牌包含 token 和 beta 两个字段，
    beta 用于未来功能的灰度发布控制。
    ---
    tags:
      - API Tokens
    security:
      - ApiKeyAuth: []
    parameters:
      - in: query
        name: name
        type: string
        required: false
        description: Name of the token.
    responses:
      200:
        description: Token generated successfully.
        schema:
          type: object
          properties:
            token:
              type: string
              description: The generated API token.
    """
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")

        tenant_id = [tenant for tenant in tenants if tenant.role == "owner"][0].tenant_id
        # 构建 token 对象，包含双令牌机制：token 用于认证，beta 用于灰度控制
        obj = {
            "tenant_id": tenant_id,
            "token": generate_confirmation_token(),
            "beta": generate_confirmation_token().replace("ragflow-", "")[:32],
            "create_time": current_timestamp(),
            "create_date": datetime_format(datetime.now()),
            "update_time": None,
            "update_date": None,
        }

        if not APITokenService.save(**obj):
            return get_data_error_result(message="Fail to new a dialog!")

        return get_json_result(data=obj)
    except Exception as e:
        return server_error_response(e)


@manager.route("/system/tokens/<token>", methods=["DELETE"])  # noqa: F821
@login_required
def rm(token):
    """
    删除指定的 API 访问令牌。只有令牌所属租户的管理员才能执行删除操作。
    ---
    tags:
      - API Tokens
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: token
        type: string
        required: true
        description: The API token to remove.
    responses:
      200:
        description: Token removed successfully.
        schema:
          type: object
          properties:
            success:
              type: boolean
              description: Deletion status.
    """
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")

        tenant_id = tenants[0].tenant_id
        APITokenService.filter_delete([APIToken.tenant_id == tenant_id, APIToken.token == token])
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


# =============================================================================
#  日志级别动态配置接口
# =============================================================================

@manager.route("/system/config/log", methods=["GET"])  # noqa: F821
@login_required
async def get_logger_levels():
    """
    获取当前所有包的日志级别配置。
    ---
    tags:
        - System
    responses:
        200:
            description: Return current log levels
    """
    return get_json_result(data=get_log_levels())


@manager.route("/system/config/log", methods=["PUT"])  # noqa: F821
@login_required
async def set_logger_level():
    """
    动态设置指定包的日志级别（DEBUG / INFO / WARNING / ERROR），
    无需重启服务即可生效，方便线上问题排查。
    ---
    tags:
        - System
    parameters:
        - in: body
          name: body
          required: true
          schema:
            type: object
            properties:
                pkg_name:
                    type: string
                    description: Package name (e.g., "rag.utils.es_conn")
                level:
                    type: string
                    description: Log level (DEBUG, INFO, WARNING, ERROR)
    responses:
        200:
            description: Log level updated successfully
    """
    from quart import request
    data = await request.get_json()
    if not data or "pkg_name" not in data or "level" not in data:
        return get_data_error_result(message="pkg_name and level are required")
    pkg_name = data["pkg_name"]
    level = data["level"]
    success = set_log_level(pkg_name, level)
    if success:
        return get_json_result(data={"pkg_name": pkg_name, "level": level})
    else:
        return get_data_error_result(message=f"Invalid log level: {level}")
