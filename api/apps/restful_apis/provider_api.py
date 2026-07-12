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
Provider（模型供应商）管理 API 蓝图。

本模块提供模型供应商及其实例、模型的完整 CRUD RESTful 接口，包括：
- 供应商（Provider）的列表/详情/添加/删除
- 模型实例（Instance）的创建/查询/删除
- 模型中具体模型（Model）的添加/启用禁用/聊天测试
- API Key 连接验证

所有接口挂载在 ``/api/v1/providers`` 路径下，需要登录认证。
"""

import logging

from quart import request

from api.apps import login_required
from api.utils.api_utils import (
    add_tenant_id_to_kwargs,
    get_error_argument_result,
    get_error_data_result,
    get_result,
)
from api.apps.services import provider_api_service


# =============================================================================
# Provider（供应商）基础 CRUD
# =============================================================================

@manager.route("/providers", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def list_providers(tenant_id: str = None):
    """获取供应商列表。

    支持两种查询模式：
    - ``available=true`` —— 列出所有系统可用的供应商（未租户化配置的）
    - 不传或传其他值 —— 列出当前租户已配置的供应商实例
    """
    available_only = request.args.get("available", "").lower() == "true"
    try:
        success, result = provider_api_service.list_providers(tenant_id, available_only)
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers", methods=["PUT"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def add_provider(tenant_id: str = None):
    """为租户添加一个供应商。

    请求体需包含 ``provider_name`` 字段，指定要添加的供应商/工厂名称。
    添加成功后该租户即可使用该供应商下的模型。
    """
    data = await request.get_json()
    if not data or "provider_name" not in data:
        return get_error_argument_result(message="provider_name is required")

    provider_name = data["provider_name"]

    try:
        success, msg = provider_api_service.add_provider(tenant_id, provider_name)
        if success:
            return get_result(message=msg)
        else:
            return get_error_data_result(message=msg)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>", methods=["GET"])  # noqa: F821
@login_required
def show_provider(provider_name: str):
    """查看供应商详情。

    返回指定供应商的配置信息、支持的能力（LLM/Embedding/Rerank 等）和元数据。
    """
    try:
        success, result = provider_api_service.show_provider(provider_name)
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>", methods=["DELETE"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def delete_provider(tenant_id: str = None, provider_name: str = None):
    """删除租户下的供应商及其所有模型。

    注意：这是一个级联操作，会同时删除该供应商下配置的所有模型实例。
    """
    try:
        success, msg = provider_api_service.delete_provider(tenant_id, provider_name)
        if success:
            return get_result(message=msg)
        else:
            return get_error_data_result(message=msg)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/models", methods=["GET"])  # noqa: F821
@login_required
async def list_provider_models(provider_name: str):
    """列出供应商的可用模型列表。

    可选传入 ``api_key`` 和 ``base_url`` 查询参数，用于从供应商 API 实时
    拉取最新模型列表（而非本地缓存）。适用于 OpenAI 兼容接口等场景。
    """
    try:
        api_key = request.args.get("api_key")
        base_url = request.args.get("base_url")
        success, result = await provider_api_service.list_provider_models(provider_name, api_key, base_url)
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/models/<path:model_name>", methods=["GET"])  # noqa: F821
@login_required
def show_provider_model(provider_name: str, model_name: str):
    """查看供应商下某个具体模型的详细信息。

    ``model_name`` 路径参数使用 ``<path:...>`` 匹配，支持包含 "/" 的模型名
    （如 "openai/gpt-4" 格式）。
    """
    try:
        success, result = provider_api_service.show_provider_model(provider_name, model_name)
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


# =============================================================================
# Provider Instance（供应商实例）CRUD
# 实例 = Provider + API Key + Base URL 的具体配置，一个租户可为同一
# 供应商创建多个实例（如指向不同的 API 端点或使用不同的密钥）。
# =============================================================================

@manager.route("/providers/<provider_name>/instances", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def create_provider_instance(tenant_id: str = None, provider_name: str = None):
    """创建供应商实例。

    实例是供应商的具体配置化体现，必填字段：
    - ``instance_name`` —— 实例名称（用于区分同一供应商的不同配置）
    - ``api_key`` —— API 密钥

    可选字段：``base_url``（自定义 API 端点）、``region``（区域）、
    ``model_info``（预配置的模型列表）。
    """
    data = await request.get_json()
    if not data or "instance_name" not in data or "api_key" not in data:
        return get_error_argument_result(message="instance_name and api_key are required")

    instance_name = data["instance_name"]
    api_key = data["api_key"]
    base_url = data.get("base_url", "")
    region = data.get("region", "")
    model_info = data.get("model_info", [])

    try:
        success, msg = await provider_api_service.create_provider_instance(tenant_id, provider_name, instance_name, api_key, base_url, region, model_info)
        if success:
            return get_result(message=msg)
        else:
            return get_error_data_result(message=msg)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/connection", methods=["POST"])  # noqa: F821
@login_required
async def verify_provider_api_key(provider_name: str = None):
    """验证供应商 API Key 是否有效。

    向供应商 API 发起实际连接请求，测试提供的 API Key 和 Base URL 是否可用。
    常用于创建实例前的连通性检查，避免保存无效的密钥配置。

    请求体至少包含 ``api_key``，可选 ``base_url`` 和 ``region``。
    """
    data = await request.get_json()
    if not data or "api_key" not in data:
        return get_error_argument_result(message="api_key is required")

    base_url = data.get("base_url", "")
    api_key = data["api_key"]
    region = data.get("region", "default")
    model_info = data.get("model_info", [])

    try:
        success, msg = await provider_api_service.verify_api_key(provider_name, api_key, base_url, region, model_info)
        if success:
            return get_result(message=msg)
        else:
            return get_error_data_result(message=msg)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/instances", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def list_provider_instances(tenant_id: str = None, provider_name: str = None):
    """列出租户在指定供应商下创建的所有实例。"""
    try:
        success, result = provider_api_service.list_provider_instances(tenant_id, provider_name)
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/instances/<instance_name>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def show_provider_instance(tenant_id: str = None, provider_name: str = None, instance_name: str = None):
    """查看指定实例的详细信息，包括 API Key（脱敏）、Base URL 等配置。"""
    try:
        success, result = provider_api_service.show_provider_instance(tenant_id, provider_name, instance_name)
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/instances", methods=["DELETE"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def drop_provider_instances(tenant_id: str = None, provider_name: str = None):
    """批量删除供应商实例。

    请求体需包含 ``instances`` 字段（实例名称列表）。
    支持一次删除多个实例，在服务层逐个处理。
    """
    data = await request.get_json()
    if not data or "instances" not in data:
        return get_error_argument_result(message="instances is required")

    instances = data["instances"]
    if not instances:
        return get_error_argument_result(message="instances is required")

    try:
        success, msg = provider_api_service.drop_provider_instances(tenant_id, provider_name, instances)
        if success:
            return get_result(message=msg)
        else:
            return get_error_data_result(message=msg)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


# =============================================================================
# Instance Model（实例内的具体模型）管理
# 在已创建的实例中，为每种模型类型（chat/embedding/rerank 等）配置
# 具体的模型名称（如 gpt-4o、text-embedding-3-small 等）。
# =============================================================================

@manager.route("/providers/<provider_name>/instances/<instance_name>/models", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def list_instance_models(tenant_id: str = None, provider_name: str = None, instance_name: str = None):
    """列出实例中已配置的模型列表。

    可选 ``supported=true`` 查询参数，过滤出标记为受支持的模型。
    """
    supported_only = request.args.get("supported", "").lower() == "true"
    try:
        success, result = provider_api_service.list_instance_models(
            tenant_id, provider_name, instance_name, supported_only
        )
        if success:
            return get_result(data=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/instances/<instance_name>/models", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def add_model_to_instance(tenant_id: str, provider_name: str, instance_name: str):
    """向实例添加一个模型配置。

    必填字段：
    - ``model_name`` —— 模型名称（如 "gpt-4o"、"text-embedding-3-small"）
    - ``model_type`` —— 模型类型（如 "chat" / "embedding" / "rerank" 等）

    可选字段：``max_tokens``（最大 token 数，默认 8192）、
    ``extra``（额外参数，如 temperature、top_p 等）。
    """
    data = await request.get_json()
    if not data or "model_name" not in data or "model_type" not in data:
        return get_error_argument_result(message="model_name and model_type are required")

    model_name = data["model_name"]
    model_type = data["model_type"]
    max_tokens = data.get("max_tokens", 8192)
    extra = data.get("extra", {})

    try:
        success, result = provider_api_service.add_model_to_instance(
            tenant_id, provider_name, instance_name, model_name, model_type, max_tokens, extra
        )
        if success:
            return get_result(message=result)
        else:
            return get_error_data_result(message=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


@manager.route("/providers/<provider_name>/instances/<instance_name>/models/<path:model_name>", methods=["PATCH"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def enable_or_disable_model(tenant_id: str = None, provider_name: str = None, instance_name: str = None, model_name: str = None):
    """启用或禁用实例中的某个模型。

    PATCH 请求，请求体需包含 ``status`` 字段，取值仅限：
    - ``"active"`` —— 启用模型
    - ``"inactive"`` —— 禁用模型（禁用的模型在应用中将不可见/不可用）
    """
    data = await request.get_json()
    if not data or "status" not in data:
        return get_error_argument_result(message="status is required")

    status = data["status"]
    if status not in ("active", "inactive"):
        return get_error_argument_result(message="status must be 'active' or 'inactive'")

    try:
        success, msg = provider_api_service.update_model_status(tenant_id, provider_name, instance_name, model_name, status)
        if success:
            return get_result(message=msg)
        else:
            return get_error_data_result(message=msg)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")


# =============================================================================
# 模型聊天测试接口
# =============================================================================

@manager.route("/providers/<provider_name>/instances/<instance_name>/models/<path:model_name>", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def chat_to_model(tenant_id: str = None, provider_name: str = None, instance_name: str = None, model_name: str = None):
    """向指定模型发送聊天消息（调试/测试用）。

    支持两种响应模式：
    - **非流式**（``stream=false``）：一次性返回完整回复
    - **流式 SSE**（``stream=true``）：通过 Server-Sent Events 逐 token 推送回复，
      前端可实时渲染模型输出。数据格式为 ``data: [MESSAGE]<chunk>\\n\\n``，
      以 ``data: [DONE]\\n\\n`` 结束。

    可选 ``thinking=true`` 启用深度推理模式（如 DeepSeek-R1 等支持思考链的模型）。
    """
    data = await request.get_json()
    if not data or "message" not in data:
        return get_error_argument_result(message="message is required")

    message = data["message"]
    stream = data.get("stream", False)
    thinking = data.get("thinking", False)

    try:
        success, result = await provider_api_service.chat_to_model(
            tenant_id, provider_name, instance_name, model_name, message, stream, thinking
        )
        if not success:
            return get_error_data_result(message=result)

        # 流式响应：使用 SSE (Server-Sent Events) 协议推送
        if stream and isinstance(result, dict) and result.get("type") == "stream":
            from quart import Response
            llm = result["llm"]

            async def generate():
                """异步生成器 —— 逐 chunk 从 LLM 获取输出并通过 SSE 推送。"""
                async for chunk in llm.async_chat_streamly(
                    None,
                    [{"role": "user", "content": message}],
                    {"temperature": 0.9},
                ):
                    # 过滤包含 **ERROR** 标记的异常 chunk
                    if chunk and isinstance(chunk, str) and chunk.find("**ERROR**") < 0:
                        yield f"data: [MESSAGE]{chunk}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate(), mimetype="text/event-stream", headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })

        # 非流式响应
        return get_result(data=result)
    except Exception as e:
        logging.exception(e)
        return get_error_data_result(message="Internal server error")
