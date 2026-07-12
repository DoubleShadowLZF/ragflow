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

API 工具模块

提供 API 层的通用工具函数，主要包括以下功能：
- 请求数据解析：JSON body / form data 统一获取与缓存
- JSON 序列化：递归序列化复杂对象（ModelMetaclass 等）
- 统一响应构建：get_json_result / get_result / get_error_data_result 等标准响应格式
- 请求校验装饰器：validate_request（参数验证）、active_required（用户激活检查）等
- 解析器配置：get_parser_config 合并默认与自定义分块参数
- OpenAI 兼容格式：构建符合 OpenAI API 格式的聊天响应
- 数据工具：deep_merge（深度合并字典）、remap_dictionary_keys（键名映射）、group_by（分组）
- MCP 工具获取与模型压力测试
"""

import asyncio
import functools
import inspect
import json
import logging
import sys
import time
from copy import deepcopy
from functools import wraps
from typing import Any

import requests
from quart import (
    jsonify,
    request,
    has_app_context,
)
from werkzeug.exceptions import BadRequest as WerkzeugBadRequest

try:
    from quart.exceptions import BadRequest as QuartBadRequest
except ImportError:  # pragma: no cover - optional dependency
    QuartBadRequest = None

from peewee import OperationalError

from common.constants import ActiveEnum, LLMType
from api.utils.json_encode import CustomJSONEncoder
from common.mcp_tool_call_conn import MCPToolCallSession, close_multiple_mcp_toolcall_sessions
from api.db.services.tenant_llm_service import LLMFactoriesService
from common.connection_utils import timeout
from common.constants import RetCode
from common import settings
from common.misc_utils import thread_pool_exec

# 全局替换 requests 库的 JSON 序列化器，使用自定义编码器处理特殊类型（如 datetime）
requests.models.complexjson.dumps = functools.partial(json.dumps, cls=CustomJSONEncoder)


# =============================================================================
#  请求数据解析
# =============================================================================

def _safe_jsonify(payload: dict):
    """安全的 jsonify 包装：有 Quart 应用上下文时正常序列化，否则直接返回原始 dict。"""
    if has_app_context():
        return jsonify(payload)
    return payload


async def _coerce_request_data() -> dict:
    """统一的请求体解析函数。

    按优先级处理三种请求体格式：
    1. 无 body → 返回空 dict
    2. Content-Type 为 application/json → JSON 解析
    3. 其他 Content-Type → 尝试 form data 解析

    结果会缓存到 request._cached_payload，避免重复解析。
    """
    if hasattr(request, "_cached_payload"):
        return request._cached_payload
    payload: Any = None

    body_bytes = await request.get_data()
    has_body = bool(body_bytes)
    content_type = (request.content_type or "").lower()
    is_json = content_type.startswith("application/json")

    if not has_body:
        payload = {}
    elif is_json:
        payload = await request.get_json(force=False, silent=False)
        if isinstance(payload, dict):
            payload = payload or {}
        elif isinstance(payload, str):
            raise AttributeError("'str' object has no attribute 'get'")
        else:
            raise TypeError("JSON payload must be an object.")
    else:
        form = await request.form
        payload = form.to_dict() if form else None
        if payload is None:
            raise TypeError("Request body is not a valid form payload.")

    request._cached_payload = payload
    return payload


async def get_request_json():
    """获取已缓存的请求 JSON 数据（委托给 _coerce_request_data）。"""
    return await _coerce_request_data()

# =============================================================================
#  JSON 序列化工具
# =============================================================================

def serialize_for_json(obj):
    """
    递归序列化对象，使其可被 JSON 编码。

    处理策略（按优先级）：
    1. 有 __dict__ 的对象 → 序列化其非私有属性
    2. 类 / 元类（有 __name__） → 返回 "<module.ClassName>" 格式字符串
    3. list/tuple → 递归序列化每个元素
    4. dict → 递归序列化每个 value
    5. 基本类型（str/int/float/bool/None） → 原样返回
    6. 其他类型 → fallback 为 str() 字符串
    """
    if hasattr(obj, "__dict__"):
        try:
            return {key: serialize_for_json(value) for key, value in obj.__dict__.items() if not key.startswith("_")}
        except (AttributeError, TypeError):
            return str(obj)
    elif hasattr(obj, "__name__"):
        return f"<{obj.__module__}.{obj.__name__}>" if hasattr(obj, "__module__") else f"<{obj.__name__}>"
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)


# =============================================================================
#  错误响应与统一结果构建
# =============================================================================

def get_data_error_result(code=RetCode.DATA_ERROR, message="Sorry! Data missing!"):
    """构建数据错误响应。

    当有活跃异常时记录完整 traceback，否则仅记录错误消息。
    """
    if sys.exc_info()[0] is not None:
        logging.exception(message)
    else:
        logging.error(message)
    result_dict = {"code": code, "message": message}
    response = {}
    for key, value in result_dict.items():
        if value is None and key != "code":
            continue
        else:
            response[key] = value
    return _safe_jsonify(response)


def server_error_response(e):
    """统一的服务端异常响应处理。

    特性：
    - 自动识别 401 未授权异常并返回对应状态码
    - 识别 Elasticsearch 索引不存在错误，给出友好提示
    - 识别通用 not_found 错误
    - 其余异常返回 EXCEPTION_ERROR 码 + 异常描述
    """
    logging.error("Unhandled exception during request", exc_info=(type(e), e, e.__traceback__))
    try:
        msg = repr(e).lower()
        if getattr(e, "code", None) == 401 or ("unauthorized" in msg) or ("401" in msg):
            resp = get_json_result(code=RetCode.UNAUTHORIZED, message="Unauthorized")
            resp.status_code = RetCode.UNAUTHORIZED
            return resp
    except Exception as ex:
        logging.warning(f"error checking authorization: {ex}")

    if repr(e).find("index_not_found_exception") >= 0:
        return get_json_result(code=RetCode.EXCEPTION_ERROR, message="No chunk found, please upload file and parse it.")

    if "not_found" in str(e):
        return get_error_data_result(message="No chunk found! Check the chunk status please!")

    return get_json_result(code=RetCode.EXCEPTION_ERROR, message=repr(e))


# =============================================================================
#  请求校验装饰器
# =============================================================================

def validate_request(*args, **kwargs):
    """请求参数校验装饰器。

    用法：
        @validate_request("name", "type")        # 必填参数检查
        @validate_request("status", status=1)     # 必填 + 固定值校验
        @validate_request("type", type=["a","b"]) # 必填 + 允许值集合校验

    校验逻辑：
    - args 中的参数必须在请求体中存在
    - kwargs 中的参数必须存在且值匹配（支持固定值或允许值列表）
    """
    def process_args(input_arguments):
        no_arguments = []
        error_arguments = []
        for arg in args:
            if arg not in input_arguments:
                no_arguments.append(arg)
        for k, v in kwargs.items():
            config_value = input_arguments.get(k, None)
            if config_value is None:
                no_arguments.append(k)
            elif isinstance(v, (tuple, list)):
                if config_value not in v:
                    error_arguments.append((k, set(v)))
            elif config_value != v:
                error_arguments.append((k, v))
        if no_arguments or error_arguments:
            error_string = ""
            if no_arguments:
                error_string += "required argument are missing: {}; ".format(",".join(no_arguments))
            if error_arguments:
                error_string += "required argument values: {}".format(",".join(["{}={}".format(a[0], a[1]) for a in error_arguments]))
            return error_string
        return None

    def wrapper(func):
        @wraps(func)
        async def decorated_function(*_args, **_kwargs):
            exception_types = (AttributeError, TypeError, WerkzeugBadRequest)
            if QuartBadRequest is not None:
                exception_types = exception_types + (QuartBadRequest,)
            if args or kwargs:
                try:
                    input_arguments = await _coerce_request_data()
                except exception_types:
                    input_arguments = {}
            else:
                input_arguments = await _coerce_request_data()
            errs = process_args(input_arguments)
            if errs:
                return get_json_result(code=RetCode.ARGUMENT_ERROR, message=errs)
            if inspect.iscoroutinefunction(func):
                return await func(*_args, **_kwargs)
            return func(*_args, **_kwargs)

        return decorated_function

    return wrapper


def not_allowed_parameters(*params):
    """禁止参数装饰器：若请求体包含指定参数，直接返回参数错误。"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            input_arguments = await _coerce_request_data()
            for param in params:
                if param in input_arguments:
                    return get_json_result(code=RetCode.ARGUMENT_ERROR, message=f"Parameter {param} isn't allowed")
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper

    return decorator


def active_required(func):
    """用户激活检查装饰器：仅激活状态的用户可访问被装饰的端点。"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        from api.db.services import UserService
        from api.apps import current_user

        user_id = current_user.id
        usr = UserService.filter_by_id(user_id)
        if not usr or not usr.is_active == ActiveEnum.ACTIVE.value:
            return get_json_result(code=RetCode.FORBIDDEN, message="User isn't active, please activate first.")
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def add_tenant_id_to_kwargs(func):
    """自动注入 tenant_id 到函数关键字参数的装饰器。"""
    @wraps(func)
    async def wrapper(**kwargs):
        from api.apps import current_user
        kwargs["tenant_id"] = current_user.id
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        return func(**kwargs)
    return wrapper


# =============================================================================
#  标准 JSON 响应构建函数
# =============================================================================

def get_json_result(code: RetCode = RetCode.SUCCESS, message="success", data=None):
    """构建标准 JSON 响应：{"code": ..., "message": ..., "data": ...}。

    这是最常用的成功响应构建函数，RAGFlow 的内部 API 统一使用此格式。
    """
    response = {"code": code, "message": message, "data": data}
    return _safe_jsonify(response)


def build_error_result(code=RetCode.FORBIDDEN, message="success"):
    """构建错误响应并设置 HTTP 状态码。"""
    response = {"code": code, "message": message}
    response = _safe_jsonify(response)
    if hasattr(response, "status_code"):
        response.status_code = code
    return response


def construct_json_result(code: RetCode = RetCode.SUCCESS, message="success", data=None):
    """构建 JSON 响应，data 为 None 时省略该字段。"""
    if data is None:
        return _safe_jsonify({"code": code, "message": message})
    return _safe_jsonify({"code": code, "message": message, "data": data})


def get_result(code=RetCode.SUCCESS, message="", data=None, total=None):
    """
    标准 API 分页响应格式。

    成功时返回 {"code": 0, "data": [...], "total_datasets": 47}
    失败时返回 {"code": xxx, "message": "..."}

    注意：total 字段在 JSON 中映射为 total_datasets，用于前端兼容。
    """
    response = {"code": code}

    if code == RetCode.SUCCESS:
        if data is not None:
            response["data"] = data
        if total is not None:
            response["total_datasets"] = total
    else:
        response["message"] = message or "Error"

    return _safe_jsonify(response)


def get_error_data_result(
    message="Sorry! Data missing!",
    code=RetCode.DATA_ERROR,
):
    """构建数据错误响应，过滤掉值为 None 的非 code 字段。"""
    result_dict = {"code": code, "message": message}
    response = {}
    for key, value in result_dict.items():
        if value is None and key != "code":
            continue
        else:
            response[key] = value
    return _safe_jsonify(response)


def get_error_argument_result(message="Invalid arguments"):
    """快捷构建参数错误响应。"""
    return get_result(code=RetCode.ARGUMENT_ERROR, message=message)


def get_error_permission_result(message="Permission error"):
    """快捷构建权限错误响应。"""
    return get_result(code=RetCode.PERMISSION_ERROR, message=message)


def get_error_operating_result(message="Operating error"):
    """快捷构建操作错误响应。"""
    return get_result(code=RetCode.OPERATING_ERROR, message=message)


# =============================================================================
#  Token 生成
# =============================================================================

def generate_confirmation_token():
    """生成 API 访问令牌，格式为 "ragflow-" + 32 字节 URL 安全随机字符串。"""
    import secrets

    return "ragflow-" + secrets.token_urlsafe(32)


# =============================================================================
#  文档解析器配置
# =============================================================================

def get_parser_config(chunk_method, parser_config):
    """获取文档分块解析器的完整配置。

    将用户自定义配置与各分块方法的默认值深度合并。默认值包含：
    - 基础参数：chunk_token_num、delimiter、layout_recognize 等
    - RAPTOR 递归摘要参数
    - GraphRAG 知识图谱构建参数
    - 父子分块（parent_child）参数

    合并策略：用户配置的字段覆盖默认值，未配置的字段保留默认值。
    """
    if not chunk_method:
        chunk_method = "naive"

    # 各分块方法的默认配置
    base_defaults = {
        "table_context_size": 0,
        "image_context_size": 0,
    }
    key_mapping = {
        "naive": {
            "layout_recognize": "DeepDOC",
            "chunk_token_num": 512,
            "delimiter": "\n",
            "auto_keywords": 0,
            "auto_questions": 0,
            "html4excel": False,
            "topn_tags": 3,
            "raptor": {
                "use_raptor": True,
                "prompt": "Please summarize the following paragraphs. Be careful with the numbers, do not make things up. Paragraphs as following:\n      {cluster_content}\nThe above is the content you need to summarize.",
                "max_token": 256,
                "threshold": 0.1,
                "max_cluster": 64,
                "random_seed": 0,
            },
            "graphrag": {
                "use_graphrag": True,
                "entity_types": [
                    "organization",
                    "person",
                    "geo",
                    "event",
                    "category",
                ],
                "method": "light",
                "batch_chunk_token_size": 4096,
                "retry_attempts": 2,
                "retry_backoff_seconds": 2.0,
                "retry_backoff_max_seconds": 60.0,
                "build_subgraph_timeout_per_chunk_seconds": 300,
                "build_subgraph_min_timeout_seconds": 600,
                "merge_timeout_seconds": 180,
                "resolution_timeout_seconds": 1800,
                "community_timeout_seconds": 1800,
                "lock_acquire_timeout_seconds": 600,
            },
            "parent_child": {
                "use_parent_child": False,
                "children_delimiter": "\n",
            },
        },
        "qa": {"raptor": {"use_raptor": False}, "graphrag": {"use_graphrag": False}},
        "tag": None,
        "resume": None,
        "manual": {"raptor": {"use_raptor": False}, "graphrag": {"use_graphrag": False}},
        "table": None,
        "paper": {"raptor": {"use_raptor": False}, "graphrag": {"use_graphrag": False}},
        "book": {"raptor": {"use_raptor": False}, "graphrag": {"use_graphrag": False}},
        "laws": {"raptor": {"use_raptor": False}, "graphrag": {"use_graphrag": False}},
        "presentation": {"raptor": {"use_raptor": False}, "graphrag": {"use_graphrag": False}},
        "one": None,
        "knowledge_graph": {
            "chunk_token_num": 8192,
            "delimiter": r"\n",
            "entity_types": ["organization", "person", "location", "event", "time"],
            "raptor": {"use_raptor": False},
            "graphrag": {"use_graphrag": False},
        },
        "email": None,
        "picture": None,
    }

    default_config = key_mapping[chunk_method]

    # 无用户配置 → 合并基础默认值和方法默认值
    if not parser_config:
        if default_config is None:
            merged_config = deep_merge(base_defaults, {})
        else:
            merged_config = deep_merge(base_defaults, default_config)
    elif default_config is None:
        # 有用户配置但该方法无默认值 → 仅合并基础默认值和用户配置
        merged_config = deep_merge(base_defaults, parser_config)
    else:
        # 有用户配置且有方法默认值 → 三层合并（基础 → 方法默认 → 用户自定义）
        merged_config = deep_merge(base_defaults, default_config)
        merged_config = deep_merge(merged_config, parser_config)

    # 将 parent_child 嵌套配置展平为 children_delimiter，供执行层直接使用
    pc = merged_config.get("parent_child", {})
    if pc.get("use_parent_child"):
        merged_config["children_delimiter"] = pc.get("children_delimiter", "\n")
    elif pc:
        merged_config["children_delimiter"] = ""

    return merged_config


# =============================================================================
#  OpenAI 兼容格式响应构建
# =============================================================================

def get_data_openai(id=None, created=None, model=None, prompt_tokens=0, completion_tokens=0, content=None, finish_reason=None, object="chat.completion", param=None, stream=False):
    """构建符合 OpenAI API 格式的聊天响应。

    支持两种模式：
    - stream=True: 返回 chat.completion.chunk 格式的流式响应块
    - stream=False: 返回标准 chat.completion 格式，含 usage token 统计
    """
    total_tokens = prompt_tokens + completion_tokens

    if stream:
        return {
            "id": f"{id}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                    "index": 0,
                }
            ],
        }

    return {
        "id": f"{id}",
        "object": object,
        "created": int(time.time()) if created else None,
        "model": model,
        "param": param,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": finish_reason,
                "index": 0,
            }
        ],
    }


# =============================================================================
#  数据校验与工具函数
# =============================================================================

def check_duplicate_ids(ids, id_type="item"):
    """
    检查 ID 列表中的重复项，返回去重后的 ID 列表和重复错误消息。

    Args:
        ids: ID 列表
        id_type: ID 类型名称，用于错误消息（如 'document', 'dataset', 'chunk'）

    Returns:
        tuple: (unique_ids, error_messages)
    """
    id_count = {}
    duplicate_messages = []

    # Count occurrences of each ID
    for id_value in ids:
        id_count[id_value] = id_count.get(id_value, 0) + 1

    # Check for duplicates
    for id_value, count in id_count.items():
        if count > 1:
            duplicate_messages.append(f"Duplicate {id_type} ids: {id_value}")

    # Return unique IDs and error messages
    return list(set(ids)), duplicate_messages


def verify_embedding_availability(embd_id: str, tenant_id: str) -> tuple[bool, str | None]:
    """
    验证指定租户是否可用某个嵌入模型。

    验证流程：
    1. 解析 embd_id 中的模型名和厂商信息
    2. 检查模型在系统中的注册状态
    3. 验证租户级别的模型授权
    4. 确认模型在内置模型列表中

    Args:
        embd_id: 嵌入模型标识符，格式 "model_name@factory"（如 "text-embedding@openai"）
        tenant_id: 租户 ID

    Returns:
        tuple[bool, str | None]: (是否可用, 错误信息或None)
    """
    from api.db.joint_services.tenant_model_service import get_model_config_from_provider_instance
    try:
        get_model_config_from_provider_instance(tenant_id, LLMType.EMBEDDING, embd_id)
    except LookupError as e:
        return False, str(e)
    except OperationalError as e:
        logging.exception(e)
        return False, "Database operation failed"
    except Exception as e:
        logging.exception(e)
        return False, "Internal server error"

    return True, None


# =============================================================================
#  字典操作工具
# =============================================================================

def deep_merge(default: dict, custom: dict) -> dict:
    """
    深度合并两个字典，custom 中的值优先。

    使用栈式迭代（非递归）实现，避免深层嵌套导致的递归深度问题。

    合并规则：
    - 两个 dict 的同名 key 且 value 均为 dict → 递归合并
    - 其他情况（非 dict value 或类型不匹配） → custom 完全覆盖 default

    Example:
        >>> default = {"a": 1, "nested": {"x": 10, "y": 20}}
        >>> custom = {"b": 2, "nested": {"y": 99, "z": 30}}
        >>> deep_merge(default, custom)
        {'a': 1, 'b': 2, 'nested': {'x': 10, 'y': 99, 'z': 30}}
    """
    merged = deepcopy(default)
    stack = [(merged, custom)]

    while stack:
        base_dict, override_dict = stack.pop()

        for key, val in override_dict.items():
            if key in base_dict and isinstance(val, dict) and isinstance(base_dict[key], dict):
                stack.append((base_dict[key], val))
            else:
                base_dict[key] = val

    return merged


def remap_dictionary_keys(source_data: dict, key_aliases: dict = None) -> dict:
    """
    将字典的键名按映射表转换（仅改键名，值不变）。

    默认映射（旧字段名 → 新字段名）：
    - chunk_num → chunk_count
    - doc_num → document_count
    - parser_id → chunk_method
    - embd_id → embedding_model

    可通过 key_aliases 参数传入自定义映射覆盖默认值。
    """
    DEFAULT_KEY_MAP = {
        "chunk_num": "chunk_count",
        "doc_num": "document_count",
        "parser_id": "chunk_method",
        "embd_id": "embedding_model",
    }

    transformed_data = {}
    mapping = key_aliases or DEFAULT_KEY_MAP

    for original_key, value in source_data.items():
        mapped_key = mapping.get(original_key, original_key)
        transformed_data[mapped_key] = value

    return transformed_data


def group_by(list_of_dict, key):
    """按指定键对字典列表进行分组，返回 {key_value: [matched_items, ...]} 的结构。"""
    res = {}
    for item in list_of_dict:
        if item[key] in res.keys():
            res[item[key]].append(item)
        else:
            res[item[key]] = [item]
    return res


# =============================================================================
#  MCP 工具获取
# =============================================================================

def get_mcp_tools(mcp_servers: list, timeout: float | int = 10) -> tuple[dict, str]:
    """从 MCP 服务器列表获取所有可用工具。

    遍历每个 MCP 服务器，获取其工具列表并合并缓存中的 enabled 状态。
    完成后关闭所有 MCP 会话释放资源。

    Returns:
        tuple[dict, str]: (按服务器分组的工具列表, 错误信息（空串表示成功）)
    """
    results = {}
    tool_call_sessions = []
    try:
        for mcp_server in mcp_servers:
            server_key = mcp_server.id

            cached_tools = mcp_server.variables.get("tools", {})

            tool_call_session = MCPToolCallSession(mcp_server, mcp_server.variables)
            tool_call_sessions.append(tool_call_session)

            try:
                tools = tool_call_session.get_tools(timeout)
            except Exception:
                tools = []

            results[server_key] = []
            for tool in tools:
                tool_dict = tool.model_dump()
                cached_tool = cached_tools.get(tool_dict["name"], {})

                # 合并缓存中的 enabled 状态，默认启用
                tool_dict["enabled"] = cached_tool.get("enabled", True)
                results[server_key].append(tool_dict)

        close_multiple_mcp_toolcall_sessions(tool_call_sessions)
        return results, ""
    except Exception as e:
        return {}, str(e)


# =============================================================================
#  模型压力测试（GraphRAG 任务前置检查）
# =============================================================================

async def is_strong_enough(chat_model, embedding_model):
    """对聊天模型和嵌入模型进行并发压力测试。

    并发执行 STRONG_TEST_COUNT 次模型调用，验证模型在并发场景下的可用性。
    主要用于 GraphRAG 任务的前置检查，避免在模型不可用时启动大规模处理。

    每个并发任务包含：
    - 嵌入模型：编码测试文本（10 秒超时）
    - 聊天模型：发送测试对话（30 秒超时）

    若任一并发任务失败，取消所有剩余任务并向上抛出异常。
    """
    count = settings.STRONG_TEST_COUNT
    if not chat_model or not embedding_model:
        return
    if isinstance(count, int) and count <= 0:
        return

    @timeout(60, 2)
    async def _is_strong_enough():
        nonlocal chat_model, embedding_model
        if embedding_model:
            await asyncio.wait_for(
                thread_pool_exec(embedding_model.encode, ["Are you strong enough!?"]),
                timeout=10
            )

        if chat_model:
            res = await asyncio.wait_for(
                chat_model.async_chat("Nothing special.", [{"role": "user", "content": "Are you strong enough!?"}]),
                timeout=30
            )
            if "**ERROR**" in res:
                raise Exception(res)

    # 创建 STRONG_TEST_COUNT 个并发压力测试任务
    tasks = [
        asyncio.create_task(_is_strong_enough())
        for _ in range(count)
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logging.error(f"Pressure test failed: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# =============================================================================
#  LLM 厂商过滤
# =============================================================================

def get_allowed_llm_factories() -> list:
    """获取当前允许使用的 LLM 厂商列表。

    按 rank 降序排列，若配置了 ALLOWED_LLM_FACTORIES 白名单则过滤。
    """
    factories = list(LLMFactoriesService.get_all(reverse=True, order_by="rank"))
    if settings.ALLOWED_LLM_FACTORIES is None:
        return factories

    return [factory for factory in factories if factory.name in settings.ALLOWED_LLM_FACTORIES]
