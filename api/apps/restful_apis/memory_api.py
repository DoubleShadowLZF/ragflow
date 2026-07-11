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
import logging
import os
import time

from quart import request, g
from common.constants import RetCode
from common.exceptions import ArgumentException, NotFoundException
from api.apps import login_required, current_user
from api.utils.api_utils import validate_request, get_request_json, get_error_argument_result, get_json_result
from api.apps.services import memory_api_service
from api.utils.pagination_utils import validate_rest_api_page_size


@manager.route("/memories", methods=["POST"])  # noqa: F821
@login_required
@validate_request("name", "memory_type", "embd_id", "llm_id")
async def create_memory():
    """
    创建用户记忆（Memory），用于存储用户的偏好、上下文等信息。
    """
    timing_enabled = os.getenv("RAGFLOW_API_TIMING")
    t_start = time.perf_counter() if timing_enabled else None
    req = await get_request_json()
    t_parsed = time.perf_counter() if timing_enabled else None
    try:
        # 构建 Memory 信息
        memory_info = {
            "name": req["name"],
            "memory_type": req["memory_type"],
            "embd_id": req["embd_id"],
            "llm_id": req["llm_id"]
        }
        # 调用服务层
        success, res = await memory_api_service.create_memory(memory_info)
        # 性能日志（成功路径）
        if timing_enabled:
            logging.info(
                "api_timing create_memory parse_ms=%.2f validate_and_db_ms=%.2f total_ms=%.2f path=%s",
                (t_parsed - t_start) * 1000,            # JSON 解析耗时
                (time.perf_counter() - t_parsed) * 1000,      # 验证和数据库操作耗时
                (time.perf_counter() - t_start) * 1000,       # 总耗时
                request.path,                                 # 请求路径
            )
        if success:
            return get_json_result(message=True, data=res)
        else:
            return get_json_result(message=res, code=RetCode.SERVER_ERROR)

    except ArgumentException as arg_error:
        logging.error(arg_error)
        if timing_enabled:
            logging.info(
                "api_timing create_memory error=%s parse_ms=%.2f total_ms=%.2f path=%s",
                str(arg_error),
                (t_parsed - t_start) * 1000,
                (time.perf_counter() - t_start) * 1000,
                request.path,
            )
        return get_error_argument_result(str(arg_error))

    except Exception as e:
        logging.error(e)
        if timing_enabled:
            logging.info(
                "api_timing create_memory error=%s parse_ms=%.2f total_ms=%.2f path=%s",
                str(e),
                (t_parsed - t_start) * 1000,
                (time.perf_counter() - t_start) * 1000,
                request.path,
            )
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/memories/<memory_id>", methods=["PUT"])  # noqa: F821
@login_required
async def update_memory(memory_id):
    """
    更新指定 Memory 的配置信息，支持部分更新。
    """
    req = await get_request_json()
    new_settings = {k: req[k] for k in [
        "name", "permissions", "llm_id", "embd_id", "memory_type", "memory_size", "forgetting_policy", "temperature",
        "avatar", "description", "system_prompt", "user_prompt", "tenant_llm_id", "tenant_embd_id"
    ] if k in req}
    try:
        success, res = await memory_api_service.update_memory(memory_id, new_settings)
        if success:
            return get_json_result(message=True, data=res)
        else:
            return get_json_result(message=res, code=RetCode.SERVER_ERROR)
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except ArgumentException as arg_error:
        logging.error(arg_error)
        return get_error_argument_result(str(arg_error))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/memories/<memory_id>", methods=["DELETE"])  # noqa: F821
@login_required
async def delete_memory(memory_id):
    """
    删除指定的 Memory 记录。
    """
    try:
        await memory_api_service.delete_memory(memory_id)
        return get_json_result(message=True)
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/memories", methods=["GET"])  # noqa: F821
@login_required
async def list_memory():
    """
    获取 Memory 列表，支持按类型、租户、所有者、存储类型过滤，以及关键词搜索和分页。
    """
    filter_params = {
        k: request.args.get(k) for k in ["memory_type", "tenant_id", "owner_ids", "storage_type"] if k in request.args
    }
    keywords = request.args.get("keywords")
    page = int(request.args.get("page", 1))
    page_size = validate_rest_api_page_size(int(request.args.get("page_size", 50)))
    try:
        res = await memory_api_service.list_memory(filter_params, keywords, page, page_size)
        return get_json_result(message=True, data=res)
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/memories/<memory_id>/config", methods=["GET"])  # noqa: F821
@login_required
async def get_memory_config(memory_id):
    """
    获取指定 Memory 的完整配置信息。
    """
    try:
        res = await memory_api_service.get_memory_config(memory_id)
        return get_json_result(message=True, data=res)
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/memories/<memory_id>", methods=["GET"])  # noqa: F821
@login_required
async def get_memory_messages(memory_id):
    """
    获取指定 Memory 关联的消息/对话记录列表，支持按 Agent 过滤和关键词搜索。
    """
    args = request.args
    agent_ids = args.getlist("agent_id")
    if len(agent_ids) == 1 and ',' in agent_ids[0]:
        agent_ids = agent_ids[0].split(',')
    keywords = args.get("keywords", "")
    keywords = keywords.strip()
    page = int(args.get("page", 1))
    page_size = validate_rest_api_page_size(int(args.get("page_size", 50)))
    try:
        res = await memory_api_service.get_memory_messages(
            memory_id, agent_ids, keywords, page, page_size
        )
        return get_json_result(message=True, data=res)
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/messages", methods=["POST"]) # noqa: F821
@login_required
@validate_request("memory_id", "agent_id", "session_id", "user_input", "agent_response")
async def add_message():
    """
    向指定的 Memory（一个或多个）中添加用户和助手的对话消息。
    """
    req = await get_request_json()
    memory_ids = req["memory_id"]

    # JWT / session users cannot spoof attribution; API-key callers may supply an external subject id.
    try:
        trust_client_subject = bool(getattr(g, "auth_via_api_token", False))
    except RuntimeError:
        trust_client_subject = False
    if trust_client_subject:
        effective_user_id = req.get("user_id", "")
    else:
        effective_user_id = current_user.id

    message_dict = {
        "user_id": effective_user_id,
        "agent_id": req["agent_id"],
        "session_id": req["session_id"],
        "user_input": req["user_input"],
        "agent_response": req["agent_response"],
    }

    res, msg = await memory_api_service.add_message(memory_ids, message_dict)
    if res:
        return get_json_result(message=msg)

    return get_json_result(message="Some messages failed to add. Detail:" + msg, code=RetCode.SERVER_ERROR)


@manager.route("/messages/<memory_id>:<message_id>", methods=["DELETE"]) # noqa: F821
@login_required
async def forget_message(memory_id: str, message_id: int):
    """
    从指定 Memory 中删除/遗忘一条消息。
    """
    try:
        # 调用 Service 层执行删除
        res = await memory_api_service.forget_message(memory_id, message_id)
        return get_json_result(message=res)
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/messages/<memory_id>:<message_id>", methods=["PUT"]) # noqa: F821
@login_required
@validate_request("status")
async def update_message(memory_id: str, message_id: int):
    """
    更新 Memory 中指定消息的状态（布尔值）。
    """
    req = await get_request_json()
    status = req["status"]
    if not isinstance(status, bool):
        return get_error_argument_result("Status must be a boolean.")

    try:
        update_succeed = await memory_api_service.update_message_status(memory_id, message_id, status)
        if update_succeed:
            return get_json_result(message=update_succeed)
        else:
            return get_json_result(code=RetCode.SERVER_ERROR, message=f"Failed to set status for message '{message_id}' in memory '{memory_id}'.")
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/messages/search", methods=["GET"]) # noqa: F821
@login_required
async def search_message():
    """
    在指定的 Memory 中，基于语义相似度和关键词匹配搜索消息。
    """
    args = request.args
    # 解析 Memory ID 参数,支持两种格式：
    # ?memory_id=mem1&memory_id=mem2
    # ?memory_id=mem1,mem2
    memory_ids = args.getlist("memory_id")
    if len(memory_ids) == 1 and ',' in memory_ids[0]:
        memory_ids = memory_ids[0].split(',')
    # 提取搜索参数
    query = args.get("query")                                                          # 搜索查询文本
    similarity_threshold = float(args.get("similarity_threshold", 0.2))                # 相似度阈值
    keywords_similarity_weight = float(args.get("keywords_similarity_weight", 0.7))    # 关键词权重
    top_n = int(args.get("top_n", 5))                                                  # 返回结果数量
    # 提取过滤条件
    agent_id = args.get("agent_id", "")
    session_id = args.get("session_id", "")
    user_id = args.get("user_id", "")

    # 构建过滤器和参数
    filter_dict = {
        "memory_id": memory_ids,
        "agent_id": agent_id,
        "session_id": session_id,
        "user_id": user_id
    }
    params = {
        "query": query,
        "similarity_threshold": similarity_threshold,
        "keywords_similarity_weight": keywords_similarity_weight,
        "top_n": top_n
    }
    # 调用服务层
    res = await memory_api_service.search_message(filter_dict, params)
    return get_json_result(message=True, data=res)

@manager.route("/messages", methods=["GET"]) # noqa: F821
@login_required
async def get_messages():
    """获取指定 Memory 中的消息列表，支持按 Agent 和 Session 过滤。"""
    args = request.args
    # 解析 Memory ID 参数,支持两种格式：
    # ?memory_id=mem1&memory_id=mem2
    # ?memory_id=mem1,mem2
    memory_ids = args.getlist("memory_id")
    if len(memory_ids) == 1 and ',' in memory_ids[0]:
        memory_ids = memory_ids[0].split(',')
    # 提取过滤和分页参数
    agent_id = args.get("agent_id", "")         # 按 Agent 过滤
    session_id = args.get("session_id", "")     # 按 Session 过滤
    limit = int(args.get("limit", 10))          # 返回消息数量（默认 10）
    # 验证必填参数
    if not memory_ids:
        return get_error_argument_result("memory_ids is required.")
    try:
        # 调用服务层
        res = await memory_api_service.get_messages(memory_ids, agent_id, session_id, limit)
        return get_json_result(message=True, data=res)
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")


@manager.route("/messages/<memory_id>:<message_id>/content", methods=["GET"]) # noqa: F821
@login_required
async def get_message_content(memory_id: str, message_id: int):
    """获取 Memory 中指定消息的完整内容。"""
    try:
        res = await memory_api_service.get_message_content(memory_id, message_id)
        return get_json_result(message=True, data=res)
    except NotFoundException as not_found_exception:
        logging.error(not_found_exception)
        return get_json_result(code=RetCode.NOT_FOUND, message=str(not_found_exception))
    except Exception as e:
        logging.error(e)
        return get_json_result(code=RetCode.SERVER_ERROR, message="Internal server error")
