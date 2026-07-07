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

import json
import logging
import math
import os
import re
import tempfile
from copy import deepcopy
from types import SimpleNamespace

from quart import Response, request

from api.apps import current_user, login_required
from api.apps.restful_apis._generation_params import merge_generation_config, pop_generation_config
from api.db.joint_services.tenant_model_service import (
    get_tenant_default_model_by_type, get_model_config_from_provider_instance, get_api_key, split_model_name
)
from api.db.services.chunk_feedback_service import ChunkFeedbackService
from api.db.services.conversation_service import ConversationService, structure_answer
from api.db.services.dialog_service import DialogService, async_chat, gen_mindmap
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.db.services.search_service import SearchService
from api.db.services.user_service import TenantService, UserTenantService
from api.utils.api_utils import (
    check_duplicate_ids,
    get_data_error_result,
    get_json_result,
    get_request_json,
    server_error_response,
    validate_request,
)
from api.utils.pagination_utils import validate_rest_api_page_size
from common.constants import LLMType, RetCode, StatusEnum
from common import settings
from common.misc_utils import get_uuid, thread_pool_exec
from rag.prompts.generator import chunks_format
from rag.prompts.template import load_prompt

def _sanitize_json_floats(obj):
    """Replace NaN/Infinity floats with None so the result is RFC 8259 JSON.

    `json.dumps` emits the literal tokens `NaN`/`Infinity` by default
    (allow_nan=True). Those tokens are valid Python JSON output but invalid
    per the JSON spec, and downstream proxies / Go consumers reject the
    response with `failed to encode response: json: unsupported value: NaN`
    (fixes #15245). Retrieval scores (similarity, vector_similarity,
    term_similarity) can become NaN when an aggregation runs over an empty
    set or when a similarity denominator is zero, so the chat completions
    stream is the realistic trigger.

    `isinstance(obj, float)` alone catches Python float and numpy.float64
    (a float subclass) but misses numpy.float32 / numpy.float16 and any
    other duck-typed numeric. Probe via math.isnan/isinf in a try/except
    so any object math can evaluate gets sanitized — without changing
    upstream callers like chunks_format or rag/nlp/search.py.
    """
    try:
        if math.isnan(obj) or math.isinf(obj):
            return None
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_json_floats(v) for v in obj)
    return obj


_DEFAULT_PROMPT_CONFIG = {
    "system": (
        'You are an intelligent assistant. Please summarize the content of the dataset to answer the question. '
        'Please list the data in the dataset and answer in detail. When all dataset content is irrelevant to the '
        'question, your answer must include the sentence "The answer you are looking for is not found in the dataset!" '
        "Answers need to consider chat history.\n"
        "      Here is the knowledge base:\n"
        "      {knowledge}\n"
        "      The above is the knowledge base."
    ),
    "prologue": "Hi! I'm your assistant. What can I do for you?",
    "parameters": [{"key": "knowledge", "optional": False}],
    "empty_response": "Sorry! No relevant content was found in the knowledge base!",
    "quote": True,
    "tts": False,
    "refine_multiturn": True,
}
_DEFAULT_DIRECT_CHAT_PROMPT_CONFIG = {
    "system": "",                      # 系统提示词
    "prologue": "",                    # 开场白
    "parameters": [],                  # 参数列表
    "empty_response": "",              # 空响应消息
    "quote": False,                    # 引用开关
    "tts": False,                      # 语音合成开关
    "refine_multiturn": True,          # 多轮对话精炼
}
_DEFAULT_RERANK_MODELS = {"BAAI/bge-reranker-v2-m3", "maidalun1020/bce-reranker-base_v1"}
_READONLY_FIELDS = {"id", "tenant_id", "created_by", "create_time", "create_date", "update_time", "update_date"}
_PERSISTED_FIELDS = set(DialogService.model._meta.fields)


def _build_chat_response(chat):
    data = chat.to_dict() if hasattr(chat, "to_dict") else dict(chat)
    kb_ids, kb_names = _resolve_kb_names(data.get("kb_ids", []))
    data["dataset_ids"] = kb_ids
    data.pop("kb_ids", None)
    data["kb_names"] = kb_names
    return data


def _resolve_kb_names(kb_ids):
    ids, names = [], []
    for kb_id in kb_ids or []:
        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok or kb.status != StatusEnum.VALID.value:
            continue
        ids.append(kb_id)
        names.append(kb.name)
    return ids, names


def _has_knowledge_placeholder(prompt_config):
    return "{knowledge}" in (prompt_config or {}).get("system", "")


def _validate_name(name, *, required=True):
    if name is None:
        if required:
            return None, "`name` is required."
        return None, None
    if not isinstance(name, str):
        return None, "Chat name must be a string."
    name = name.strip()
    if not name:
        return None, "`name` is required." if required else "`name` cannot be empty."
    if len(name.encode("utf-8")) > 255:
        return None, f"Chat name length is {len(name.encode('utf-8'))} which is larger than 255."
    return name, None


def _build_session_response(conv: dict) -> dict:
    conv = dict(conv)
    conv["chat_id"] = conv.pop("dialog_id", conv.get("chat_id"))
    conv["messages"] = conv.pop("message", conv.get("messages", []))
    return conv


async def _ensure_owned_chat(chat_id):
    return await thread_pool_exec(
        DialogService.query,
        tenant_id=current_user.id, id=chat_id, status=StatusEnum.VALID.value
    )


def _build_default_completion_dialog():
    return SimpleNamespace(
        tenant_id=current_user.id,
        llm_id="",
        tenant_llm_id=None,
        llm_setting={},
        prompt_config=deepcopy(_DEFAULT_DIRECT_CHAT_PROMPT_CONFIG),
        kb_ids=[],
        top_n=6,
        top_k=1024,
        rerank_id="",
        similarity_threshold=0.1,
        vector_similarity_weight=0.3,
        meta_data_filter=None,
    )


async def _create_session_for_completion(chat_id, dialog, user_id):
    conv = {
        "id": get_uuid(),
        "dialog_id": chat_id,
        "name": "New session",
        "message": [{"role": "assistant", "content": dialog.prompt_config.get("prologue", "")}],
        "user_id": user_id,
        "reference": [],
    }
    await thread_pool_exec(ConversationService.save, **conv)
    ok, conv_obj = await thread_pool_exec(ConversationService.get_by_id, conv["id"])
    if not ok:
        raise LookupError("Fail to create a session!")
    return conv_obj


def _get_bool_request_flag(req, *names, default=False):
    for name in names:
        if name not in req:
            continue
        value = req.pop(name)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return default


def _normalize_completion_messages(req):
    messages = req.get("messages")
    if messages is None:
        question = req.get("question")
        if question is None:
            return None, get_data_error_result(
                code=RetCode.ARGUMENT_ERROR,
                message="required argument are missing: messages",
            )
        messages = [{"role": "user", "content": question}]
        if req.get("files"):
            messages[-1]["files"] = req["files"]

    if not isinstance(messages, list) or not messages:
        return None, get_data_error_result(
            code=RetCode.ARGUMENT_ERROR,
            message="`messages` must be a non-empty list.",
        )

    for message in messages:
        if not isinstance(message, dict):
            return None, get_data_error_result(
                code=RetCode.ARGUMENT_ERROR,
                message="Every item in `messages` must be an object.",
            )
        if "role" not in message or "content" not in message:
            return None, get_data_error_result(
                code=RetCode.ARGUMENT_ERROR,
                message="Every item in `messages` must include `role` and `content`.",
            )

    msg = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append(m)

    if not msg:
        return None, get_data_error_result(
            code=RetCode.ARGUMENT_ERROR,
            message="`messages` must contain a user message.",
        )
    if msg[-1]["role"] != "user":
        return None, get_data_error_result(
            code=RetCode.ARGUMENT_ERROR,
            message="The last message must be from user.",
        )
    if not msg[-1].get("id"):
        msg[-1]["id"] = get_uuid()

    # till now, message and msg are sharing the same copy
    return (messages, msg), None


async def _validate_llm_id(llm_id, tenant_id, llm_setting=None):
    if not llm_id:
        return None

    conf_model_type = (llm_setting or {}).get("model_type")
    if isinstance(conf_model_type, str):
        model_type = conf_model_type if conf_model_type in {"chat", "image2text"} else "chat"
    elif isinstance(conf_model_type, list):
        model_type = "image2text" if "image2text" in conf_model_type else "chat"
    else:
        model_type = "chat"
    try:
        await thread_pool_exec(
            get_model_config_from_provider_instance,
            tenant_id=tenant_id,
            model_name=llm_id,
            model_type=model_type,
        )
    except Exception as e:
        logging.error(f"Fail to get model config for {llm_id}: {e}")
        return f"`llm_id` {llm_id} doesn't exist"

    return None

async def _validate_rerank_id(rerank_id, tenant_id):
    if not rerank_id:
        return None
    parts = rerank_id.split('@')
    llm_name = parts[0]
    if llm_name in _DEFAULT_RERANK_MODELS:
        return None
    try:
        await thread_pool_exec(
            get_model_config_from_provider_instance,
            tenant_id=tenant_id,
            model_name=rerank_id,
            model_type="rerank",
        )
    except Exception as e:
        logging.error(f"Fail to get model config for {rerank_id}: {e}")
        return f"`rerank_id` {rerank_id} doesn't exist"
    return None


# def _validate_prompt_config(prompt_config):
#     for parameter in prompt_config.get("parameters", []):
#         if parameter.get("optional"):
#             continue
#         if prompt_config.get("system", "").find("{%s}" % parameter["key"]) < 0:
#             return f"Parameter '{parameter['key']}' is not used"
#     return None


async def _validate_dataset_ids(dataset_ids, tenant_id):
    if dataset_ids is None:
        return []
    if not isinstance(dataset_ids, list):
        return "`dataset_ids` should be a list."

    normalized_ids = [dataset_id for dataset_id in dataset_ids if dataset_id]
    kbs = []
    for dataset_id in normalized_ids:
        if not await thread_pool_exec(KnowledgebaseService.accessible, kb_id=dataset_id, user_id=tenant_id):
            return f"You don't own the dataset {dataset_id}"
        matches = await thread_pool_exec(KnowledgebaseService.query, id=dataset_id)
        if not matches:
            return f"You don't own the dataset {dataset_id}"
        kb = matches[0]
        if kb.chunk_num == 0:
            return f"The dataset {dataset_id} doesn't own parsed file"
        kbs.append(kb)

    embd_ids = [split_model_name(kb.embd_id)[0] for kb in kbs]
    if len(set(embd_ids)) > 1:
        return f'Datasets use different embedding models: {[kb.embd_id for kb in kbs]}'

    return normalized_ids


def _apply_prompt_defaults(req):
    prompt_config = req.setdefault("prompt_config", {})
    for key, value in _DEFAULT_PROMPT_CONFIG.items():
        temp = prompt_config.get(key)
        if (key == "system" and not temp) or key not in prompt_config:
            prompt_config[key] = deepcopy(value)

    if req.get("kb_ids") and not prompt_config.get("parameters") and "{knowledge}" in prompt_config.get("system", ""):
        prompt_config["parameters"] = [{"key": "knowledge", "optional": False}]


@manager.route("/chats", methods=["POST"])  # noqa: F821
@login_required
async def create():
    """
    创建一个新的对话配置，包括知识库、LLM 模型、检索参数等。
    """
    try:
        # 租户验证
        req = await get_request_json()
        ok, tenant = TenantService.get_by_id(current_user.id)
        if not ok:
            return get_data_error_result(message="Tenant not found!")

        # Validate tenant_id should not be provided
        # 禁止 tenant_id
        # 1.不允许用户手动指定租户 ID
        # 2.自动使用当前用户的租户 ID
        if req.get("tenant_id"):
            return get_data_error_result(message="`tenant_id` must not be provided.")

        # Validate name
        # 名称验证
        name, err = _validate_name(req.get("name"), required=True)
        if err:
            return get_data_error_result(message=err)
        req["name"] = name

        # 知识库验证
        # 1.验证知识库 ID 列表
        # 2.确保知识库属于当前用户
        # 3.映射 dataset_ids → kb_ids
        if "dataset_ids" in req:
            kb_ids = await _validate_dataset_ids(req.get("dataset_ids"), current_user.id)
            if isinstance(kb_ids, str):
                return get_data_error_result(message=kb_ids)
            req["kb_ids"] = kb_ids
            req.pop("dataset_ids", None)

        # LLM 模型验证
        # 1.验证 LLM 模型可用性
        # 2.检查模型配置
        if "llm_id" in req:
            err = await _validate_llm_id(req.get("llm_id"), current_user.id, req.get("llm_setting"))
            if err:
                return get_data_error_result(message=err)

        # 重排序模型验证
        # 1.验证重排序模型可用性
        if "rerank_id" in req:
            err = await _validate_rerank_id(req.get("rerank_id"), current_user.id)
            if err:
                return get_data_error_result(message=err)

        if "prompt_config" in req:
            if not isinstance(req["prompt_config"], dict):
                return get_data_error_result(message="`prompt_config` should be an object.")
            # err = _validate_prompt_config(req["prompt_config"])
            # if err:
            #     return get_data_error_result(message=err)

        # 设置默认值
        # 1.使用租户默认 LLM 模型
        # 2.设置检索参数默认值
        # 3.应用提示词默认配置
        req.setdefault("kb_ids", [])
        req.setdefault("llm_id", tenant.llm_id)
        if req["llm_id"] is None:
            req["llm_id"] = tenant.llm_id
        req.setdefault("llm_setting", {})
        req.setdefault("description", "A helpful Assistant")
        req.setdefault("top_n", 6)
        req.setdefault("top_k", 1024)
        req.setdefault("rerank_id", "")
        req.setdefault("similarity_threshold", 0.1)
        req.setdefault("vector_similarity_weight", 0.3)
        req.setdefault("icon", "")
        _apply_prompt_defaults(req)
        # err = _validate_prompt_config(req["prompt_config"])
        # if err:
        #     return get_data_error_result(message=err)

        # 字段过滤
        # 1.只保留允许存储的字段
        # 2.移除只读字段（如 create_time）
        req = {field: value for field, value in req.items() if field in _PERSISTED_FIELDS}
        for field in _READONLY_FIELDS:
            req.pop(field, None)

        # 名称唯一性检查
        # 1.确保同一租户下对话名称唯一
        # 2.只检查有效状态的对话
        if DialogService.query(
            name=req["name"],
            tenant_id=current_user.id,
            status=StatusEnum.VALID.value,
        ):
            return get_data_error_result(message="Duplicated chat name in creating chat.")

        # 保存对话
        # 1.生成唯一 ID
        # 2.设置租户 ID
        # 3.保存到数据库
        req["id"] = get_uuid()
        req["tenant_id"] = current_user.id
        if not DialogService.save(**req):
            return get_data_error_result(message="Failed to create chat.")

        # 返回结果
        # 1.获取创建的对话
        # 2.构建 API 响应格式
        ok, chat = DialogService.get_by_id(req["id"])
        if not ok:
            return get_data_error_result(message="Failed to retrieve created chat.")
        return get_json_result(data=_build_chat_response(chat))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats", methods=["GET"])  # noqa: F821
@login_required
async def list_chats():
    """
    获取当前用户有权限访问的对话列表，支持多种查询条件。
    """
    chat_id = request.args.get("id")    # 精确匹配对话 ID
    name = request.args.get("name")     # 精确匹配对话名称
    keywords = request.args.get("keywords", "") # 模糊搜索关键词
    orderby = request.args.get("orderby", "create_time") # 排序字段（默认 create_time）
    desc = request.args.get("desc", "true").lower() != "false" # 是否降序（默认 true）
    owner_ids = request.args.getlist("owner_ids") # 所有者租户 ID 列表
    exact_filters = {"id": chat_id, "name": name}
    if chat_id or name:
        keywords = ""

    try:
        # 分页参数
        # 分页从 1 开始（(page_number - 1) * items_per_page）
        page_number = int(request.args.get("page", 0)) # page=0：不分页，返回全部
        items_per_page = validate_rest_api_page_size(int(request.args.get("page_size", 0))) # page_size=0：返回全部

        # 有 owner_ids 的查询
        if owner_ids:
            # 先获取所有匹配的对话
            chats, total = await thread_pool_exec(
                DialogService.get_by_tenant_ids,
                owner_ids, current_user.id, 0, 0, orderby, desc, keywords, **exact_filters,
            )
            # 在应用层过滤 tenant_id
            chats = [chat for chat in chats if chat["tenant_id"] in owner_ids]
            total = len(chats)
            # 在应用层分页
            if page_number and items_per_page:
                start = (page_number - 1) * items_per_page
                chats = chats[start : start + items_per_page]
        # 无 owner_ids 的查询
        else:
            # 1.使用数据库分页
            # 2.只查询当前用户有权限的对话
            chats, total = await thread_pool_exec(
                DialogService.get_by_tenant_ids,
                [], current_user.id, page_number, items_per_page, orderby, desc, keywords, **exact_filters,
            )

        # 返回结果
        # 1.转换对话对象为 API 响应格式
        # 2.返回对话列表和总数
        return get_json_result(
            data={"chats": [_build_chat_response(chat) for chat in chats], "total": total}
        )
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>", methods=["GET"])  # noqa: F821
@login_required
async def get_chat(chat_id):
    """
    获取指定 ID 的对话的完整配置信息。
    """
    try:
        # 获取用户的所有租户
        # 1.查询当前用户加入的所有租户（团队）
        # 2.包括用户自己的租户和加入的团队
        tenants = await thread_pool_exec(UserTenantService.query, user_id=current_user.id)
        # 多租户权限校验
        # 1.遍历用户的所有租户
        # 2.检查对话是否属于某个租户
        # 3.如果找到，退出循环
        # 4.如果遍历完所有租户都未找到，返回认证错误
        for tenant in tenants:
            if await thread_pool_exec(
                DialogService.query,
                tenant_id=tenant.tenant_id, id=chat_id, status=StatusEnum.VALID.value,
            ):
                break
        else:
            return get_json_result(
                data=False,
                message="No authorization.",
                code=RetCode.AUTHENTICATION_ERROR,
            )

        # 获取对话详情
        # 1.从数据库获取对话信息
        # 2.不存在时返回错误
        # 3.构建 API 响应格式
        ok, chat = await thread_pool_exec(DialogService.get_by_id, chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found!")
        return get_json_result(data=_build_chat_response(chat))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>", methods=["PUT"])  # noqa: F821
@login_required
async def update_chat(chat_id):
    """
    更新指定对话的配置信息，支持部分更新。
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(
            data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR
        )

    try:
        # 获取当前对话信息
        req = await get_request_json()
        ok, tenant = TenantService.get_by_id(current_user.id)
        if not ok:
            return get_data_error_result(message="Tenant not found!")

        ok, current_chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found!")
        current_chat = current_chat.to_dict()

        # 禁止 tenant_id
        if req.get("tenant_id"):
            return get_data_error_result(message="`tenant_id` must not be provided.")

        # 名称验证
        if "name" in req:
            name, err = _validate_name(req.get("name"), required=True)
            if err:
                return get_data_error_result(message=err)
            req["name"] = name

        # 知识库验证
        if "dataset_ids" in req:
            kb_ids = await _validate_dataset_ids(req.get("dataset_ids"), current_user.id)
            if isinstance(kb_ids, str):
                return get_data_error_result(message=kb_ids)
            req["kb_ids"] = kb_ids
            req.pop("dataset_ids", None)

        # LLM 模型验证
        if "llm_id" in req:
            err = await _validate_llm_id(req.get("llm_id"), current_user.id, req.get("llm_setting"))
            if err:
                return get_data_error_result(message=err)

        # 重排序模型验证
        if "rerank_id" in req:
            err = await _validate_rerank_id(req.get("rerank_id"), current_user.id)
            if err:
                return get_data_error_result(message=err)

        # 提示词配置验证
        if "prompt_config" in req:
            if not isinstance(req["prompt_config"], dict):
                return get_data_error_result(message="`prompt_config` should be an object.")
            # err = _validate_prompt_config(req["prompt_config"])
            # if err:
            #     return get_data_error_result(message=err)

        # prompt_config = req.get("prompt_config", {})
        # if not prompt_config:
        #     prompt_config = current_chat.get("prompt_config", {})
        # kb_ids = req.get("kb_ids", current_chat.get("kb_ids", []))
        # if not kb_ids and not prompt_config.get("tavily_api_key") and _has_knowledge_placeholder(prompt_config):
        #     return get_data_error_result(message="Please remove `{knowledge}` in system prompt since no dataset / Tavily used here.")
        # 字段过滤
        # 1.只保留允许存储的字段
        # 2.移除只读字段
        req = {field: value for field, value in req.items() if field in _PERSISTED_FIELDS}
        for field in _READONLY_FIELDS:
            req.pop(field, None)

        # 名称唯一性检查
        # 1.只有名称变化时才检查
        # 2.确保同一租户下对话名称唯一
        if (
            "name" in req
            and req["name"].lower() != current_chat["name"].lower()
            and DialogService.query(
                name=req["name"],
                tenant_id=current_user.id,
                status=StatusEnum.VALID.value,
            )
        ):
            return get_data_error_result(message="Duplicated chat name.")

        # 保存更新
        # 1.执行更新操作
        # 2.更新失败时返回错误
        if not DialogService.update_by_id(chat_id, req):
            return get_data_error_result(message="Chat not found!")

        # 返回更新后的对话
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Failed to retrieve updated chat.")
        return get_json_result(data=_build_chat_response(chat))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>", methods=["PATCH"])  # noqa: F821
@login_required
async def patch_chat(chat_id):
    """
    部分更新对话配置，对嵌套字段（如 prompt_config、llm_setting）进行深度合并而非替换。
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(
            data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR
        )

    try:
        # 获取当前对话信息
        req = await get_request_json()
        ok, tenant = TenantService.get_by_id(current_user.id)
        if not ok:
            return get_data_error_result(message="Tenant not found!")

        ok, current_chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found!")
        current_chat = current_chat.to_dict()

        if "name" in req:
            name, err = _validate_name(req.get("name"), required=False)
            if err:
                return get_data_error_result(message=err)
            if name is not None:
                req["name"] = name

        if "dataset_ids" in req:
            kb_ids = await _validate_dataset_ids(req.get("dataset_ids"), current_user.id)
            if isinstance(kb_ids, str):
                return get_data_error_result(message=kb_ids)
            req["kb_ids"] = kb_ids
            req.pop("dataset_ids", None)

        if "llm_id" in req:
            err = await _validate_llm_id(req.get("llm_id"), current_user.id, req.get("llm_setting"))
            if err:
                return get_data_error_result(message=err)

        if "rerank_id" in req:
            err = await _validate_rerank_id(req.get("rerank_id"), current_user.id)
            if err:
                return get_data_error_result(message=err)

        # 深度合并：prompt_config
        # 1.关键差异：先复制现有配置，再更新传入的字段
        # 2.未传入的字段保持不变
        # 3.支持部分更新嵌套配置
        if "prompt_config" in req:
            if not isinstance(req["prompt_config"], dict):
                return get_data_error_result(message="`prompt_config` should be an object.")
            prompt_config = deepcopy(current_chat.get("prompt_config", {}))
            prompt_config.update(req["prompt_config"])
            req["prompt_config"] = prompt_config
            # err = _validate_prompt_config(prompt_config)
            # if err:
            #     return get_data_error_result(message=err)

        # 深度合并：llm_setting
        # 1.同样的深度合并策略
        # 2.只更新指定的 LLM 参数
        if "llm_setting" in req:
            llm_setting = deepcopy(current_chat.get("llm_setting", {}))
            llm_setting.update(req["llm_setting"])
            req["llm_setting"] = llm_setting

        # if "prompt_config" in req or "kb_ids" in req:
        #     prompt_config = req.get("prompt_config", current_chat.get("prompt_config", {}))
        #     kb_ids = req.get("kb_ids", current_chat.get("kb_ids", []))
        #     if not kb_ids and not prompt_config.get("tavily_api_key") and _has_knowledge_placeholder(prompt_config):
        #         return get_data_error_result(message="Please remove `{knowledge}` in system prompt since no dataset / Tavily used here.")

        # 字段过滤
        req = {field: value for field, value in req.items() if field in _PERSISTED_FIELDS}
        for field in _READONLY_FIELDS:
            req.pop(field, None)

        # 名称唯一性检查
        if (
            "name" in req
            and req["name"].lower() != current_chat["name"].lower()
            and DialogService.query(
                name=req["name"],
                tenant_id=current_user.id,
                status=StatusEnum.VALID.value,
            )
        ):
            return get_data_error_result(message="Duplicated chat name.")

        # 保存更新
        if not DialogService.update_by_id(chat_id, req):
            return get_data_error_result(message="Failed to update chat.")

        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Failed to retrieve updated chat.")
        return get_json_result(data=_build_chat_response(chat))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>", methods=["DELETE"])  # noqa: F821
@login_required
async def delete_chat(chat_id):
    """
    软删除指定的对话，将其状态设置为无效。
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(
            data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR
        )

    try:
        # 执行软删除
        # 1.更新对话的 status 字段为 "0"（无效/已废弃）
        # 2.软删除：数据仍然保留在数据库中
        # 3.更新失败时返回错误
        if not DialogService.update_by_id(chat_id, {"status": StatusEnum.INVALID.value}):
            return get_data_error_result(message=f"Failed to delete chat {chat_id}")
        return get_json_result(data=True)
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats", methods=["DELETE"])  # noqa: F821
@login_required
async def bulk_delete_chats():
    """
    批量删除对话，支持多种删除模式，并返回详细的执行结果。
    """
    req = await get_request_json()
    if not req:
        return get_json_result(data={})

    ids = req.get("ids")
    if not ids:
        # 模式1：删除全部
        # 1.获取当前用户的所有有效对话 ID
        # 2.没有对话时返回空结果
        if req.get("delete_all") is True:
            ids = [
                chat.id
                for chat in DialogService.query(
                    tenant_id=current_user.id, status=StatusEnum.VALID.value
                )
            ]
            if not ids:
                return get_json_result(data={})
        # 模式2：指定 ID 列表
        # 1.如果没有 ids，检查是否有 chat_id（旧版兼容）
        # 2.单条删除直接返回结果
        else:
            # keep backward compatibility, DELETE with chat_id in request body
            chat_id = req.get("chat_id")
            if chat_id:
                try:
                    if not DialogService.update_by_id(chat_id, {"status": StatusEnum.INVALID.value}):
                        return get_data_error_result(message=f"Failed to delete chat {chat_id}")
                    return get_json_result(data=True)
                except Exception as ex:
                    return server_error_response(ex)
            return get_json_result(data={})

    errors = []
    success_count = 0
    # ID 去重
    # 1.去除重复 ID
    # 2.记录重复错误信息
    unique_ids, duplicate_messages = check_duplicate_ids(ids, "chat")

    # 批量删除执行
    # 1.遍历所有 ID
    # 2.校验权限
    # 3.执行软删除
    # 4.统计成功数量
    for chat_id in unique_ids:
        if not await _ensure_owned_chat(chat_id):
            errors.append(f"Chat({chat_id}) not found.")
            continue
        success_count += DialogService.update_by_id(chat_id, {"status": StatusEnum.INVALID.value})

    # 返回结果
    all_errors = errors + duplicate_messages
    if all_errors:
        if success_count > 0:
            return get_json_result(
                data={"success_count": success_count, "errors": all_errors},
                message=f"Partially deleted {success_count} chats with {len(all_errors)} errors",
            )
        return get_data_error_result(message="; ".join(all_errors))

    return get_json_result(data={"success_count": success_count})


@manager.route("/chats/<chat_id>/sessions", methods=["POST"])  # noqa: F821
@login_required
async def create_session(chat_id):
    """Create a new conversation session for the given chat, owned by the authenticated user.
    为对话创建新的会话实例，包含开场白和会话元数据。
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)
    try:
        # 获取对话信息
        # 1.验证对话是否存在
        # 2.获取对话配置（用于开场白）
        req = await get_request_json()
        ok, dia = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found!")
        # 验证会话名称
        # 1.名称默认为 "New session"
        # 2.必须是字符串且不能为空
        # 3.长度限制为 255 字符
        name = req.get("name", "New session")
        if not isinstance(name, str) or not name.strip():
            return get_data_error_result(message="`name` can not be empty.")
        name = name.strip()[:255]
        # 构建会话对象
        conv = {
            "id": get_uuid(),
            "dialog_id": chat_id,                                                                   # 关联的对话 ID
            "name": name,
            "message": [{"role": "assistant", "content": dia.prompt_config.get("prologue", "")}],   # 初始消息，包含对话的开场白（prologue）
            "user_id": current_user.id,                                                             # 当前用户 ID
            "reference": [],                                                                        # 引用列表（初始为空）
        }
        # 保存会话
        # 1.保存会话到数据库
        # 2.重新获取以确认保存成功
        ConversationService.save(**conv)
        ok, conv_obj = ConversationService.get_by_id(conv["id"])
        if not ok:
            return get_data_error_result(message="Fail to create a session!")
        return get_json_result(data=_build_session_response(conv_obj.to_dict()))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>/sessions", methods=["GET"])  # noqa: F821
@login_required
async def list_sessions(chat_id):
    """
    获取指定对话下的会话列表，支持分页、排序和按 ID/名称/用户过滤。

    响应示例
    {
        "data": [
            {
                "id": "session_001",
                "name": "技术咨询-2024-01-15",
                "user_id": "user_001",
                "message": [
                    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
                    {"role": "user", "content": "Python是什么？"},
                    {"role": "assistant", "content": "Python是一种高级编程语言..."}
                ],
                "reference": [...],
                "create_time": "2024-01-15T10:00:00Z",
                "update_time": "2024-01-15T10:30:00Z"
            },
            {
                "id": "session_002",
                "name": "性能优化咨询",
                "user_id": "user_001",
                "message": [...],
                "reference": [...],
                "create_time": "2024-01-16T09:00:00Z",
                "update_time": "2024-01-16T09:45:00Z"
            }
        ],
        "code": 0,
        "message": "success"
    }
    """
    try:
        if not await _ensure_owned_chat(chat_id):
            return get_json_result(
                data=False,
                message="No authorization.",
                code=RetCode.AUTHENTICATION_ERROR,
            )
        # 分页与排序参数
        page_number = int(request.args.get("page", 1))  # 页码（从 1 开始）
        items_per_page = validate_rest_api_page_size(int(request.args.get("page_size", 30)))    # 每页数量（0 表示不分页）
        orderby = request.args.get("orderby", "create_time")    # 排序字段
        desc = request.args.get("desc", "true").lower() != "false"  # 是否降序

        # 过滤参数
        session_id = request.args.get("id") # 精确匹配会话 ID
        name = request.args.get("name") # 精确匹配会话名称
        user_id = request.args.get("user_id") # 按用户 ID 过滤
        # 查询会话列表
        # 1.调用 ConversationService.get_list 查询
        # 2.支持所有过滤和分页参数
        convs = ConversationService.get_list(
            chat_id, page_number, items_per_page, orderby, desc, session_id, name, user_id
        )
        # 处理不分页情况
        # 1.page_size=0 时返回空列表
        # 2.与 ConversationService.get_list 的行为保持一致
        if items_per_page == 0:
            convs = []
        return get_json_result(data=[_build_session_response(c) for c in convs])
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>/sessions/<session_id>", methods=["GET"])  # noqa: F821
@login_required
async def get_session(chat_id, session_id):
    """
    获取指定会话的完整信息，包括消息历史、引用来源和关联的对话头像。
    响应示例
    {
        "data": {
            "id": "session_xyz789",
            "name": "技术咨询-2024-01-15",
            "user_id": "user_001",
            "avatar": "data:image/png;base64,iVBORw0KGgo...",
            "message": [
                {
                    "role": "assistant",
                    "content": "你好！我是技术文档助手，有什么可以帮助你的？"
                },
                {
                    "role": "user",
                    "content": "Python是什么？"
                },
                {
                    "role": "assistant",
                    "content": "Python是一种高级编程语言，由Guido van Rossum于1991年创建..."
                }
            ],
            "reference": [
                {
                    "doc_id": "doc_001",
                    "chunk_id": "chunk_001",
                    "content": "Python是一种高级编程语言...",
                    "chunks": [
                        {
                            "id": "chunk_001",
                            "content": "Python是一种高级编程语言...",
                            "doc_id": "doc_001"
                        }
                    ]
                }
            ],
            "create_time": "2024-01-15T10:00:00Z",
            "update_time": "2024-01-15T10:30:00Z"
        },
        "code": 0,
        "message": "success"
    }
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)
    try:
        # 获取会话信息
        ok, conv = await thread_pool_exec(ConversationService.get_by_id, session_id)
        if not ok:
            return get_data_error_result(message="Session not found!")
        # 确保会话属于指定的对话
        if conv.dialog_id != chat_id:
            return get_data_error_result(message="Session does not belong to this chat!")
        # 获取对话头像
        # 1._ensure_owned_chat 返回对话信息
        # 2.提取对话的 icon 作为头像
        dialog = await _ensure_owned_chat(chat_id)
        avatar = dialog[0].icon if dialog else ""
        # 格式化引用
        # 1.遍历引用列表
        # 2.跳过列表类型的引用
        # 3.对字典类型的引用格式化 chunks 字段
        for ref in conv.reference:
            if isinstance(ref, list):
                continue
            ref["chunks"] = chunks_format(ref)
        # 构建响应
        # 1.构建会话响应格式
        # 2.添加对话头像
        # 3.返回完整信息
        result = _build_session_response(conv.to_dict())
        result["avatar"] = avatar
        return get_json_result(data=result)
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>/sessions/<session_id>", methods=["PATCH"])  # noqa: F821
@login_required
async def update_session(chat_id, session_id):
    """
    更新会话的可修改字段（主要是名称），保护不可变字段（消息和引用）。
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)
    try:
        req = await get_request_json()
        # 验证会话存在
        # 1.检查会话是否存在且属于该对话
        # 2.使用 query 方法验证
        if not ConversationService.query(id=session_id, dialog_id=chat_id):
            return get_data_error_result(message="Session not found!")
        # 保护不可变字段
        # 1.禁止修改 message / messages
        # 2.禁止修改 reference
        # 3.这些字段只能通过对话交互更新
        if "message" in req or "messages" in req:
            return get_data_error_result(message="`messages` cannot be changed.")
        if "reference" in req:
            return get_data_error_result(message="`reference` cannot be changed.")
        # 验证名称字段
        # 1.名称必须是字符串
        # 2.不能为空
        # 3.长度限制为 255 字符
        name = req.get("name")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                return get_data_error_result(message="`name` can not be empty.")
            req["name"] = name.strip()[:255]
        # 过滤可更新字段
        # 1.只允许更新非只读字段
        # 2.排除 id、dialog_id、chat_id、user_id
        update_fields = {k: v for k, v in req.items() if k not in {"id", "dialog_id", "chat_id", "user_id"}}
        # 执行更新
        # 1.更新会话信息
        # 2.更新失败时返回错误
        if not ConversationService.update_by_id(session_id, update_fields):
            return get_data_error_result(message="Session not found!")
        # 返回更新后的会话
        # 1.重新获取更新后的会话
        # 2.构建 API 响应
        ok, conv = ConversationService.get_by_id(session_id)
        if not ok:
            return get_data_error_result(message="Fail to update a session!")
        return get_json_result(data=_build_session_response(conv.to_dict()))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>/sessions", methods=["DELETE"])  # noqa: F821
@login_required
async def delete_sessions(chat_id):
    """
    批量删除对话下的会话，支持按 ID 列表或全部删除，并清理关联的附件文件。
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)
    try:
        req = await get_request_json()
        if not req:
            return get_json_result(data={})

        # 获取要删除的会话 ID 列表
        # 模式1：按 ids 列表删除
        # 模式2：delete_all=True 删除全部
        # 其他情况：返回空结果
        session_ids = req.get("ids")
        if not session_ids:
            if req.get("delete_all") is True:
                session_ids = [conv.id for conv in ConversationService.query(dialog_id=chat_id)]
                if not session_ids:
                    return get_json_result(data={})
            else:
                return get_json_result(data={})
        # ID 去重
        # 1.去除重复 ID
        # 2.记录重复错误
        unique_ids, duplicate_messages = check_duplicate_ids(session_ids, "session")
        # 批量删除执行
        # 1.遍历所有会话 ID
        # 2.验证会话属于该对话
        # 3.清理附件：遍历消息中的文件，从存储中删除
        # 4.删除会话记录
        errors = []
        success_count = 0
        for sid in unique_ids:
            if not ConversationService.query(id=sid, dialog_id=chat_id):
                errors.append(f"The chat doesn't own the session {sid}")
                continue
            # 清理附件文件
            # 1.删除会话前清理附件文件
            # 2.释放存储空间
            # 3.防止孤儿文件
            ok, conv = ConversationService.get_by_id(sid)
            if ok:
                for msg in conv.message or []:
                    for file in msg.get("files") or []:
                        file_id = file.get("id")
                        if not file_id:
                            continue
                        try:
                            settings.STORAGE_IMPL.rm(f"{current_user.id}-downloads", file_id)
                        except Exception:
                            logging.warning("Failed to delete chat upload blob %s/%s", current_user.id, file_id)
            ConversationService.delete_by_id(sid)
            success_count += 1
        all_errors = errors + duplicate_messages
        if all_errors:
            if success_count > 0:
                return get_json_result(
                    data={"success_count": success_count, "errors": all_errors},
                    message=f"Partially deleted {success_count} sessions with {len(all_errors)} errors",
                )
            return get_data_error_result(message="; ".join(all_errors))
        return get_json_result(data=True)
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>/sessions/<session_id>/messages/<msg_id>", methods=["DELETE"])  # noqa: F821
@login_required
async def delete_session_message(chat_id, session_id, msg_id):
    """
    删除会话中指定消息 ID 的消息以及其对应的助手回复，同时清理关联的引用。

    消息列表（成对出现）
    {
        "message": [
            {"id": "msg_001", "role": "user", "content": "Python是什么？"},
            {"id": "msg_001", "role": "assistant", "content": "Python是一种高级编程语言..."},
            {"id": "msg_002", "role": "user", "content": "它的优点是什么？"},
            {"id": "msg_002", "role": "assistant", "content": "Python的主要优点包括..."}
        ],
        "reference": [
            {"doc_id": "doc_001", "chunk_id": "chunk_001"},  // msg_001 的引用
            {"doc_id": "doc_002", "chunk_id": "chunk_003"}   // msg_002 的引用
        ]
    }

    删除 msg_001 后的结果
    {
        "message": [
            {"id": "msg_002", "role": "user", "content": "它的优点是什么？"},
            {"id": "msg_002", "role": "assistant", "content": "Python的主要优点包括..."}
        ],
        "reference": [
            {"doc_id": "doc_002", "chunk_id": "chunk_003"}
        ]
    }
    """
    if not await _ensure_owned_chat(chat_id):
        return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)
    try:
        ok, conv = ConversationService.get_by_id(session_id)
        # 验证会话存在且属于该对话
        if not ok or conv.dialog_id != chat_id:
            return get_data_error_result(message="Session not found!")
        conv = conv.to_dict()
        # 查找并删除消息对
        # 1.遍历消息列表，查找匹配的 msg_id
        # 2.关键假设：用户消息和助手回复成对出现
        # 3.assert 验证下一条消息的 ID 相同（确保是同一对）
        # 4.删除用户消息和助手回复（两条）
        # 5.删除对应的引用（i // 2 - 1）
        for i, msg in enumerate(conv["message"]):
            if msg_id != msg.get("id", ""):
                continue
            assert conv["message"][i + 1]["id"] == msg_id
            conv["message"].pop(i)
            conv["message"].pop(i)
            # 删除对应的引用
            # 索引计算：i // 2 - 1（每对消息对应一个引用）
            # max(0, ...) 防止负索引
            conv["reference"].pop(max(0, i // 2 - 1))
            break
        # 保存更新
        ConversationService.update_by_id(conv["id"], conv)
        return get_json_result(data=_build_session_response(conv))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chats/<chat_id>/sessions/<session_id>/messages/<msg_id>/feedback", methods=["PUT"])  # noqa: F821
@login_required
async def update_message_feedback(chat_id, session_id, msg_id):
    """
    更新助手消息的点赞/点踩状态，并将反馈传播到引用的文档分块，用于优化检索排序。

    反馈状态转换
    之前状态	新状态	操作
    None	True	点赞 → 应用积极反馈
    None	False	点踩 → 应用消极反馈
    True	False	取消点赞 → 移除积极反馈
    False	True	取消点踩 → 移除消极反馈
    """
    owned = await _ensure_owned_chat(chat_id)
    if not owned:
        return get_json_result(data=False, message="No authorization.", code=RetCode.AUTHENTICATION_ERROR)
    try:
        req = await get_request_json()
        ok, conv = ConversationService.get_by_id(session_id)
        # 验证会话存在且属于该对话
        if not ok or conv.dialog_id != chat_id:
            return get_data_error_result(message="Session not found!")
        # thumbup 必须是布尔值
        thumb_raw = req.get("thumbup")
        if not isinstance(thumb_raw, bool):
            return get_data_error_result(message="thumbup must be a boolean")
        # feedback 是可选文本反馈
        feedback = req.get("feedback", "")

        # 查找目标消息
        # 1.只处理 role == "assistant" 的消息
        # 2.记录之前的反馈状态
        # 3.更新反馈状态
        # 4.决定是否需要应用分块反馈
        conv_dict = conv.to_dict()
        message_index = None
        apply_chunk_feedback = False
        prior_thumb = None
        for i, msg in enumerate(conv_dict["message"]):
            if msg_id == msg.get("id", "") and msg.get("role", "") == "assistant":
                prior_thumb = msg.get("thumbup") # 原有的反馈状态
                if thumb_raw is True: # 该次是点赞
                    msg["thumbup"] = True
                    msg.pop("feedback", None)
                    apply_chunk_feedback = prior_thumb is not True  # 之前是点踩或者没操作，则会应用积极反馈
                else: # 该次是点踩
                    msg["thumbup"] = False
                    if feedback:
                        msg["feedback"] = feedback
                    apply_chunk_feedback = prior_thumb is not False # 之前是点赞或者没操作，则会应用消极反馈
                message_index = i
                break

        # 应用分块反馈
        if message_index is not None and apply_chunk_feedback:
            try:
                ref_index = (message_index - 1) // 2
                if 0 <= ref_index < len(conv_dict.get("reference", [])):
                    reference = conv_dict["reference"][ref_index]
                    """
                    状态转换总结
                    之前状态 (prior_thumb)	新状态 (thumb_raw)	第一次 apply_feedback	第二次 apply_feedback	说明
                    True (点赞)	            False (点踩)	        撤销积极反馈	            应用消极反馈	            状态变更时，先撤销旧的，再应用新的
                    False (点踩)          	True (点赞)	        撤销消极反馈	            应用积极反馈	            同上
                    None	                True 或 False	    不执行	                应用对应的积极/消极反馈	    从无反馈直接变为有反馈
                    """
                    if reference:
                        if isinstance(prior_thumb, bool) and prior_thumb != thumb_raw:
                            await thread_pool_exec(
                                ChunkFeedbackService.apply_feedback,
                                tenant_id=current_user.id,
                                reference=reference,
                                is_positive=not prior_thumb,
                            )
                        # thumb_raw 为点赞，会应用积极反馈。
                        feedback_result = await thread_pool_exec(
                            ChunkFeedbackService.apply_feedback,
                            tenant_id=current_user.id,
                            reference=reference,
                            is_positive=thumb_raw is True,
                        )
                        logging.debug(
                            "Chunk feedback applied: %s succeeded, %s failed",
                            feedback_result["success_count"],
                            feedback_result["fail_count"],
                        )
            except Exception as e:
                logging.warning("Failed to apply chunk feedback: %s", e)

        #  保存更新
        await thread_pool_exec(ConversationService.update_by_id, conv_dict["id"], conv_dict)
        return get_json_result(data=_build_session_response(conv_dict))
    except Exception as ex:
        return server_error_response(ex)


@manager.route("/chat/audio/speech", methods=["POST"])  # noqa: F821
@login_required
async def tts():
    """
    将输入文本转换为语音，以流式方式返回 MP3 音频数据。
    """
    req = await get_request_json()
    text = req["text"]

    try:
        # 获取租户默认的 TTS 模型配置
        default_tts_model_config = get_tenant_default_model_by_type(current_user.id, LLMType.TTS)
    except Exception as e:
        return get_data_error_result(message=str(e))

    # 创建 LLMBundle 实例
    tts_mdl = LLMBundle(current_user.id, default_tts_model_config)

    # 流式音频生成
    # 1.按标点符号分割文本（中文和英文标点）
    # 2.逐段调用 TTS 模型生成音频
    # 3.流式返回音频数据块
    # 4.错误时返回 SSE 错误消息
    def stream_audio():
        try:
            for txt in re.split(r"[，。/《》？；：！\n\r:;]+", text):
                for chunk in tts_mdl.tts(txt):
                    yield chunk
        except Exception as e:
            yield ("data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e)}}, ensure_ascii=False)).encode("utf-8")

    resp = Response(stream_audio(), mimetype="audio/mpeg")
    resp.headers.add_header("Cache-Control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    return resp


@manager.route("/chat/audio/transcription", methods=["POST"])  # noqa: F821
@login_required
async def transcription():
    """
    将上传的音频文件转换为文本，支持流式和非流式转录。

    非流式响应
    {
        "data": {
            "text": "你好，这是RAGFlow的语音识别功能。"
        },
        "code": 0,
        "message": "success"
    }

    流式响应
    data: {"event": "progress", "text": "你好"}
    data: {"event": "progress", "text": "你好，这是"}
    data: {"event": "progress", "text": "你好，这是RAGFlow的语音识别功能。"}
    data: {"event": "done", "text": "你好，这是RAGFlow的语音识别功能。"}
    """
    # 解析请求
    # 1.使用 multipart/form-data 格式
    # 2.支持 stream 参数（true 或 false）
    # 3.必须包含 file 字段
    req = await request.form
    stream_mode = req.get("stream", "false").lower() == "true"
    files = await request.files
    if "file" not in files:
        return get_data_error_result(message="Missing 'file' in multipart form-data")

    uploaded = files["file"]

    # 验证音频格式
    ALLOWED_EXTS = {
        ".wav", ".mp3", ".m4a", ".aac",
        ".flac", ".ogg", ".webm",
        ".opus", ".wma",
    }

    filename = uploaded.filename or ""
    suffix = os.path.splitext(filename)[-1].lower()
    if suffix not in ALLOWED_EXTS:
        return get_data_error_result(
            message=f"Unsupported audio format: {suffix}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        )

    # 保存临时文件
    # 1.创建临时文件
    # 2.保存上传的音频文件
    fd, temp_audio_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    await uploaded.save(temp_audio_path)

    # 获取 ASR 模型：获取租户默认的语音识别模型
    try:
        default_asr_model_config = get_tenant_default_model_by_type(current_user.id, LLMType.SPEECH2TEXT)
    except Exception as e:
        return get_data_error_result(message=str(e))

    asr_mdl = LLMBundle(current_user.id, default_asr_model_config)
    # 非流式模式
    # 1.一次完整的转录
    # 2.返回完整文本
    # 3.清理临时文件
    if not stream_mode:
        text = asr_mdl.transcription(temp_audio_path)
        try:
            os.remove(temp_audio_path)
        except Exception as e:
            logging.error(f"Failed to remove temp audio file: {str(e)}")
        return get_json_result(data={"text": text})

    # 流式模式
    # 1.流式返回转录结果
    # 2.使用 Server-Sent Events（SSE）
    # 3.实时输出转录片段
    async def event_stream():
        try:
            for evt in asr_mdl.stream_transcription(temp_audio_path):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"event": "error", "text": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                logging.error(f"Failed to remove temp audio file: {str(e)}")

    return Response(event_stream(), content_type="text/event-stream")


@manager.route("/chat/mindmap", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
async def mindmap():
    """
    根据用户问题，从知识库中检索相关信息，生成思维导图。
    """
    req = await get_request_json()
    search_id = req.get("search_id", "")

    # 加载搜索配置
    # 如果提供了 search_id，加载对应的搜索配置
    search_app = SearchService.get_detail(search_id) if search_id else {}
    # 搜索配置可能包含默认的知识库、检索参数等
    search_config = search_app.get("search_config", {}) if search_app else {}
    # 合并知识库 ID
    kb_ids = search_config.get("kb_ids", [])
    # 合并搜索配置中的知识库 ID 和请求中的知识库 ID
    kb_ids.extend(req["kb_ids"])
    # 去重
    kb_ids = list(set(kb_ids))

    # 生成思维导图
    # 1.调用 gen_mindmap 生成思维导图
    # 2.传入问题、知识库 ID、租户 ID 和搜索配置
    mind_map = await gen_mindmap(req["question"], kb_ids, search_app.get("tenant_id", current_user.id), search_config)
    if "error" in mind_map:
        return server_error_response(Exception(mind_map["error"]))
    return get_json_result(data=mind_map)


@manager.route("/chat/recommendation", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question")
async def recommendation():
    """
    根据用户输入的问题，使用 LLM 生成相关的搜索推荐词，帮助用户扩展查询。
    """
    req = await get_request_json()

    search_id = req.get("search_id", "")
    search_config = {}
    if search_id:
        if search_app := SearchService.get_detail(search_id):
            search_config = search_app.get("search_config", {})

    question = req["question"]

    # 初始化 Chat 模型
    # 1.优先使用搜索配置中的 Chat 模型
    # 2.否则使用租户默认模型
    chat_id = search_config.get("chat_id", "")
    if chat_id:
        chat_model_config = get_model_config_from_provider_instance(current_user.id, LLMType.CHAT, chat_id)
    else:
        chat_model_config = get_tenant_default_model_by_type(current_user.id, LLMType.CHAT)
    chat_mdl = LLMBundle(current_user.id, chat_model_config)

    # 生成推荐词
    # 1.使用较高的 temperature（0.9）增加多样性
    # 2.加载 related_question 提示词模板
    # 3.调用 LLM 生成相关搜索词
    gen_conf = search_config.get("llm_setting", {"temperature": 0.9})
    if "parameter" in gen_conf:
        del gen_conf["parameter"]
    prompt = load_prompt("related_question")
    ans = await chat_mdl.async_chat(
        prompt,
        [
            {
                "role": "user",
                "content": f"\nKeywords: {question}\nRelated search terms:\n    ",
            }
        ],
        gen_conf,
    )
    # 解析结果
    # 1.按换行分割结果
    # 2.筛选以 数字. 开头的行
    # 3.移除编号前缀
    # 4.返回推荐词列表
    return get_json_result(data=[re.sub(r"^[0-9]\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]\. ", a)])


@manager.route("/chat/completions", methods=["POST"])  # noqa: F821
@login_required
async def session_completion(chat_id_in_arg=""):
    """Handle chat completion requests, streaming or non-streaming, scoped to the authenticated user.
    处理用户的对话补全请求，支持会话管理、消息历史、流式/非流式响应。
    """
    # 消息规范化
    # 1.将请求中的消息格式规范化
    # 2.确保消息格式正确
    req = await get_request_json()
    normalized, error = _normalize_completion_messages(req)
    if error:
        return error
    request_messages, request_msg = normalized
    # 参数解析
    pass_all_history_messages = _get_bool_request_flag(req, "pass_all_history_messages", "pass_all_history", default=False) # 是否传递全部历史消息
    msg = request_msg
    message_id = request_msg[-1].get("id")
    chat_id = req.pop("chat_id", "") or "" # 对话 ID
    chat_id = chat_id or chat_id_in_arg
    session_id = req.pop("session_id", "") or req.pop("conversation_id", "") or "" # 会话 ID
    chat_model_id = req.pop("llm_id", "") # 使用的 LLM 模型 ID

    # 模型配置：提取生成配置（temperature、top_p 等）
    chat_model_config = pop_generation_config(req)

    try:
        conv = None
        # 会话管理
        # 1.验证 chat_id 和 session_id 的关系
        # 2.如果 session_id 不存在，自动创建新会话
        if session_id and not chat_id:
            return get_data_error_result(message="`chat_id` is required when `session_id` is provided.")

        if chat_id:
            if not await _ensure_owned_chat(chat_id):
                return get_json_result(
                    data=False,
                    message="No authorization.",
                    code=RetCode.AUTHENTICATION_ERROR,
                )
            e, dia = await thread_pool_exec(DialogService.get_by_id, chat_id)
            if not e:
                return get_data_error_result(message="Chat not found!")
            if session_id:
                e, conv = await thread_pool_exec(ConversationService.get_by_id, session_id)
                if not e:
                    return get_data_error_result(message="Session not found!")
                if conv.dialog_id != chat_id:
                    return get_data_error_result(message="Session does not belong to this chat!")
            else:
                conv = await _create_session_for_completion(chat_id, dia, current_user.id)
                session_id = conv.id

            # 消息处理
            # 1.pass_all_history_messages=True：使用全部历史消息
            # 2.否则只追加最新消息
            if pass_all_history_messages:
                conv.message = deepcopy(request_messages)
                msg = request_msg
            else:
                if not conv.message:
                    conv.message = []
                conv.message.append(deepcopy(request_msg[-1]))
                msg = []
                for m in conv.message:
                    if m["role"] == "system":
                        continue
                    if m["role"] == "assistant" and not msg:
                        continue
                    msg.append(m)
        else:
            dia = _build_default_completion_dialog()

        req.pop("messages", None)
        req.pop("question", None)

        if conv is not None:
            if not conv.reference:
                conv.reference = []
            conv.reference = [r for r in conv.reference if r]
            conv.reference.append({"chunks": [], "doc_aggs": []})

        # 模型选择
        # 1.优先使用用户指定的模型
        # 2.否则使用租户默认模型
        if chat_model_id:
            if not await thread_pool_exec(get_api_key, tenant_id=dia.tenant_id, model_name=chat_model_id):
                return get_data_error_result(message=f"Cannot use specified model {chat_model_id}.")
            dia.llm_id = chat_model_id
            dia.llm_setting = chat_model_config
        elif not dia.llm_id:
            logging.info("empty chat_model_id in req, use default chat model.")
            _, tenant_info = TenantService.get_by_id(dia.tenant_id)
            if not tenant_info or not tenant_info.llm_id:
                raise LookupError("No default chat model for tenant.")
            dia.llm_id = tenant_info.llm_id
            merge_generation_config(dia, chat_model_config)

        stream_mode = req.pop("stream", True)

        def _format_answer(ans):
            """Wrap a raw answer dict with session and chat identifiers."""
            formatted = structure_answer(conv, ans, message_id, session_id)
            if chat_id:
                formatted["chat_id"] = chat_id
            return formatted

        async def stream():
            """Yield SSE-formatted chunks from the async chat generator."""
            nonlocal dia, msg, req, conv
            try:
                async for ans in async_chat(dia, msg, True, session_id=session_id, **req):
                    ans = _format_answer(ans)
                    payload = _sanitize_json_floats({"code": 0, "message": "", "data": ans})
                    yield "data:" + json.dumps(payload, ensure_ascii=False) + "\n\n"
                if conv is not None:
                    await thread_pool_exec(ConversationService.update_by_id, conv.id, conv.to_dict())
            except Exception as ex:
                logging.exception(ex)
                yield "data:" + json.dumps({"code": 500, "message": str(ex), "data": {"answer": "**ERROR**: " + str(ex), "reference": []}}, ensure_ascii=False) + "\n\n"
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

        # 流式响应
        if stream_mode:
            resp = Response(stream(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        # 非流式响应
        answer = None
        async for ans in async_chat(dia, msg, False, session_id=session_id, **req):
            answer = _format_answer(ans)
            if conv is not None:
                await thread_pool_exec(ConversationService.update_by_id, conv.id, conv.to_dict())
            break
        return get_json_result(data=_sanitize_json_floats(answer))
    except Exception as ex:
        return server_error_response(ex)
