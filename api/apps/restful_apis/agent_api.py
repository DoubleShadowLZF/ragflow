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
import base64
import copy
import hashlib
import hmac
import inspect
import ipaddress
import json
import logging
import time
from functools import partial, wraps
from typing import Set

from api.utils.web_utils import CONTENT_TYPE_MAP, apply_safe_file_response_headers
import jwt
from quart import Response, jsonify, request, make_response

from api.apps import current_user, login_required
from api.apps.services.canvas_replica_service import CanvasReplicaService
from api.db import CanvasCategory
from api.db.db_models import Task
from api.db.services.api_service import API4ConversationService
from api.db.services.canvas_service import (
    CanvasTemplateService,
    UserCanvasService,
    completion as agent_completion,
    completion_openai,
)
from api.db.services.document_service import DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.pipeline_operation_log_service import PipelineOperationLogService
from api.db.services.task_service import CANVAS_DEBUG_DOC_ID, TaskService, queue_dataflow
from api.db.services.user_service import TenantService, UserService
from api.db.services.user_canvas_version import UserCanvasVersionService
from api.utils.api_utils import (
    add_tenant_id_to_kwargs,
    check_duplicate_ids,
    get_data_error_result,
    get_error_data_result,
    get_json_result,
    get_result,
    get_request_json,
    server_error_response,
    validate_request,
)
from api.utils.pagination_utils import validate_rest_api_page_size
from common import settings
from common.ssrf_guard import assert_host_is_safe
from common.constants import RetCode
from common.misc_utils import get_uuid, thread_pool_exec
from peewee import MySQLDatabase, PostgresqlDatabase

# Keeps strong references to fire-and-forget tasks so they are not GC'd before completion.
_background_tasks: Set[asyncio.Task] = set()


def _require_canvas_access_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not UserCanvasService.accessible(kwargs.get('agent_id'), kwargs.get('tenant_id')):
            return get_json_result(data=False, message="Make sure you have permission to access the agent.", code=RetCode.OPERATING_ERROR)
        return func(*args, **kwargs)
    return wrapper


def _require_canvas_access_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        agent_id = kwargs.get('agent_id')
        tenant_id = kwargs.get('tenant_id')
        if not await thread_pool_exec(UserCanvasService.accessible, agent_id, tenant_id):
            return get_json_result(data=False, message="Make sure you have permission to access the agent.", code=RetCode.OPERATING_ERROR)
        return await func(*args, **kwargs)
    return wrapper


def _require_canvas_owner_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not UserCanvasService.query(user_id=kwargs.get('tenant_id'), id=kwargs.get('agent_id')):
            return get_json_result(data=False, message="Only the owner of the agent is authorized for this operation.", code=RetCode.OPERATING_ERROR)
        return func(*args, **kwargs)
    return wrapper


def _get_user_nickname(user_id: str) -> str:
    exists, user = UserService.get_by_id(user_id)
    if not exists:
        return user_id
    return str(getattr(user, "nickname", "") or user_id)


def _build_sse_response(body):
    resp = Response(body, mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp


def _normalize_agent_reference_entry(reference):
    if not isinstance(reference, dict):
        return {"chunks": [], "doc_aggs": []}
    if "chunks" in reference or "doc_aggs" in reference:
        return {
            "chunks": reference.get("chunks", []),
            "doc_aggs": reference.get("doc_aggs", []),
        }
    return {
        "chunks": reference.get("reference", reference.get("chunks", [])) or [],
        "doc_aggs": reference.get("doc_aggs", []) or [],
    }


def _normalize_agent_reference_chunk(chunk):
    if not isinstance(chunk, dict):
        return {
            "id": chunk,
            "content": str(chunk),
            "document_id": None,
            "document_name": None,
            "dataset_id": None,
            "image_id": None,
            "positions": None,
        }

    return {
        "id": chunk.get("chunk_id", chunk.get("id")),
        "content": chunk.get("content_with_weight", chunk.get("content")),
        "document_id": chunk.get("doc_id", chunk.get("document_id")),
        "document_name": chunk.get("docnm_kwd", chunk.get("document_name")),
        "dataset_id": chunk.get("kb_id", chunk.get("dataset_id")),
        "image_id": chunk.get("image_id", chunk.get("img_id")),
        "positions": chunk.get("positions", chunk.get("position_int")),
    }


def _normalize_agent_session(conv):
    conv["message"] = conv.get("message", [])
    for info in conv["message"]:
        if "prompt" in info:
            info.pop("prompt")
    conv["agent_id"] = conv.pop("dialog_id")
    if isinstance(conv["reference"], dict):
        if "chunks" in conv["reference"]:
            conv["reference"] = [conv["reference"]]
        else:
            conv["reference"] = [value for _, value in sorted(conv["reference"].items(), key=lambda item: int(item[0]))]
    elif isinstance(conv["reference"], list):
        conv["reference"] = [_normalize_agent_reference_entry(reference) for reference in conv["reference"]]
    else:
        conv["reference"] = []

    if conv["reference"]:
        messages = [message for i, message in enumerate(conv["message"]) if i != 0 and message["role"] != "user"]
        for message, reference in zip(messages, conv["reference"]):
            chunks = reference.get("chunks", [])
            message["reference"] = [_normalize_agent_reference_chunk(chunk) for chunk in chunks]
    del conv["reference"]
    return conv


def _agent_session_list_result(data, total):
    return jsonify({"code": RetCode.SUCCESS, "message": "success", "data": data, "total": total})


async def _run_workflow_session(
    tenant_id,
    agent_id,
    workflow_conv,
    canvas,
    query,
    files,
    inputs,
    user_id,
    session_id,
    custom_header,
    canvas_title,
    canvas_category,
    return_trace,
    stream,
    chat_template_kwargs=None,
):
    async def commit_runtime_replica():
        commit_ok = CanvasReplicaService.commit_after_run(
            canvas_id=agent_id,
            tenant_id=str(tenant_id),
            runtime_user_id=user_id,
            dsl=json.loads(str(canvas)),
            canvas_category=canvas_category,
            title=canvas_title,
        )
        if not commit_ok:
            logging.error(
                "Canvas runtime replica commit failed: canvas_id=%s tenant_id=%s runtime_user_id=%s",
                agent_id,
                tenant_id,
                user_id,
            )

    workflow_conv.setdefault("message", [])
    if isinstance(workflow_conv.get("reference"), dict):
        if "chunks" in workflow_conv["reference"]:
            workflow_conv["reference"] = [workflow_conv["reference"]]
        else:
            workflow_conv["reference"] = [
                value for _, value in sorted(workflow_conv["reference"].items(), key=lambda item: int(item[0]))
            ]
    elif not isinstance(workflow_conv.get("reference"), list):
        workflow_conv["reference"] = []
    workflow_conv["reference"] = [_normalize_agent_reference_entry(reference) for reference in workflow_conv["reference"]]

    turn_id = workflow_conv["message"][-1].get("id") if workflow_conv["message"] else get_uuid()
    full_content = ""
    reference = {}
    final_ans = {}
    trace_items = []
    structured_output = {}
    run_kwargs = {
        "query": query,
        "files": files,
        "user_id": user_id,
        "inputs": inputs,
    }
    if chat_template_kwargs is not None:
        run_kwargs["chat_template_kwargs"] = chat_template_kwargs

    async def persist_workflow_session():
        if not final_ans:
            return
        workflow_conv["message"].append(
            {
                "role": "assistant",
                "content": full_content,
                "created_at": time.time(),
                "id": turn_id,
            }
        )
        workflow_conv["reference"].append(_normalize_agent_reference_entry(reference))
        workflow_conv["dsl"] = json.loads(str(canvas))
        workflow_conv["source"] = workflow_conv.get("source") or "workflow"
        await thread_pool_exec(API4ConversationService.append_message, session_id, workflow_conv)
        await commit_runtime_replica()

    if stream:

        async def sse():
            nonlocal full_content, reference, final_ans, trace_items, structured_output
            done_sent = False
            try:
                async for ans in canvas.run(**run_kwargs):
                    ans["session_id"] = session_id
                    if ans.get("event") == "message":
                        full_content += ans.get("data", {}).get("content", "")
                    if ans.get("data", {}).get("reference", None):
                        reference.update(ans["data"]["reference"])
                    if ans.get("event") == "node_finished":
                        data = ans.get("data", {})
                        node_out = data.get("outputs", {})
                        component_id = data.get("component_id")
                        if component_id is not None and "structured" in node_out:
                            structured_output[component_id] = copy.deepcopy(node_out["structured"])
                        if return_trace:
                            trace_items.append(
                                {
                                    "component_id": data.get("component_id"),
                                    "trace": [copy.deepcopy(data)],
                                }
                            )
                    final_ans = ans
                    yield "data:" + json.dumps(ans, ensure_ascii=False) + "\n\n"

                if final_ans:
                    if "data" not in final_ans or not isinstance(final_ans["data"], dict):
                        final_ans["data"] = {}
                    final_ans["data"]["content"] = full_content
                    final_ans["data"]["reference"] = reference
                    if structured_output:
                        final_ans["data"]["structured"] = structured_output
                    if trace_items:
                        final_ans["data"]["trace"] = trace_items
                await persist_workflow_session()
            except Exception as exc:
                logging.exception(exc)
                canvas.cancel_task()
                yield (
                    "data:"
                    + json.dumps({"code": 500, "message": str(exc), "data": False}, ensure_ascii=False)
                    + "\n\n"
                )
            finally:
                if not done_sent:
                    done_sent = True
                    yield "data:[DONE]\n\n"

        return _build_sse_response(sse())

    try:
        async for ans in canvas.run(**run_kwargs):
            ans["session_id"] = session_id
            if ans.get("event") == "message":
                full_content += ans.get("data", {}).get("content", "")
            if ans.get("data", {}).get("reference", None):
                reference.update(ans["data"]["reference"])
            if ans.get("event") == "node_finished":
                data = ans.get("data", {})
                node_out = data.get("outputs", {})
                component_id = data.get("component_id")
                if component_id is not None and "structured" in node_out:
                    structured_output[component_id] = copy.deepcopy(node_out["structured"])
                if return_trace:
                    trace_items.append(
                        {
                            "component_id": data.get("component_id"),
                            "trace": [copy.deepcopy(data)],
                        }
                    )
            final_ans = ans
    except Exception as exc:
        logging.exception(exc)
        canvas.cancel_task()
        return get_result(data=f"**ERROR**: {str(exc)}")

    if not final_ans:
        await commit_runtime_replica()
        return get_result(data={})

    if "data" not in final_ans or not isinstance(final_ans["data"], dict):
        final_ans["data"] = {}
    final_ans["data"]["content"] = full_content
    final_ans["data"]["reference"] = reference
    if structured_output:
        final_ans["data"]["structured"] = structured_output
    if trace_items:
        final_ans["data"]["trace"] = trace_items

    await persist_workflow_session()
    return get_result(data=final_ans)


@manager.route("/agents/<agent_id>/sessions", methods=["GET"])  # noqa: F821
@login_required # 要求用户登录
@add_tenant_id_to_kwargs    # 自动注入租户 ID
@_require_canvas_access_sync    # 同步校验用户对 Agent 的访问权限
def list_agent_sessions(agent_id, tenant_id):
    """
    获取指定 Agent 下的所有会话记录，支持多种过滤条件、分页和排序。
    :param agent_id:
    :param tenant_id:
    :return:
    """
    session_id = request.args.get("id")                         # 精确查询会话ID
    user_id = request.args.get("user_id")                       # 按用户过滤
    page_number = int(request.args.get("page", 1))              # 页码（默认1）
    items_per_page = validate_rest_api_page_size(               # 每页数量（默认30）
        int(request.args.get("page_size", 30))
    )
    keywords = request.args.get("keywords")                     # 关键词搜索
    from_date = request.args.get("from_date")                   # 开始日期
    to_date = request.args.get("to_date")                       # 结束日期
    orderby = request.args.get("orderby", "update_time")        # 排序字段（默认更新时间）
    exp_user_id = request.args.get("exp_user_id")               # 实验用户ID
    desc = request.args.get("desc") not in {"False", "false"}   # 是否降序

    # 实验用户模式（快速返回）
    # 实验用户：只返回会话名称列表（不含完整内容）
    # 优化目的：A/B 测试场景下快速获取会话概览
    # 不进行分页：直接返回所有匹配会话
    if exp_user_id:
        sessions = API4ConversationService.get_names(agent_id, exp_user_id)
        return _agent_session_list_result(sessions, len(sessions))

    include_dsl = request.args.get("dsl") not in {"False", "false"}  # 是否包含DSL
    # 正常查询模式
    total, sessions = API4ConversationService.get_list(
        agent_id,
        tenant_id,
        page_number,
        items_per_page,
        orderby,
        desc,
        session_id,
        user_id,
        include_dsl,
        keywords,
        from_date,
        to_date,
        exp_user_id=exp_user_id,
    )
    # 数据规范化
    # _normalize_agent_session：字段映射和格式转换
    # _agent_session_list_result：统一响应格式
    sessions = [_normalize_agent_session(session) for session in sessions]
    return _agent_session_list_result(sessions, total)


@manager.route("/agents/<agent_id>/sessions", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def create_agent_session(agent_id, tenant_id):
    """
    核心功能:为一个 Agent（智能体）创建一个新的会话实例，包括初始化对话状态、设置 DSL（领域特定语言）、记录会话元数据等。
    """
    from agent.canvas import Canvas

    # 参数解析
    req = await get_request_json()
    user_id = req.get("user_id") or request.args.get("user_id", tenant_id)      # 会话所属用户（默认使用租户 ID）
    release_mode = bool(req.get("release", request.args.get("release", False))) #是否使用已发布的 Agent 版本

    # 获取 Agent DSL
    #
    # get_agent_dsl_with_release 的作用：
    # 参数	                说明
    # release_mode=False	获取草稿版本（最新编辑的内容）
    # release_mode=True	    获取已发布的稳定版本
    #
    # 返回值：
    # cvs：Canvas 对象（包含 Agent 元信息）
    # dsl：DSL 配置（定义 Agent 的行为逻辑）
    try:
        cvs, dsl = UserCanvasService.get_agent_dsl_with_release(agent_id, release_mode, tenant_id)
    except LookupError:
        return get_data_error_result(message="Agent not found.")
    except PermissionError as e:
        return get_data_error_result(message=str(e))

    # 初始化 Canvas
    # 1.生成新的会话 ID
    # 2.创建 Canvas 实例并重置状态（清空对话历史）
    # 3.将 Canvas 对象序列化为 JSON DSL
    session_id = get_uuid()
    canvas = Canvas(dsl, tenant_id, agent_id, canvas_id=cvs.id)
    canvas.reset()

    cvs.dsl = json.loads(str(canvas))
    # 获取版本标题
    # 1.记录当前使用的 Agent 版本标题
    # 2.便于后续追溯和版本管理
    version_title = UserCanvasVersionService.get_latest_version_title(cvs.id, release_mode=release_mode)
    # 建会话对象
    #
    # 字段说明：
    # 字段	        值           	说明
    # id	        新生成的 UUID	会话唯一标识
    # name	        用户传入或空 	    会话名称
    # dialog_id	    Canvas ID	    关联的 Agent
    # user_id	    用户 ID	        会话所有者
    # message	    Agent 开场白	    初始助手指令
    # source	    "agent"     	标记为 Agent 来源
    # dsl	        Canvas DSL	    保存当前 Agent 配置快照
    # version_title	版本标题	        记录使用的版本
    conv = {
        "id": session_id,
        "name": req.get("name", ""),
        "dialog_id": cvs.id,
        "user_id": user_id,
        "exp_user_id": user_id,
        "message": [{"role": "assistant", "content": canvas.get_prologue()}],
        "source": "agent",
        "dsl": cvs.dsl,
        "reference": [],
        "version_title": version_title,
    }
    API4ConversationService.save(**conv)
    return get_result(data=_normalize_agent_session(conv))


@manager.route("/agents/<agent_id>/sessions/<session_id>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_sync
def get_agent_session(agent_id, session_id, tenant_id):
    """
    根据 Agent ID 和会话 ID，用于获取指定 Agent 下某个会话的完整详情。它支持开发者或前端应用获取一个已存在的 Agent 会话的所有信息，包括对话历史、引用来源和执行元数据。
    """
    exists, conv = API4ConversationService.get_by_id(session_id)
    if not exists:
        return get_data_error_result(message="Session not found!")
    return get_json_result(data=conv.to_dict())


@manager.route("/agents/<agent_id>/sessions/<session_id>", methods=["DELETE"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_sync
def delete_agent_session_item(agent_id, session_id, tenant_id):
    """
    根据 Agent ID 和会话 ID，删除指定的会话记录。
    """
    return get_json_result(data=API4ConversationService.delete_by_id(session_id))


@manager.route("/agents/<agent_id>/sessions", methods=["DELETE"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def delete_agent_session(tenant_id, agent_id):
    """
    用于批量删除指定 Agent 下的多个会话。它支持按 ID 列表删除或一键删除所有会话，并返回详细的执行结果。
    """
    # 变量初始化
    # 收集删除失败的错误信息
    errors = []
    # 统计成功删除的数量
    success_count = 0
    req = await get_request_json()
    # Agent 存在性校验
    # 1.验证用户是否拥有该 Agent
    # 2.防止越权操作
    cvs = await thread_pool_exec(UserCanvasService.query, user_id=tenant_id, id=agent_id)
    if not cvs:
        return get_error_data_result(f"You don't own the agent {agent_id}")

    if not req:
        return get_result()

    # 解析删除目标
    # 删除模式：
    #
    # 模式	    触发条件	            行为
    # 指定 ID	ids 存在	            删除列表中指定的会话
    # 删除全部	delete_all == True	删除 Agent 下所有会话
    # 无操作	    两者都不满足	        直接返回成功
    ids = req.get("ids")
    if not ids:
        if req.get("delete_all") is True:
            ids = [conv.id for conv in await thread_pool_exec(API4ConversationService.query, dialog_id=agent_id)]
            if not ids:
                return get_result()
        else:
            return get_result()

    # ID 去重
    # 1.去除重复的会话 ID
    # 2.记录重复 ID 的错误信息
    conv_list = ids

    unique_conv_ids, duplicate_messages = check_duplicate_ids(conv_list, "session")
    conv_list = unique_conv_ids

    # 批量删除执行
    for session_id in conv_list:
        # 查询会话是否存在且属于该 Agent
        conv = await thread_pool_exec(API4ConversationService.query, id=session_id, dialog_id=agent_id)
        # 如果不满足条件，记录错误
        if not conv:
            errors.append(f"The agent doesn't own the session {session_id}")
            continue
        # 如果满足条件，执行删除
        await thread_pool_exec(API4ConversationService.delete_by_id, session_id)
        # 成功计数 +1
        success_count += 1

    # 结果返回
    # 返回逻辑：
    #
    # 情况	    响应类型	                    说明
    # 全部成功	get_result()	            无错误信息
    # 部分成功	get_result()	            包含成功数和错误列表
    # 全部失败	get_error_data_result()	    返回错误信息
    if errors:
        if success_count > 0:
            return get_result(data={"success_count": success_count, "errors": errors},
                              message=f"Partially deleted {success_count} sessions with {len(errors)} errors")
        else:
            return get_error_data_result(message="; ".join(errors))

    if duplicate_messages:
        if success_count > 0:
            return get_result(
                message=f"Partially deleted {success_count} sessions with {len(duplicate_messages)} errors",
                data={"success_count": success_count, "errors": duplicate_messages})
        else:
            return get_error_data_result(message=";".join(duplicate_messages))

    return get_result()


@manager.route("/agents/download", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def download_agent_file(tenant_id):
    """
    根据文件 ID 从对象存储中获取文件内容，并以二进制流的形式返回给客户端。
    """
    id = request.args.get("id")
    logging.info("Agent file download requested: tenant_id=%s file_id=%s", tenant_id, id)
    blob = await thread_pool_exec(FileService.get_blob, tenant_id, id)
    return Response(blob)


async def _iter_session_completion_events(tenant_id, agent_id, req, return_trace):
    # Stream and non-stream session completions share the same event parsing and trace injection.
    trace_items = []
    async for answer in agent_completion(tenant_id=tenant_id, agent_id=agent_id, **req):
        if isinstance(answer, str):
            try:
                ans = json.loads(answer[5:])
            except Exception:
                continue
        else:
            ans = answer

        event = ans.get("event")
        if event == "node_finished":
            if return_trace:
                data = ans.get("data", {})
                trace_items.append(
                    {
                        "component_id": data.get("component_id"),
                        "trace": [copy.deepcopy(data)],
                    }
                )
                ans.setdefault("data", {})["trace"] = trace_items
            yield ans
            continue

        if event in ["message", "message_end"]:
            yield ans


@manager.route("/agents/templates", methods=["GET"])  # noqa: F821
@login_required
def list_agent_template():
    """
    获取所有 Agent 模板的列表，返回模板的名称、描述、配置等信息。
    """
    return get_json_result(data=[item.to_dict() for item in CanvasTemplateService.get_all()])


@manager.route("/agents/prompts", methods=["GET"])  # noqa: F821
@login_required
def prompts():
    """
    返回 Agent 工作流中各个阶段使用的系统级提示词模板，供前端调试或展示。
    """
    from rag.prompts.generator import (
        ANALYZE_TASK_SYSTEM,
        ANALYZE_TASK_USER,
        CITATION_PROMPT_TEMPLATE,
        NEXT_STEP,
        REFLECT,
    )

    return get_json_result(
        data={
            "task_analysis": f"{ANALYZE_TASK_SYSTEM}\n\n{ANALYZE_TASK_USER}",
            "plan_generation": NEXT_STEP,
            "reflection": REFLECT,
            "citation_guidelines": CITATION_PROMPT_TEMPLATE,
        }
    )


@manager.route("/agents", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def list_agents(tenant_id):
    """
    返回用户可访问的 Agent 列表，支持按关键词、所有者、标签、分类过滤，并支持分页和排序。
    :param tenant_id:
    :return:
    """
    keywords = request.args.get("keywords", "")  # 关键词搜索
    canvas_category = request.args.get("canvas_category")  # Agent分类
    owner_ids = [item for item in request.args.get("owner_ids", "").strip().split(",") if item]  # 所有者过滤
    tags = [item for item in request.args.get("tags", "").strip().split(",") if item]  # 标签过滤

    page_number = int(request.args.get("page", 0))  # 页码（从0开始）
    items_per_page = validate_rest_api_page_size(int(request.args.get("page_size", 0)))  # 每页数量
    order_by = request.args.get("orderby", "create_time")  # 排序字段
    desc = str(request.args.get("desc", "true")).lower() != "false"  # 是否降序
    # 权限与所有者过滤
    # 1.获取用户加入的所有租户（团队）
    # 2.加上用户自己的租户 ID
    # 3.构建授权所有者集合
    tenants = TenantService.get_joined_tenants_by_user_id(tenant_id)
    authorized_owner_ids = {member["tenant_id"] for member in tenants}
    authorized_owner_ids.add(tenant_id)

    # 所有者过滤校验：防止越权访问
    if owner_ids:
        requested_owner_ids = set(owner_ids)
        unauthorized_owner_ids = requested_owner_ids - authorized_owner_ids
        if unauthorized_owner_ids:
            return get_json_result(
                data=False,
                message="Only authorized owner_ids can be queried.",
                code=RetCode.OPERATING_ERROR,
            )
        effective_owner_ids = list(requested_owner_ids)
    else:
        effective_owner_ids = list(authorized_owner_ids)

    # 查询 Agent 列表
    # canvas：Agent 列表数据
    # total：匹配的总记录数（用于分页）
    canvas, total = UserCanvasService.get_by_tenant_ids(
        effective_owner_ids,
        tenant_id,
        page_number,
        items_per_page,
        order_by,
        desc,
        keywords,
        canvas_category,
        tags,
    )

    return get_json_result(data={"canvas": canvas, "total": total})


@manager.route("/agents/tags", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def list_agent_tags(tenant_id):
    """Aggregate tag usage counts across agents visible to the caller."""
    canvas_category = request.args.get("canvas_category")
    tenants = TenantService.get_joined_tenants_by_user_id(tenant_id)
    joined_ids = list({member["tenant_id"] for member in tenants} | {tenant_id})
    counts = UserCanvasService.list_tags(joined_ids, tenant_id, canvas_category)
    logging.info(
        "list_agent_tags tenant=%s canvas_category=%s tags_count=%d",
        tenant_id,
        canvas_category,
        len(counts),
    )
    return get_json_result(data=[{"tag": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))])


@manager.route("/agents/<canvas_id>/tags", methods=["PUT"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def update_agent_tags(tenant_id, canvas_id):
    """
    更新指定 Agent 的标签列表，支持批量标签管理。
    """
    # 权限校验
    # 检查用户是否有权访问该 Agent
    # 无权限时记录日志并返回错误
    if not UserCanvasService.accessible(canvas_id, tenant_id):
        logging.info(
            "update_agent_tags denied tenant=%s canvas_id=%s reason=no_permission",
            tenant_id,
            canvas_id,
        )
        return get_json_result(
            data=False,
            message="Agent not found or no permission.",
            code=RetCode.OPERATING_ERROR,
        )
    # 参数解析与规范化
    # 支持两种输入格式：
    #
    # 格式	    示例	                处理方式
    # 列表	    ["生产", "客服"]	    直接使用
    # 字符串	    "生产,客服"	        按逗号分割
    req = await get_request_json()
    tags = req.get("tags", "")
    incoming = tags if isinstance(tags, (list, tuple)) else [t for t in str(tags).split(",") if t.strip()]
    # 执行更新
    rows_affected = UserCanvasService.update_tags(canvas_id, tags)
    # 结果校验与日志
    # 返回逻辑：
    #
    # 情况	                响应	        说明
    # rows_affected > 0	    data=True	更新成功
    # rows_affected == 0	data=False	未找到或无权更新
    if rows_affected == 0:
        logging.info(
            "update_agent_tags miss tenant=%s canvas_id=%s incoming_count=%d rows=0",
            tenant_id,
            canvas_id,
            len(incoming),
        )
        return get_json_result(
            data=False,
            message="Agent not found or no permission.",
            code=RetCode.OPERATING_ERROR,
        )
    logging.info(
        "update_agent_tags ok tenant=%s canvas_id=%s incoming_count=%d rows=%d",
        tenant_id,
        canvas_id,
        len(incoming),
        rows_affected,
    )
    return get_json_result(data=True)


@manager.route("/agents", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def create_agent(tenant_id):
    """
    创建一个新的 Agent，包括保存 DSL 配置、创建版本记录、同步到 Replica 系统，并返回创建的 Agent 信息。
    """
    # 过滤掉 None 值（避免覆盖默认值）
    req = {k: v for k, v in (await get_request_json()).items() if v is not None}
    req["canvas_type"] = req.get("canvas_type","")
    # 设置 user_id 为当前租户 ID
    req["user_id"] = tenant_id
    # 默认 canvas_category 为 Agent
    req["canvas_category"] = req.get("canvas_category") or CanvasCategory.Agent
    # release 转为布尔值
    req["release"] = bool(req.get("release", ""))

    # DSL 校验与规范化
    # 1.确保 dsl 存在
    # 2.规范化 DSL 格式（确保节点、边、工具等结构正确）
    # 3.校验失败时返回具体错误
    if req.get("dsl") is None:
        return get_json_result(
            data=False,
            message="No DSL data in request.",
            code=RetCode.ARGUMENT_ERROR,
        )

    try:
        req["dsl"] = CanvasReplicaService.normalize_dsl(req["dsl"])
    except ValueError as exc:
        return get_json_result(
            data=False,
            message=str(exc),
            code=RetCode.ARGUMENT_ERROR,
        )

    # 标题校验与去重
    # 1.标题不能为空
    # 2.同一租户下不能有同名 Agent
    if req.get("title") is None:
        return get_json_result(
            data=False,
            message="No title in request.",
            code=RetCode.ARGUMENT_ERROR,
        )

    req["title"] = req["title"].strip()
    if UserCanvasService.query(
        user_id=tenant_id,
        title=req["title"],
        canvas_category=req["canvas_category"],
    ):
        return get_data_error_result(message=f"{req['title']} already exists.")

    # 保存 Agent
    # 1.生成唯一 ID
    # 2.保存到数据库
    req["id"] = get_uuid()
    if not UserCanvasService.save(**req):
        return get_data_error_result(message="Fail to create agent.")

    # 创建版本快照
    # 1.获取所有者昵称
    # 2.构建版本标题：{昵称} - {Agent名称}
    # 3.保存最新版本快照
    owner_nickname = _get_user_nickname(tenant_id)
    UserCanvasVersionService.save_or_replace_latest(
        user_canvas_id=req["id"],
        title=UserCanvasVersionService.build_version_title(owner_nickname, req.get("title")),
        dsl=req["dsl"],
        release=req.get("release"),
    )
    # 同步到 Replica 系统
    # 1.将 Agent 同步到 Replica 系统（用于分布式执行）
    # 2.同步失败时返回错误（但 Agent 已保存）
    replica_ok = CanvasReplicaService.replace_for_set(
        canvas_id=req["id"],
        tenant_id=str(tenant_id),
        runtime_user_id=str(tenant_id),
        dsl=req["dsl"],
        canvas_category=req["canvas_category"],
        title=req.get("title", ""),
    )
    if not replica_ok:
        return get_data_error_result(message="canvas saved, but replica sync failed.")

    # 查询并返回创建的 Agent
    # 1.从数据库查询刚创建的 Agent
    # 2.返回完整的 Agent 信息
    exists, created_agent = UserCanvasService.get_by_canvas_id(req["id"])
    if not exists:
        return get_data_error_result(message="Fail to create agent.")
    return get_json_result(data=created_agent)


@manager.route("/agents/<agent_id>/upload", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def upload_agent_file(agent_id, tenant_id):
    """
    为指定的 Agent 上传一个或多个文件，支持通过 URL 抓取文件，并返回文件的元数据信息。
    """
    # 解析上传的文件
    # 1.从请求中获取文件列表
    # 2.支持 file 字段的多个文件上传
    # 3.如果没有文件，返回空列表
    files = await request.files
    file_objs = files.getlist("file") if files and files.get("file") else []
    # 日志记录
    # 1.记录上传请求的关键信息
    # 2.便于审计和问题排查
    logging.info(
        "Agent file upload requested: tenant_id=%s agent_id=%s file_count=%s",
        tenant_id,
        agent_id,
        len(file_objs),
    )
    try:
        # 文件上传处理
        # 1.单文件使用 thread_pool_exec 执行
        # 2.支持通过 url 参数抓取远程文件
        if len(file_objs) == 1:
            uploaded = await thread_pool_exec(
                FileService.upload_info, tenant_id, file_objs[0], request.args.get("url")
            )
            return get_json_result(data=uploaded)
        # 多文件上传
        # 1.多文件使用 asyncio.gather 并发上传
        # 2.每个文件独立处理，互不影响
        results = await asyncio.gather(
            *(thread_pool_exec(FileService.upload_info, tenant_id, file_obj) for file_obj in file_objs)
        )
        return get_json_result(data=results)
    # 错误处理
    # 1.捕获所有异常并记录
    # 2.返回服务器错误响应
    except Exception as exc:
        logging.exception(
            "Agent file upload failed: tenant_id=%s agent_id=%s",
            tenant_id,
            agent_id,
        )
        return server_error_response(exc)


@manager.route("/agents/<agent_id>/components/<component_id>/input-form", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_sync
def get_agent_component_input_form(agent_id, component_id, tenant_id):
    try:
        from agent.canvas import Canvas

        exists, user_canvas = UserCanvasService.get_by_id(agent_id)
        if not exists:
            return get_data_error_result(message="canvas not found.")
        canvas = Canvas(json.dumps(user_canvas.dsl), tenant_id, canvas_id=user_canvas.id)
        return get_json_result(data=canvas.get_component_input_form(component_id))
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/agents/<agent_id>/components/<component_id>/debug", methods=["POST"])  # noqa: F821
@validate_request("params")
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def debug_agent_component(agent_id, component_id, tenant_id):
    """
    在调试模式下执行 Agent 工作流中的指定组件，注入测试参数，并返回组件的执行结果。
    """
    req = await get_request_json()
    try:
        from agent.canvas import Canvas
        from agent.component import LLM

        # 获取 Agent 配置
        # 从数据库获取 Agent 的 DSL 配置
        _, user_canvas = UserCanvasService.get_by_id(agent_id)
        # 创建 Canvas 实例并重置状态
        canvas = Canvas(json.dumps(user_canvas.dsl), tenant_id, canvas_id=user_canvas.id)
        canvas.reset()
        # 生成新的消息 ID（用于追踪）
        canvas.message_id = get_uuid()
        # 获取目标组件
        # 1.根据组件 ID 从 Canvas 中获取组件对象
        # 2.重置组件状态（清空缓存等）
        component = canvas.get_component(component_id)["obj"]
        component.reset()

        # 注入测试参数
        # 1.如果是 LLM 组件，调用 set_debug_inputs 设置调试输入
        # 2.调用组件的 invoke 方法，传入参数
        if isinstance(component, LLM):
            component.set_debug_inputs(req["params"])
        # 参数结构化
        component.invoke(**{k: o["value"] for k, o in req["params"].items()})
        # 异步输出处理
        # 1.获取组件输出
        # 2.如果输出是 partial 类型（流式输出）：
        # 1).检测是异步生成器还是同步生成器
        # 2).遍历并拼接所有输出片段
        # 3.将流式输出合并为完整字符串
        outputs = component.output()
        for k in outputs.keys():
            if isinstance(outputs[k], partial):
                txt = ""
                iter_obj = outputs[k]()
                # 流式输出支持
                if inspect.isasyncgen(iter_obj):
                    async for c in iter_obj:
                        txt += c
                else:
                    for c in iter_obj:
                        txt += c
                outputs[k] = txt
        # 返回结果
        return get_json_result(data=outputs)
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/agents/<agent_id>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
def get_agent(agent_id, tenant_id):
    """
    获取指定 Agent 的完整信息，包括 DSL 配置、发布时间、关联数据集等，并确保 Replica 系统同步。
    """
    # 权限校验
    # 1.检查用户是否有权访问该 Agent
    # 2.无权访问时返回 404 错误
    if not UserCanvasService.accessible(agent_id, tenant_id):
        return get_data_error_result(message="canvas not found.")

    # 获取 Agent 数据
    # 1.从数据库查询 Agent 信息
    # 2.不存在时返回 404 错误
    exists, canvas = UserCanvasService.get_by_canvas_id(agent_id)
    if not exists:
        return get_data_error_result(message="canvas not found.")

    try:
        # Replica 系统同步
        # 1.确保 Replica 系统中有该 Agent 的配置
        # 2.用于分布式执行环境
        # 3.同步失败时返回错误
        CanvasReplicaService.bootstrap(
            canvas_id=agent_id,
            tenant_id=str(tenant_id),
            runtime_user_id=str(tenant_id),
            dsl=canvas.get("dsl"),
            canvas_category=canvas.get("canvas_category", CanvasCategory.Agent),
            title=canvas.get("title", ""),
        )
    except ValueError as exc:
        return get_data_error_result(message=str(exc))

    # 计算发布时间
    # 1.获取 Agent 的所有版本
    # 2.筛选已发布的版本（release=True）
    # 3.按更新时间降序排列
    # 4.取最新发布版本的时间
    last_publish_time = None
    versions = UserCanvasVersionService.list_by_canvas_id(agent_id)
    if versions:
        released_versions = [version for version in versions if version.release]
        if released_versions:
            released_versions.sort(key=lambda version: version.update_time, reverse=True)
            last_publish_time = released_versions[0].update_time

    from agent.dsl_migration import normalize_chunker_dsl
    # DSL 迁移
    # 1.对 DSL 进行规范化/迁移
    # 2.确保 DSL 格式与当前版本兼容
    # 3.处理不同版本间的数据结构变化
    canvas["dsl"] = normalize_chunker_dsl(canvas.get("dsl", {}))
    # 添加发布时间到响应：将计算出的发布时间添加到返回数据中
    canvas["last_publish_time"] = last_publish_time

    # DataFlow 特殊处理
    # 1.如果是 DataFlow 类型（数据流/管道）
    # 2.查询关联的数据集（知识库）
    # 3.返回数据集 ID、名称、头像
    if canvas.get("canvas_category") == CanvasCategory.DataFlow:
        datasets = list(KnowledgebaseService.query(pipeline_id=agent_id))
        canvas["datasets"] = [{"id": item.id, "name": item.name, "avatar": item.avatar} for item in datasets]

    return get_json_result(data=canvas)


@manager.route("/agents/<agent_id>/versions", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_sync
def list_agent_versions(agent_id, tenant_id):
    """
    获取指定 Agent 的所有版本快照，包括每个版本的 DSL 配置、标题、发布时间等信息。
    """
    try:
        versions = sorted(
            [item.to_dict() for item in UserCanvasVersionService.list_by_canvas_id(agent_id)],
            key=lambda item: item["update_time"] * -1,
        )
        return get_json_result(data=versions)
    except Exception as exc:
        return get_data_error_result(message=f"Error getting history files: {exc}")


@manager.route("/agents/<agent_id>/versions/<version_id>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_sync
def get_agent_version(agent_id, version_id, tenant_id):
    """
    根据 Agent ID 和版本 ID，返回该版本的所有信息，包括 DSL 配置、版本标题、发布时间等。
    """
    try:
        exists, version = UserCanvasVersionService.get_by_id(version_id)
        if not exists or not version or str(version.user_canvas_id) != str(agent_id):
            return get_data_error_result(message="Version not found.")
        return get_json_result(data=version.to_dict())
    except Exception as exc:
        return get_data_error_result(message=f"Error getting history file: {exc}")


@manager.route("/agents/<agent_id>/logs/<message_id>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def get_agent_logs(agent_id, message_id, tenant_id):
    """
    根据 Agent ID 和消息 ID，从 Redis 中获取该消息的执行日志，用于调试和监控 Agent 的执行过程。
    """
    try:
        from rag.utils.redis_conn import REDIS_CONN

        binary = await thread_pool_exec(REDIS_CONN.get, f"{agent_id}-{message_id}-logs")
        if not binary:
            return get_json_result(data={})

        payload = binary.decode("utf-8") if isinstance(binary, bytes) else binary
        return get_json_result(data=json.loads(payload))
    except Exception as exc:
        logging.exception(exc)
        return server_error_response(exc)


@manager.route("/agents/<agent_id>", methods=["DELETE"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_owner_sync
def delete_agent(agent_id, tenant_id):
    """
    删除指定 ID 的 Agent 及其所有关联数据（版本历史、Replica 同步记录等）。
    """
    UserCanvasService.delete_by_id(agent_id)
    return get_json_result(data=True)


@manager.route("/agents/<agent_id>", methods=["PUT"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def update_agent(agent_id, tenant_id):
    """
    更新 Agent 的配置信息，包括标题、DSL、分类、发布状态等，并同步更新版本历史和 Replica 系统。
    """
    # 过滤掉 None 值（避免覆盖默认值）
    req = {k: v for k, v in (await get_request_json()).items() if v is not None}
    # canvas_type 默认为空字符串
    req["canvas_type"] = req.get("canvas_type","")
    # release 转为布尔值
    req["release"] = bool(req.get("release", ""))

    if req.get("dsl") is not None:
        try:
            # DSL 校验与规范化
            req["dsl"] = CanvasReplicaService.normalize_dsl(req["dsl"])
        except ValueError as exc:
            return get_json_result(
                data=False,
                message=str(exc),
                code=RetCode.ARGUMENT_ERROR,
            )

    # 标题处理
    if req.get("title") is not None:
        req["title"] = req["title"].strip()

    # 获取当前 Agent 信息
    # 1.获取当前的 Agent 信息（用于版本标题）
    # 2.如果更新了标题，使用新标题
    # 3.否则使用原标题
    _, current_agent = UserCanvasService.get_by_id(agent_id)
    agent_title_for_version = req.get("title") or (current_agent.title if current_agent else "")
    canvas_category = (
        req.get("canvas_category")
        or (current_agent.canvas_category if current_agent else CanvasCategory.Agent)
    )
    owner_nickname = _get_user_nickname(tenant_id)
    # 更新 Agent
    # 1.更新数据库中的 Agent 记录
    # 2.只更新传入的字段
    UserCanvasService.update_by_id(agent_id, req)

    # DSL 更新后的同步操作
    # 1.只有更新 DSL 时才执行版本和 Replica 同步
    # 2.版本标题格式：{昵称} - {Agent名称}
    # 3.Replica 同步失败时返回错误
    if req.get("dsl") is not None:
        # 1. 创建版本快照
        UserCanvasVersionService.save_or_replace_latest(
            user_canvas_id=agent_id,
            title=UserCanvasVersionService.build_version_title(owner_nickname, agent_title_for_version),
            dsl=req["dsl"],
            release=req.get("release"),
        )

        # 2. 同步到 Replica 系统
        replica_ok = CanvasReplicaService.replace_for_set(
            canvas_id=agent_id,
            tenant_id=str(tenant_id),
            runtime_user_id=str(tenant_id),
            dsl=req["dsl"],
            canvas_category=canvas_category,
            title=agent_title_for_version,
        )
        if not replica_ok:
            return get_data_error_result(message="agent saved, but replica sync failed.")

    return get_json_result(data=True)


@manager.route("/agents/<agent_id>/reset", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
@_require_canvas_access_async
async def reset_agent(agent_id, tenant_id):
    try:
        from agent.canvas import Canvas

        exists, user_canvas = UserCanvasService.get_by_id(agent_id)
        if not exists:
            return get_data_error_result(message="canvas not found.")

        canvas = Canvas(json.dumps(user_canvas.dsl), tenant_id, canvas_id=user_canvas.id)
        canvas.reset()
        dsl = json.loads(str(canvas))
        UserCanvasService.update_by_id(agent_id, {"dsl": dsl})
        replica_ok = CanvasReplicaService.replace_for_set(
            canvas_id=agent_id,
            tenant_id=str(tenant_id),
            runtime_user_id=str(tenant_id),
            dsl=dsl,
            canvas_category=user_canvas.canvas_category,
            title=user_canvas.title,
        )
        if not replica_ok:
            return get_data_error_result(message="agent reset, but replica sync failed.")
        return get_json_result(data=dsl)
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/agents/rerun", methods=["POST"])  # noqa: F821
@validate_request("id", "dsl", "component_id")
@login_required
@add_tenant_id_to_kwargs
async def rerun_agent(tenant_id):
    from rag.nlp import search

    req = await get_request_json()
    doc = PipelineOperationLogService.get_documents_info(req["id"])
    if not doc:
        return get_data_error_result(message="Document not found.")
    doc = doc[0]
    if 0 < doc["progress"] < 1:
        return get_data_error_result(message=f"`{doc['name']}` is processing...")

    if settings.docStoreConn.index_exist(search.index_name(tenant_id), doc["kb_id"]):
        settings.docStoreConn.delete({"doc_id": doc["id"]}, search.index_name(tenant_id), doc["kb_id"])
    doc["progress_msg"] = ""
    doc["chunk_num"] = 0
    doc["token_num"] = 0
    DocumentService.clear_chunk_num_when_rerun(doc["id"])
    DocumentService.update_by_id(doc["id"], doc)
    TaskService.filter_delete([Task.doc_id == doc["id"]])

    dsl = req["dsl"]
    dsl["path"] = [req["component_id"]]
    PipelineOperationLogService.update_by_id(req["id"], {"dsl": dsl})
    queue_dataflow(
        tenant_id=tenant_id,
        flow_id=req["id"],
        task_id=get_uuid(),
        doc_id=doc["id"],
        priority=0,
        rerun=True,
    )
    return get_json_result(data=True)


@manager.route("/agents/test_db_connection", methods=["POST"])  # noqa: F821
@validate_request("db_type", "database", "username", "host", "port", "password")
@login_required
async def test_db_connection():
    req = await get_request_json()
    try:
        safe_host = assert_host_is_safe(req["host"])
    except ValueError as exc:
        logging.warning(
            "Rejected test_db_connection: unsafe host %r (db_type=%s, user=%s): %s",
            req.get("host"), req.get("db_type"), current_user.id, exc,
        )
        return get_data_error_result(message=str(exc))
    except OSError as exc:
        logging.warning(
            "Rejected test_db_connection: cannot resolve host %r (db_type=%s, user=%s): %s",
            req.get("host"), req.get("db_type"), current_user.id, exc,
        )
        logging.debug("Full resolver exception for host %r", req.get("host"), exc_info=True)
        return get_data_error_result(message=f"Could not resolve host {req.get('host')!r}.")
    try:
        if req["db_type"] in ["mysql", "mariadb"]:
            db = MySQLDatabase(
                req["database"],
                user=req["username"],
                host=safe_host,
                port=req["port"],
                password=req["password"],
            )
            with db.connection_context():
                db.execute_sql("SELECT 1")
        elif req["db_type"] == "oceanbase":
            db = MySQLDatabase(
                req["database"],
                user=req["username"],
                host=safe_host,
                port=req["port"],
                password=req["password"],
                charset="utf8mb4",
            )
            with db.connection_context():
                db.execute_sql("SELECT 1")
        elif req["db_type"] == "postgres":
            db = PostgresqlDatabase(
                req["database"],
                user=req["username"],
                host=safe_host,
                port=req["port"],
                password=req["password"],
            )
            with db.connection_context():
                db.execute_sql("SELECT 1")
        elif req["db_type"] == "mssql":
            import pyodbc

            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={safe_host},{req['port']};"
                f"DATABASE={req['database']};"
                f"UID={req['username']};"
                f"PWD={req['password']};"
            )
            db = pyodbc.connect(connection_string)
            try:
                cursor = db.cursor()
                try:
                    cursor.execute("SELECT 1")
                finally:
                    cursor.close()
            finally:
                db.close()
        elif req["db_type"] == "IBM DB2":
            import ibm_db

            conn_str = (
                f"DATABASE={req['database']};"
                f"HOSTNAME={safe_host};"
                f"PORT={req['port']};"
                f"PROTOCOL=TCPIP;"
                f"UID={req['username']};"
                f"PWD={req['password']};"
            )
            logging.info(
                "DATABASE=%s;HOSTNAME=%s;PORT=%s;PROTOCOL=TCPIP;UID=%s;PWD=****;",
                req["database"],
                safe_host,
                req["port"],
                req["username"],
            )
            conn = ibm_db.connect(conn_str, "", "")
            stmt = ibm_db.exec_immediate(conn, "SELECT 1 FROM sysibm.sysdummy1")
            ibm_db.fetch_assoc(stmt)
            ibm_db.close(conn)
        elif req["db_type"] == "trino":
            import os
            import trino

            db_name = req["database"]
            if "." in db_name:
                catalog, schema = db_name.split(".", 1)
            elif "/" in db_name:
                catalog, schema = db_name.split("/", 1)
            else:
                catalog, schema = db_name, "default"

            http_scheme = "https" if os.environ.get("TRINO_USE_TLS", "0") == "1" else "http"
            auth = None
            if http_scheme == "https" and req.get("password"):
                auth = trino.BasicAuthentication(req.get("username") or "ragflow", req["password"])

            conn = trino.dbapi.connect(
                host=safe_host,
                port=int(req["port"] or 8080),
                user=req["username"] or "ragflow",
                catalog=catalog,
                schema=schema or "default",
                http_scheme=http_scheme,
                auth=auth,
            )
            try:
                cur = conn.cursor()
                try:
                    cur.execute("SELECT 1")
                    cur.fetchall()
                finally:
                    cur.close()
            finally:
                conn.close()
        else:
            return server_error_response("Unsupported database type.")

        return get_json_result(data="Database Connection Successful!")
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/agents/chat/completion", methods=["POST"])  # noqa: F821
@manager.route("/agents/chat/completions", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def agent_chat_completion(tenant_id, agent_id=None):
    # This endpoint serves two execution modes:
    # 1. Draft/runtime execution without session state. The request runs against the caller's
    #    runtime replica, which is populated from the editable canvas state.
    # 2. Session continuation with an existing session_id. The request resumes from the stored
    #    API4Conversation state and must stay bound to the same agent and an accessible canvas.
    #
    # Security constraints:
    # - agent_id is always supplied at the route layer and is not forwarded downstream as a free-form kwarg.
    # - New runs without session_id must pass UserCanvasService.accessible(...) before the runtime replica is loaded.
    # - Existing sessions are validated here at the route layer before handing control to the lower-level
    #   completion functions, so canvas_service only executes a pre-authorized session payload.
    #
    # Response modes:
    # - Regular mode emits internal agent events.
    # - openai-compatible mode reshapes the same execution into an OpenAI-like wire format.
    req = await get_request_json()
    agent_id = agent_id or req.get("agent_id")
    openai_compatible = bool(req.get("openai-compatible", False))
    if not agent_id:
        return get_json_result(
            data=False,
            message="`agent_id` is required.",
            code=RetCode.ARGUMENT_ERROR,
        )
    # Route-level selectors should not be forwarded into the lower-level completion functions.
    req = dict(req)
    req.pop("agent_id", None)
    req.pop("openai-compatible", None)
    session_id = req.get("session_id")
    workflow_session = False
    workflow_conv = None
    if session_id:
        exists, conv = API4ConversationService.get_by_id(session_id)
        if not exists:
            return get_data_error_result(message="Session not found!")
        if conv.dialog_id != agent_id:
            return get_json_result(
                data=False,
                message="Session does not belong to the requested agent.",
                code=RetCode.OPERATING_ERROR,
            )
        if not UserCanvasService.accessible(agent_id, tenant_id):
            return get_json_result(
                data=False,
                message="Only authorized users can access this agent session.",
                code=RetCode.OPERATING_ERROR,
            )
        workflow_session = getattr(conv, "source", "") == "workflow"
        if workflow_session:
            workflow_conv = conv.to_dict()

    if openai_compatible:
        # OpenAI-compatible mode uses a different wire format, keep it separate from regular agent events.
        messages = req.get("messages", [])
        if not messages:
            return get_data_error_result(message="You must provide at least one message.")
        question = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        stream = req.pop("stream", False)
        session_id = req.pop("session_id", req.get("id", "")) or req.get("metadata", {}).get("id", "")
        if stream:
            return _build_sse_response(
                completion_openai(
                    tenant_id,
                    agent_id,
                    question,
                    session_id=session_id,
                    stream=True,
                    **req,
                )
            )

        async for response in completion_openai(
            tenant_id,
            agent_id,
            question,
            session_id=session_id,
            stream=False,
            **req,
        ):
            return jsonify(response)
        return None

    if workflow_session:
        query = req.get("query", "") or req.get("question", "")
        files = req.get("files", [])
        inputs = req.get("inputs", {})
        runtime_user_id = req.get("user_id") or tenant_id
        user_id = str(runtime_user_id)
        custom_header = req.get("custom_header", "")

        _, cvs = await thread_pool_exec(UserCanvasService.get_by_id, agent_id)
        if not cvs:
            return get_data_error_result(message="canvas not found.")

        if not isinstance(workflow_conv.get("message"), list):
            workflow_conv["message"] = []
        if isinstance(workflow_conv.get("reference"), dict):
            if "chunks" in workflow_conv["reference"]:
                workflow_conv["reference"] = [workflow_conv["reference"]]
            else:
                workflow_conv["reference"] = [
                    value for _, value in sorted(workflow_conv["reference"].items(), key=lambda item: int(item[0]))
                ]
        elif not isinstance(workflow_conv.get("reference"), list):
            workflow_conv["reference"] = []
        workflow_conv["reference"] = [_normalize_agent_reference_entry(reference) for reference in workflow_conv["reference"]]
        turn_id = get_uuid()
        workflow_conv["message"].append(
            {
                "role": "user",
                "content": query,
                "id": turn_id,
                "files": files,
                "created_at": time.time(),
            }
        )
        await thread_pool_exec(API4ConversationService.update_by_id, session_id, workflow_conv)

        try:
            from agent.canvas import Canvas

            workflow_dsl = workflow_conv.get("dsl", {})
            if isinstance(workflow_dsl, str):
                dsl_str = workflow_dsl
            else:
                dsl_str = json.dumps(workflow_dsl, ensure_ascii=False)
            canvas = Canvas(dsl_str, str(tenant_id), canvas_id=agent_id, custom_header=custom_header)
        except Exception as exc:
            return server_error_response(exc)

        return await _run_workflow_session(
            tenant_id=tenant_id,
            agent_id=agent_id,
            workflow_conv=workflow_conv,
            canvas=canvas,
            query=query,
            files=files,
            inputs=inputs,
            user_id=user_id,
            session_id=session_id,
            custom_header=custom_header,
            canvas_title=getattr(cvs, "title", ""),
            canvas_category=getattr(cvs, "canvas_category", CanvasCategory.Agent),
            return_trace=bool(req.get("return_trace", False)),
            stream=req.get("stream", True),
            chat_template_kwargs=req.get("chat_template_kwargs"),
        )

    if not session_id:
        if not UserCanvasService.accessible(agent_id, tenant_id):
            return get_json_result(
                data=False,
                message="Make sure you have permission to access the agent.",
                code=RetCode.OPERATING_ERROR,
            )

        # Keep the original workflow execution path, but assign a session_id so the
        # response shape stays closer to the older agent completion contract.
        query = req.get("query", "") or req.get("question", "")
        files = req.get("files", [])
        inputs = req.get("inputs", {})
        runtime_user_id = req.get("user_id") or tenant_id
        user_id = str(runtime_user_id)
        custom_header = req.get("custom_header", "")
        session_id = get_uuid()

        _, cvs = await thread_pool_exec(UserCanvasService.get_by_id, agent_id)
        if not cvs:
            return get_data_error_result(message="canvas not found.")

        replica_payload = CanvasReplicaService.load_for_run(
            canvas_id=agent_id,
            tenant_id=str(tenant_id),
            runtime_user_id=user_id,
        )
        if not replica_payload:
            try:
                replica_payload = CanvasReplicaService.bootstrap(
                    canvas_id=agent_id,
                    tenant_id=str(tenant_id),
                    runtime_user_id=user_id,
                    dsl=cvs.dsl,
                    canvas_category=getattr(cvs, "canvas_category", CanvasCategory.Agent),
                    title=getattr(cvs, "title", ""),
                )
            except ValueError as exc:
                return get_data_error_result(message=str(exc))
        if not replica_payload:
            return get_data_error_result(message="canvas replica not found, please fetch the agent first.")

        replica_dsl = replica_payload.get("dsl", {})
        canvas_title = replica_payload.get("title", "")
        canvas_category = replica_payload.get("canvas_category", CanvasCategory.Agent)
        dsl_str = json.dumps(replica_dsl, ensure_ascii=False)

        if cvs.canvas_category == CanvasCategory.DataFlow:
            from rag.flow.pipeline import Pipeline

            task_id = get_uuid()
            workflow_conv = {
                "id": session_id,
                "dialog_id": cvs.id,
                "user_id": user_id,
                "exp_user_id": user_id,
                "name": req.get("name", ""),
                "message": [
                    {
                        "role": "user",
                        "content": query,
                        "id": task_id,
                        "files": files,
                        "created_at": time.time(),
                    }
                ],
                "reference": [],
                "source": "workflow",
                "dsl": replica_dsl,
                "version_title": await thread_pool_exec(
                    UserCanvasVersionService.get_latest_version_title,
                    cvs.id,
                    release_mode=False,
                ),
            }
            await thread_pool_exec(API4ConversationService.save, **workflow_conv)
            Pipeline(
                dsl_str,
                tenant_id=str(tenant_id),
                doc_id=CANVAS_DEBUG_DOC_ID,
                task_id=task_id,
                flow_id=agent_id,
            )
            ok, error_message = await thread_pool_exec(
                queue_dataflow,
                user_id,
                agent_id,
                task_id,
                CANVAS_DEBUG_DOC_ID,
                files[0],
                0,
            )
            if not ok:
                return get_data_error_result(message=error_message)
            return get_json_result(data={"message_id": task_id, "session_id": session_id})

        try:
            from agent.canvas import Canvas

            canvas = Canvas(dsl_str, str(tenant_id), canvas_id=agent_id, custom_header=custom_header)
            canvas.clear_history()
        except Exception as exc:
            return server_error_response(exc)
        turn_id = get_uuid()
        workflow_conv = {
            "id": session_id,
            "dialog_id": cvs.id,
            "user_id": user_id,
            "exp_user_id": user_id,
            "name": req.get("name") or (query[:250] if query else "") or "",
            "message": [
                {
                    "role": "user",
                    "content": query,
                    "id": turn_id,
                    "files": files,
                    "created_at": time.time(),
                }
            ],
            "reference": [],
            "source": "workflow",
            "dsl": replica_dsl,
            "version_title": await thread_pool_exec(
                UserCanvasVersionService.get_latest_version_title,
                cvs.id,
                release_mode=False,
            ),
        }
        workflow_conv["reference"] = [_normalize_agent_reference_entry(reference) for reference in workflow_conv["reference"]]
        await thread_pool_exec(API4ConversationService.save, **workflow_conv)
        return await _run_workflow_session(
            tenant_id=tenant_id,
            agent_id=agent_id,
            workflow_conv=workflow_conv,
            canvas=canvas,
            query=query,
            files=files,
            inputs=inputs,
            user_id=user_id,
            session_id=session_id,
            custom_header=custom_header,
            canvas_title=canvas_title,
            canvas_category=canvas_category,
            return_trace=bool(req.get("return_trace", False)),
            stream=req.get("stream", True),
            chat_template_kwargs=req.get("chat_template_kwargs"),
        )

    return_trace = bool(req.get("return_trace", False))
    if req.get("stream", True):

        async def generate():
            async for ans in _iter_session_completion_events(tenant_id, agent_id, req, return_trace):
                yield "data:" + json.dumps(ans, ensure_ascii=False) + "\n\n"
            yield "data:[DONE]\n\n"

        return _build_sse_response(generate())

    full_content = ""
    reference = {}
    final_ans = {}
    trace_items = []
    structured_output = {}
    async for ans in _iter_session_completion_events(tenant_id, agent_id, req, return_trace):
        try:
            if ans["event"] == "message":
                full_content += ans["data"]["content"]
            if ans.get("data", {}).get("reference", None):
                reference.update(ans["data"]["reference"])
            if ans.get("event") == "node_finished":
                data = ans.get("data", {})
                node_out = data.get("outputs", {})
                component_id = data.get("component_id")
                if component_id is not None and "structured" in node_out:
                    structured_output[component_id] = copy.deepcopy(node_out["structured"])
                if return_trace:
                    trace_items.append(
                        {
                            "component_id": data.get("component_id"),
                            "trace": [copy.deepcopy(data)],
                        }
                    )
            final_ans = ans
        except Exception as exc:
            return get_result(data=f"**ERROR**: {str(exc)}")

    if not final_ans:
        return get_result(data={})

    if "data" not in final_ans or not isinstance(final_ans["data"], dict):
        final_ans["data"] = {}
    final_ans["data"]["content"] = full_content
    final_ans["data"]["reference"] = reference
    if structured_output:
        final_ans["data"]["structured"] = structured_output
    if return_trace and final_ans:
        final_ans["data"]["trace"] = trace_items
    return get_result(data=final_ans)


@manager.route("/agents/<agent_id>/webhook", methods=["POST", "GET", "PUT", "PATCH", "DELETE", "HEAD"])  # noqa: F821
@manager.route("/agents/<agent_id>/webhook/test",methods=["POST", "GET", "PUT", "PATCH", "DELETE", "HEAD"],)  # noqa: F821
async def webhook(agent_id: str):
    is_test = request.path.startswith(f"/api/v1/agents/{agent_id}/webhook/test")
    start_ts = time.time()

    # 1. Fetch canvas by agent_id
    exists, cvs = UserCanvasService.get_by_id(agent_id)
    if not exists:
        return get_data_error_result(code=RetCode.BAD_REQUEST,message="Canvas not found."),RetCode.BAD_REQUEST

    # 2. Check canvas category
    if cvs.canvas_category == CanvasCategory.DataFlow:
        return get_data_error_result(code=RetCode.BAD_REQUEST,message="Dataflow can not be triggered by webhook."),RetCode.BAD_REQUEST

    # 3. Load DSL from canvas
    dsl = getattr(cvs, "dsl", None)
    if not isinstance(dsl, dict):
        return get_data_error_result(code=RetCode.BAD_REQUEST,message="Invalid DSL format."),RetCode.BAD_REQUEST

    # 4. Check webhook configuration in DSL
    webhook_cfg = {}
    components = dsl.get("components", {})
    for k, _ in components.items():
        cpn_obj = components[k]["obj"]
        if cpn_obj["component_name"].lower() == "begin" and cpn_obj["params"]["mode"] == "Webhook":
            webhook_cfg = cpn_obj["params"]

    if not webhook_cfg:
        return get_data_error_result(code=RetCode.BAD_REQUEST,message="Webhook not configured for this agent."),RetCode.BAD_REQUEST

    # 5. Validate request method against webhook_cfg.methods
    allowed_methods = webhook_cfg.get("methods", [])
    request_method = request.method.upper()
    if allowed_methods and request_method not in allowed_methods:
        return get_data_error_result(
            code=RetCode.BAD_REQUEST,message=f"HTTP method '{request_method}' not allowed for this webhook."
        ),RetCode.BAD_REQUEST

    # 6. Validate webhook security
    async def validate_webhook_security(security_cfg: dict):
        """Validate webhook security rules based on security configuration."""

        if not security_cfg:
            return  # No security config → allowed by default

        # 1. Validate max body size
        await _validate_max_body_size(security_cfg)

        # 2. Validate IP whitelist
        _validate_ip_whitelist(security_cfg)

        # # 3. Validate rate limiting
        _validate_rate_limit(security_cfg)

        # 4. Validate authentication
        auth_type = security_cfg.get("auth_type", "none")

        if auth_type == "none":
            return

        if auth_type == "token":
            _validate_token_auth(security_cfg)

        elif auth_type == "basic":
            _validate_basic_auth(security_cfg)

        elif auth_type == "jwt":
            _validate_jwt_auth(security_cfg)

        else:
            raise Exception(f"Unsupported auth_type: {auth_type}")

    async def _validate_max_body_size(security_cfg):
        """Check request size does not exceed max_body_size."""
        max_size = security_cfg.get("max_body_size")
        if not max_size:
            return

        # Convert "10MB" → bytes
        units = {"kb": 1024, "mb": 1024**2}
        size_str = max_size.lower()

        for suffix, factor in units.items():
            if size_str.endswith(suffix):
                limit = int(size_str.replace(suffix, "")) * factor
                break
        else:
            raise Exception("Invalid max_body_size format")
        MAX_LIMIT = 10 * 1024 * 1024  # 10MB
        if limit > MAX_LIMIT:
            raise Exception("max_body_size exceeds maximum allowed size (10MB)")

        content_length = request.content_length or 0
        if content_length > limit:
            raise Exception(f"Request body too large: {content_length} > {limit}")

    def _validate_ip_whitelist(security_cfg):
        """Allow only IPs listed in ip_whitelist."""
        whitelist = security_cfg.get("ip_whitelist", [])
        if not whitelist:
            return

        client_ip = request.remote_addr


        for rule in whitelist:
            if "/" in rule:
                # CIDR notation
                if ipaddress.ip_address(client_ip) in ipaddress.ip_network(rule, strict=False):
                    return
            else:
                # Single IP
                if client_ip == rule:
                    return

        raise Exception(f"IP {client_ip} is not allowed by whitelist")

    def _validate_rate_limit(security_cfg):
        """Simple in-memory rate limiting."""
        rl = security_cfg.get("rate_limit")
        if not rl:
            return

        limit = int(rl.get("limit", 60))
        if limit <= 0:
            raise Exception("rate_limit.limit must be > 0")
        per = rl.get("per", "minute")

        window = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }.get(per)

        if not window:
            raise Exception(f"Invalid rate_limit.per: {per}")

        capacity = limit
        rate = limit / window
        cost = 1

        key = f"rl:tb:{agent_id}"
        now = time.time()

        try:
            from rag.utils.redis_conn import REDIS_CONN

            res = REDIS_CONN.lua_token_bucket(
                keys=[key],
                args=[capacity, rate, now, cost],
                client=REDIS_CONN.REDIS,
            )

            allowed = int(res[0])
            if allowed != 1:
                raise Exception("Too many requests (rate limit exceeded)")

        except Exception as e:
            raise Exception(f"Rate limit error: {e}")

    def _validate_token_auth(security_cfg):
        """Validate header-based token authentication."""
        token_cfg = security_cfg.get("token",{})
        header = token_cfg.get("token_header")
        token_value = token_cfg.get("token_value")

        provided = request.headers.get(header)
        if provided != token_value:
            raise Exception("Invalid token authentication")

    def _validate_basic_auth(security_cfg):
        """Validate HTTP Basic Auth credentials."""
        auth_cfg = security_cfg.get("basic_auth", {})
        username = auth_cfg.get("username")
        password = auth_cfg.get("password")

        auth = request.authorization
        if not auth or auth.username != username or auth.password != password:
            raise Exception("Invalid Basic Auth credentials")

    def _validate_jwt_auth(security_cfg):
        """Validate JWT token in Authorization header."""
        jwt_cfg = security_cfg.get("jwt", {})
        secret = jwt_cfg.get("secret")
        if not secret:
            raise Exception("JWT secret not configured")

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise Exception("Missing Bearer token")

        token = auth_header[len("Bearer "):].strip()
        if not token:
            raise Exception("Empty Bearer token")

        alg = (jwt_cfg.get("algorithm") or "HS256").upper()

        decode_kwargs = {
            "key": secret,
            "algorithms": [alg],
        }
        options = {}
        if jwt_cfg.get("audience"):
            decode_kwargs["audience"] = jwt_cfg["audience"]
            options["verify_aud"] = True
        else:
            options["verify_aud"] = False

        if jwt_cfg.get("issuer"):
            decode_kwargs["issuer"] = jwt_cfg["issuer"]
            options["verify_iss"] = True
        else:
            options["verify_iss"] = False
        try:
            decoded = jwt.decode(
                token,
                options=options,
                **decode_kwargs,
            )
        except Exception as e:
            raise Exception(f"Invalid JWT: {str(e)}")

        raw_required_claims = jwt_cfg.get("required_claims", [])
        if isinstance(raw_required_claims, str):
            required_claims = [raw_required_claims]
        elif isinstance(raw_required_claims, (list, tuple, set)):
            required_claims = list(raw_required_claims)
        else:
            required_claims = []

        required_claims = [
            c for c in required_claims
            if isinstance(c, str) and c.strip()
        ]

        RESERVED_CLAIMS = {"exp", "sub", "aud", "iss", "nbf", "iat"}
        for claim in required_claims:
            if claim in RESERVED_CLAIMS:
                raise Exception(f"Reserved JWT claim cannot be required: {claim}")

        for claim in required_claims:
            if claim not in decoded:
                raise Exception(f"Missing JWT claim: {claim}")

        return decoded

    try:
        security_config=webhook_cfg.get("security", {})
        await validate_webhook_security(security_config)
    except Exception as e:
        return get_data_error_result(code=RetCode.BAD_REQUEST,message=str(e)),RetCode.BAD_REQUEST
    if not isinstance(cvs.dsl, str):
        dsl = json.dumps(cvs.dsl, ensure_ascii=False)
    try:
        from agent.canvas import Canvas

        canvas = Canvas(dsl, cvs.user_id, agent_id, canvas_id=agent_id)
    except Exception as e:
        resp=get_data_error_result(code=RetCode.BAD_REQUEST,message=str(e))
        resp.status_code = RetCode.BAD_REQUEST
        return resp

    # 7. Parse request body
    async def parse_webhook_request(content_type):
        """Parse request based on content-type and return structured data."""

        # 1. Query
        query_data = {k: v for k, v in request.args.items()}

        # 2. Headers
        header_data = {k: v for k, v in request.headers.items()}

        # 3. Body
        ctype = request.headers.get("Content-Type", "").split(";")[0].strip()
        if ctype and ctype != content_type:
            raise ValueError(
                f"Invalid Content-Type: expect '{content_type}', got '{ctype}'"
            )

        body_data: dict = {}

        try:
            if ctype == "application/json":
                body_data = await request.get_json() or {}

            elif ctype == "multipart/form-data":
                nonlocal canvas
                form = await request.form
                files = await request.files

                body_data = {}

                for key, value in form.items():
                    body_data[key] = value

                if len(files) > 10:
                    raise Exception("Too many uploaded files")
                for key, file in files.items():
                    desc = FileService.upload_info(
                        cvs.user_id,           # user
                        file,              # FileStorage
                        None                   # url (None for webhook)
                    )
                    file_parsed= await canvas.get_files_async([desc])
                    body_data[key] = file_parsed

            elif ctype == "application/x-www-form-urlencoded":
                form = await request.form
                body_data = dict(form)

            else:
                # text/plain / octet-stream / empty / unknown
                raw = await request.get_data()
                if raw:
                    try:
                        body_data = json.loads(raw.decode("utf-8"))
                    except Exception:
                        body_data = {}
                else:
                    body_data = {}

        except Exception:
            body_data = {}

        return {
            "query": query_data,
            "headers": header_data,
            "body": body_data,
            "content_type": ctype,
        }

    def extract_by_schema(data, schema, name="section"):
        """
        Extract only fields defined in schema.
        Required fields must exist.
        Optional fields default to type-based default values.
        Type validation included.
        """
        props = schema.get("properties", {})
        required = schema.get("required", [])

        extracted = {}

        for field, field_schema in props.items():
            field_type = field_schema.get("type")

            # 1. Required field missing
            if field in required and field not in data:
                raise Exception(f"{name} missing required field: {field}")

            # 2. Optional → default value
            if field not in data:
                extracted[field] = default_for_type(field_type)
                continue

            raw_value = data[field]

            # 3. Auto convert value
            try:
                value = auto_cast_value(raw_value, field_type)
            except Exception as e:
                raise Exception(f"{name}.{field} auto-cast failed: {str(e)}")

            # 4. Type validation
            if not validate_type(value, field_type):
                raise Exception(
                    f"{name}.{field} type mismatch: expected {field_type}, got {type(value).__name__}"
                )

            extracted[field] = value

        return extracted


    def default_for_type(t):
        """Return default value for the given schema type."""
        if t == "file":
            return []
        if t == "object":
            return {}
        if t == "boolean":
            return False
        if t == "number":
            return 0
        if t == "string":
            return ""
        if t and t.startswith("array"):
            return []
        if t == "null":
            return None
        return None

    def auto_cast_value(value, expected_type):
        """Convert string values into schema type when possible."""

        # Non-string values already good
        if not isinstance(value, str):
            return value

        v = value.strip()

        # Boolean
        if expected_type == "boolean":
            if v.lower() in ["true", "1"]:
                return True
            if v.lower() in ["false", "0"]:
                return False
            raise Exception(f"Cannot convert '{value}' to boolean")

        # Number
        if expected_type == "number":
            # integer
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)

            # float
            try:
                return float(v)
            except Exception:
                raise Exception(f"Cannot convert '{value}' to number")

        # Object
        if expected_type == "object":
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    raise Exception("JSON is not an object")
            except Exception:
                raise Exception(f"Cannot convert '{value}' to object")

        # Array <T>
        if expected_type.startswith("array"):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                else:
                    raise Exception("JSON is not an array")
            except Exception:
                raise Exception(f"Cannot convert '{value}' to array")

        # String (accept original)
        if expected_type == "string":
            return value

        # File
        if expected_type == "file":
            return value
        # Default: do nothing
        return value


    def validate_type(value, t):
        """Validate value type against schema type t."""
        if t == "file":
            return isinstance(value, list)

        if t == "string":
            return isinstance(value, str)

        if t == "number":
            return isinstance(value, (int, float))

        if t == "boolean":
            return isinstance(value, bool)

        if t == "object":
            return isinstance(value, dict)

        # array<string> / array<number> / array<object>
        if t.startswith("array"):
            if not isinstance(value, list):
                return False

            if "<" in t and ">" in t:
                inner = t[t.find("<") + 1 : t.find(">")]

                # Check each element type
                for item in value:
                    if not validate_type(item, inner):
                        return False

            return True

        return True
    parsed = await parse_webhook_request(webhook_cfg.get("content_types"))
    SCHEMA = webhook_cfg.get("schema", {"query": {}, "headers": {}, "body": {}})

    # Extract strictly by schema
    try:
        query_clean  = extract_by_schema(parsed["query"],   SCHEMA.get("query", {}),  name="query")
        header_clean = extract_by_schema(parsed["headers"], SCHEMA.get("headers", {}), name="headers")
        body_clean   = extract_by_schema(parsed["body"],    SCHEMA.get("body", {}),    name="body")
    except Exception as e:
        return get_data_error_result(code=RetCode.BAD_REQUEST,message=str(e)),RetCode.BAD_REQUEST

    clean_request = {
        "query": query_clean,
        "headers": header_clean,
        "body": body_clean,
        "input": parsed
    }

    execution_mode = webhook_cfg.get("execution_mode", "Immediately")
    response_cfg = webhook_cfg.get("response", {})

    def append_webhook_trace(agent_id: str, start_ts: float,event: dict, ttl=600):
        from rag.utils.redis_conn import REDIS_CONN

        key = f"webhook-trace-{agent_id}-logs"

        raw = REDIS_CONN.get(key)
        obj = json.loads(raw) if raw else {"webhooks": {}}

        ws = obj["webhooks"].setdefault(
            str(start_ts),
            {"start_ts": start_ts, "events": []}
        )

        ws["events"].append({
            "ts": time.time(),
            **event
        })

        REDIS_CONN.set_obj(key, obj, ttl)

    if execution_mode == "Immediately":
        status = response_cfg.get("status", 200)
        try:
            status = int(status)
        except (TypeError, ValueError):
            return get_data_error_result(code=RetCode.BAD_REQUEST,message=str(f"Invalid response status code: {status}")),RetCode.BAD_REQUEST

        if not (200 <= status <= 399):
            return get_data_error_result(code=RetCode.BAD_REQUEST,message=str(f"Invalid response status code: {status}, must be between 200 and 399")),RetCode.BAD_REQUEST

        body_tpl = response_cfg.get("body_template", "")

        def parse_body(body: str):
            if not body:
                return None, "application/json"

            try:
                parsed = json.loads(body)
                return parsed, "application/json"
            except (json.JSONDecodeError, TypeError):
                return body, "text/plain"


        body, content_type = parse_body(body_tpl)
        resp = Response(
            json.dumps(body, ensure_ascii=False) if content_type == "application/json" else body,
            status=status,
            content_type=content_type,
        )

        async def background_run():
            try:
                async for ans in canvas.run(
                    query="",
                    user_id=cvs.user_id,
                    webhook_payload=clean_request
                ):
                    if is_test:
                        append_webhook_trace(agent_id, start_ts, ans)

                if is_test:
                    append_webhook_trace(
                        agent_id,
                        start_ts,
                        {
                            "event": "finished",
                            "elapsed_time": time.time() - start_ts,
                            "success": True,
                        }
                    )

                cvs.dsl = json.loads(str(canvas))
                UserCanvasService.update_by_id(cvs.user_id, cvs.to_dict())

            except Exception as e:
                logging.exception("Webhook background run failed")
                if is_test:
                    try:
                        append_webhook_trace(
                            agent_id,
                            start_ts,
                            {
                                "event": "error",
                                "message": str(e),
                                "error_type": type(e).__name__,
                            }
                        )
                        append_webhook_trace(
                            agent_id,
                            start_ts,
                            {
                                "event": "finished",
                                "elapsed_time": time.time() - start_ts,
                                "success": False,
                            }
                        )
                    except Exception:
                        logging.exception("Failed to append webhook trace")

        task = asyncio.create_task(background_run())
        if isinstance(task, asyncio.Task):
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        return resp
    else:
        async def sse():
            nonlocal canvas
            contents: list[str] = []
            status = 200
            try:
                async for ans in canvas.run(
                    query="",
                    user_id=cvs.user_id,
                    webhook_payload=clean_request,
                ):
                    if ans["event"] == "message":
                        content = ans["data"]["content"]
                        if ans["data"].get("start_to_think", False):
                            content = "<think>"
                        elif ans["data"].get("end_to_think", False):
                            content = "</think>"
                        if content:
                            contents.append(content)
                    if ans["event"] == "message_end":
                        status = int(ans["data"].get("status", status))
                    if is_test:
                        append_webhook_trace(
                            agent_id,
                            start_ts,
                            ans
                        )
                if is_test:
                    append_webhook_trace(
                        agent_id,
                        start_ts,
                        {
                            "event": "finished",
                            "elapsed_time": time.time() - start_ts,
                            "success": True,
                        }
                    )
                final_content = "".join(contents)
                return {
                    "message": final_content,
                    "success": True,
                    "code":  status,
                }

            except Exception as e:
                if is_test:
                    append_webhook_trace(
                        agent_id,
                        start_ts,
                        {
                            "event": "error",
                            "message": str(e),
                            "error_type": type(e).__name__,
                        }
                    )
                    append_webhook_trace(
                        agent_id,
                        start_ts,
                        {
                            "event": "finished",
                            "elapsed_time": time.time() - start_ts,
                            "success": False,
                        }
                    )
                return {"code": 400, "message": str(e),"success":False}

        result = await sse()
        return Response(
            json.dumps(result),
            status=result["code"],
            mimetype="application/json",
        )


@manager.route("/agents/<agent_id>/webhook/logs", methods=["GET"])  # noqa: F821
@login_required
async def webhook_trace(agent_id: str):
    exists, cvs = UserCanvasService.get_by_id(agent_id)
    if not exists or str(cvs.user_id) != str(current_user.id):
        return get_data_error_result(
            message="Canvas not found.",
        )

    def encode_webhook_id(start_ts: str) -> str:
        WEBHOOK_ID_SECRET = "webhook_id_secret"
        sig = hmac.new(
            WEBHOOK_ID_SECRET.encode("utf-8"),
            start_ts.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")

    def decode_webhook_id(enc_id: str, webhooks: dict) -> str | None:
        for ts in webhooks.keys():
            if encode_webhook_id(ts) == enc_id:
                return ts
        return None
    since_ts = request.args.get("since_ts", type=float)
    webhook_id = request.args.get("webhook_id")

    key = f"webhook-trace-{agent_id}-logs"
    from rag.utils.redis_conn import REDIS_CONN

    raw = REDIS_CONN.get(key)

    if since_ts is None:
        now = time.time()
        return get_json_result(
            data={
                "webhook_id": None,
                "events": [],
                "next_since_ts": now,
                "finished": False,
            }
        )

    if not raw:
        return get_json_result(
            data={
                "webhook_id": None,
                "events": [],
                "next_since_ts": since_ts,
                "finished": False,
            }
        )

    obj = json.loads(raw)
    webhooks = obj.get("webhooks", {})

    if webhook_id is None:
        candidates = [
            float(k) for k in webhooks.keys() if float(k) > since_ts
        ]

        if not candidates:
            return get_json_result(
                data={
                    "webhook_id": None,
                    "events": [],
                    "next_since_ts": since_ts,
                    "finished": False,
                }
            )

        start_ts = min(candidates)
        real_id = str(start_ts)
        webhook_id = encode_webhook_id(real_id)

        return get_json_result(
            data={
                "webhook_id": webhook_id,
                "events": [],
                "next_since_ts": start_ts,
                "finished": False,
            }
        )

    real_id = decode_webhook_id(webhook_id, webhooks)

    if not real_id:
        return get_json_result(
            data={
                "webhook_id": webhook_id,
                "events": [],
                "next_since_ts": since_ts,
                "finished": True,
            }
        )

    ws = webhooks.get(str(real_id))
    events = ws.get("events", [])
    new_events = [e for e in events if e.get("ts", 0) > since_ts]

    next_ts = since_ts
    for e in new_events:
        next_ts = max(next_ts, e["ts"])

    finished = any(e.get("event") == "finished" for e in new_events)

    return get_json_result(
        data={
            "webhook_id": webhook_id,
            "events": new_events,
            "next_since_ts": next_ts,
            "finished": finished,
        }
    )

@manager.route("/agents/attachments/<attachment_id>/download", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def download_attachment(tenant_id=None, attachment_id=None):
    """Stream a document's underlying file to the requesting user.

    Mirrors the authorization model of the preview endpoint: the user must belong
    to the tenant that owns the document's knowledge base. A denial returns the
    same "Document not found!" response so the endpoint cannot be used to
    enumerate doc ids across tenants.
    """
    try:
        # Keep backward compatibility with older callers and unit tests that still
        # pass `attachment_id` instead of the route parameter name.
        ext = request.args.get("ext", "markdown")
        data = await thread_pool_exec(settings.STORAGE_IMPL.get, tenant_id, attachment_id)
        response = await make_response(data)
        content_type = CONTENT_TYPE_MAP.get(ext, f"application/{ext}")
        apply_safe_file_response_headers(response, content_type, ext)

        return response

    except Exception as e:
        return server_error_response(e)
