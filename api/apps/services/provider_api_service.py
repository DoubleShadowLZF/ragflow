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
Provider API 服务层 —— 供应商/实例/模型的业务逻辑。

本模块是 provider_api 蓝图的业务逻辑层，负责处理供应商（Provider）、
模型实例（Instance）和具体模型（Model）的 CRUD 操作以及连接验证。

核心数据结构（三层模型）：
    Provider（供应商）── 如 OpenAI、DeepSeek、SiliconFlow 等
        └── Instance（实例）── Provider + API Key + Base URL 的具体配置
                └── Model（模型）── 如 gpt-4o、text-embedding-3-small 等

关键概念：
- ``FACTORY_LLM_INFOS`` —— 全局供应商注册表（来自 service_conf.yaml），
  定义了系统支持的所有供应商及其默认配置。
- ``TenantModelProviderService`` —— 租户-供应商关联（租户启用了哪些供应商）
- ``TenantModelInstanceService`` —— 租户的供应商实例（API Key 配置）
- ``TenantModelService`` —— 实例内具体的模型配置及启用/禁用状态
"""

import os
import json
import logging
import asyncio

from common.constants import LLMType, ActiveStatusEnum
from common.misc_utils import get_uuid
from common.settings import FACTORY_LLM_INFOS
from api.db.joint_services.tenant_model_service import get_model_config_from_provider_instance, delete_models_by_instance_ids, delete_instances_by_provider_ids
from api.db.services.tenant_model_provider_service import TenantModelProviderService
from api.db.services.tenant_model_instance_service import TenantModelInstanceService
from api.db.services.tenant_model_service import TenantModelService
from rag.llm import ChatModel, EmbeddingModel, ModelMeta, OcrModel, RerankModel, TTSModel


# =============================================================================
# 内部辅助函数
# =============================================================================

def _to_int(v, default=500):
    """安全转换为 int，转换失败返回默认值。用于解析 rank 排序权重。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _factory_model_types(llm: dict) -> list[str]:
    """从 LLM 工厂条目中提取模型类型列表。

    兼容两种配置格式：
    - ``model_type`` 为列表 → 直接返回
    - ``model_type`` 为字符串 → 包装为单元素列表
    - 未定义 → 返回空列表
    """
    model_type = llm.get("model_type")
    if isinstance(model_type, list):
        return model_type
    return [model_type] if model_type else []


def _normalize_provider_base_url(provider_name: str, base_url: str | None):
    """规范化供应商的 Base URL。

    特殊处理 VLLM 供应商：自动在 URL 末尾追加 "/v1" 后缀
    （VLLM 的 OpenAI 兼容端点始终位于 /v1 路径下）。
    """
    if provider_name != "VLLM" or not base_url:
        return base_url
    base_url = base_url.strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def _factory_llm_name(llm: dict) -> str:
    """从 LLM 工厂条目中提取模型名称。

    兼容两种键名：优先取 ``name``，回退取 ``llm_name``。
    """
    return llm.get("name") or llm.get("llm_name", "")


# =============================================================================
# Provider（供应商）业务逻辑
# =============================================================================

def list_providers(tenant_id: str, all_available: bool = False):
    """列出供应商列表。

    两种查询模式：
    - ``all_available=True`` —— 列出系统所有可用供应商（排除 Youdao、FastEmbed、
      BAAI、Builtin、siliconflow_intl 等内部/非显示型供应商），按 rank 降序排列。
    - ``all_available=False`` —— 列出当前租户已配置的供应商，同样按 rank 排列。

    特殊处理：
    - MinerU / PaddleOCR / OpenDataLoader 会额外追加 "ocr" 模型类型
    - SiliconFlow 和 Tongyi-Qianwen 会添加国际站 URL（``intl`` key）

    :param tenant_id: 租户 ID
    :param all_available: 是否列出所有可用供应商（而非租户已配置的）
    :return: (success, providers_list)
    """
    if not FACTORY_LLM_INFOS:
        return False, []

    # 构建 rank 排序映射 —— rank 越高越靠前，取负数用于 sort()
    factory_rank_mapping = {factory["name"]: -_to_int(factory.get("rank", "500")) for factory in FACTORY_LLM_INFOS}
    factory_info_map = {f["name"]: f for f in FACTORY_LLM_INFOS}
    if all_available:
        providers = []
        for factory_info in FACTORY_LLM_INFOS:
            # 跳过内部/非显示型供应商
            if factory_info["name"] in ["Youdao", "FastEmbed", "BAAI", "Builtin", "siliconflow_intl"]:
                continue
            # 提取该供应商支持的所有模型类型（去重、排序）
            model_types = sorted(set(
                model_type
                for llm in factory_info.get("llm", [])
                for model_type in _factory_model_types(llm)
            )) if factory_info.get("llm", []) else []
            if factory_info["name"] in ["MinerU", "PaddleOCR", "OpenDataLoader"]:
                model_types.append("ocr")
            provider = {
                "model_types": model_types,
                "name": factory_info["name"],
                "url": {
                    "default": factory_info.get("url", "")
                }
            }
            # 国际站 URL 处理
            if factory_info["name"].lower() == "siliconflow":
                provider["url"]["intl"] = factory_info_map.get("siliconflow_intl", {}).get("url", "https://api.siliconflow.com/v1")
            elif factory_info["name"] == "Tongyi-Qianwen":
                provider["url"]["intl"] = "https://dashscope-intl.aliyuncs.com/compatible-model/v1"
            providers.append(provider)
        providers.sort(key=lambda x: (factory_rank_mapping.get(x["name"]), x["name"]))
        return True, providers

    # 列出租户已配置的供应商
    factory_names = TenantModelProviderService.list_provider_names_by_tenant_id(tenant_id)

    providers = []
    factory_info_mapping = {f["name"]: f for f in FACTORY_LLM_INFOS}
    for name in factory_names:
        if name not in ["Youdao", "FastEmbed", "BAAI", "Builtin", "siliconflow_intl"] and factory_info_mapping.get(name):
            factory_info = factory_info_mapping[name]
            model_types = sorted(set(
                model_type
                for llm in factory_info.get("llm", [])
                for model_type in _factory_model_types(llm)
            )) if factory_info.get("llm", []) else []
            if name in ["MinerU", "PaddleOCR", "OpenDataLoader"]:
                model_types.append("ocr")

            provider = {
                "model_types": model_types,
                "name": factory_info["name"],
                "url": {
                    "default": factory_info.get("url", "")
                }
            }
            if factory_info["name"].lower() == "siliconflow":
                provider["url"]["intl"] = factory_info_map.get("siliconflow_intl", {}).get("url", "https://api.siliconflow.com/v1")
            elif factory_info["name"] == "Tongyi-Qianwen":
                provider["url"]["intl"] = "https://dashscope-intl.aliyuncs.com/compatible-model/v1"
            providers.append(provider)
    providers.sort(key=lambda x: (factory_rank_mapping.get(x["name"]), x["name"]))
    return True, providers


def add_provider(tenant_id: str, provider_name: str):
    """为租户添加供应商。

    校验：
    - 供应商名称必须在 FACTORY_LLM_INFOS 的允许列表中
    - 同一租户不可重复添加同一供应商
    """
    if not FACTORY_LLM_INFOS:
        return False, "No providers found"
    # 检查供应商是否在允许列表中
    allowed_factories = [f["name"] for f in FACTORY_LLM_INFOS]
    if provider_name not in allowed_factories:
        return False, f"Provider '{provider_name}' is not allowed"

    existing = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if existing:
        return False, f"Provider {provider_name} already exists"

    TenantModelProviderService.insert(
        tenant_id=tenant_id,
        provider_name=provider_name
    )
    return True, "success"


def delete_provider(tenant_id: str, provider_name: str):
    """删除租户下的供应商及其所有实例和模型。

    级联删除顺序：
    1. 查询该供应商下的所有实例
    2. 删除实例下的所有模型记录
    3. 删除所有实例
    4. 删除租户-供应商关联
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"Provider {provider_name} not found"
    instance_objs = TenantModelInstanceService.get_all_by_provider_id(provider_obj.id)
    if not instance_objs:
        return False, f"No instances found for provider {provider_name}"
    instance_ids = [instance_obj.id for instance_obj in instance_objs]
    delete_models_by_instance_ids(instance_ids)
    delete_instances_by_provider_ids([provider_obj.id])
    TenantModelProviderService.delete_by_tenant_id_and_provider_name(tenant_id, provider_name)
    return True, "success"


def show_provider(provider_name: str):
    """查看供应商详情 —— 从 FACTORY_LLM_INFOS 中获取基���信息。"""
    fac_list = [f for f in FACTORY_LLM_INFOS if f["name"]==provider_name]
    if not fac_list:
        return False, f"Provider '{provider_name}' not found"
    factory_info = fac_list[0]
    return True, {
        "base_url": {
            "default": factory_info.get("url", "")
        },
        "name": factory_info["name"],
        "total_models": len(factory_info.get("llm", []))
    }


async def list_provider_models(provider_name: str, api_key: str = None, base_url: str = None):
    """列出供应商的所有可用模型。

    合并两个来源的模型列表：
    1. 静态模型列表 —— 来自 FACTORY_LLM_INFOS 配置（本地注册的模型）
    2. 远程模型列表 —— 通过 ModelMeta 从供应商 API 实时拉取（如 OpenAI /v1/models）

    合并策略：远程模型覆盖同名的静态模型（remote_models 优先）。

    可选传入 api_key 和 base_url 用于从供应商 API 拉取远程模型列表。
    """
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"]==provider_name]
    if not factory_info:
        return False, f"Provider '{provider_name}' not found"
    # 构建静态模型列表
    static_llms = [{
            "name": _factory_llm_name(llm),
            "max_tokens": llm["max_tokens"],
            "model_types": _factory_model_types(llm),
            "features": (
                llm.get("features")
                if llm.get("features") is not None
                else (
                    (["is_tools"] if llm.get("is_tools") else [])
                    + (["thinking"] if llm.get("thinking") else [])
                )
            )
        } for llm in factory_info[0]["llm"]]

    model_base_url = _normalize_provider_base_url(provider_name, base_url) or factory_info[0].get("url", "")
    remote_models = []
    if provider_name in ModelMeta:
        remote_models = await ModelMeta[provider_name](api_key, model_base_url).get_model_list()

    if not static_llms and not remote_models:
        return False, f"No models found for provider '{provider_name}'"

    # 合并静态和远程模型，远程模型覆盖同名静态模型
    merged = {m["name"]: m for m in static_llms}
    merged.update({m["name"]: m for m in remote_models})
    models = list(merged.values())

    models.sort(key=lambda x: x["name"])
    return True, models


def show_provider_model(provider_name: str, model_name: str):
    """查看供应商下某个具体模型的详细信息。"""
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
    if not factory_info:
        return False, f"Provider '{provider_name}' not found"
    llms = factory_info[0]["llm"]
    if not llms:
        return False, f"No models found for provider '{provider_name}'"
    target_llm = [llm for llm in llms if _factory_llm_name(llm) == model_name]
    if not target_llm:
        return False, f"Model '{model_name}' not found"
    llm_info = target_llm[0]

    return True, {
        "name": _factory_llm_name(llm_info),
        "max_tokens": llm_info["max_tokens"],
        "model_types": _factory_model_types(llm_info),
        "thinking": None,
        "model_type_map": {model_type: True for model_type in _factory_model_types(llm_info)}
    }


# =============================================================================
# Instance（供应商实例）业务逻辑
# =============================================================================

async def create_provider_instance(tenant_id: str, provider_name: str, instance_name: str, api_key: str|dict, base_url: str, region: str, model_info: list[dict]=None):
    """创建供应商实例。

    实例 = 供应商 + API Key 的绑定记录。创建流程：
    1. 参数校验（instance_name 不能为 "default"、"provider_name" 非空）
    2. 检查供应商在系统中存在且租户已添加
    3. 检查同一 API Key 未被其他实例使用
    4. 调用 verify_api_key 验证连接可用性
    5. 创建实例记录（存储 api_key、base_url、region 等）
    6. 如传入了 model_info，批量添加模型配置

    :param model_info: 预配置的模型列表，每项格式为
        ``{"model_type": ["chat"], "model_name": "gpt-4o", "max_tokens": 4096, "extra": {}}``
    """
    if not provider_name:
        return False, "Provider name is required"

    base_url = _normalize_provider_base_url(provider_name, base_url)

    if instance_name == "default":
        return False, "Instance name cannot be 'default'"

    # 检查供应商在系统中存在
    allowed_factories = [f["name"] for f in FACTORY_LLM_INFOS]
    if provider_name not in allowed_factories:
        return False, f"Provider '{provider_name}' is not allowed"

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"Provider '{provider_name}' does not exist"

    # 检查同一 API Key 未被其他实例使用
    api_key_str = ""
    if api_key:
        api_key_str = api_key if isinstance(api_key, str) else json.dumps(api_key)
        same_key_instance = TenantModelInstanceService.get_by_provider_id_and_api_key(provider_obj.id, api_key_str)
        if same_key_instance:
            return False, f"Already exist instance: {same_key_instance.instance_name} with api_key {api_key}"
    # 先验证 API Key 可用性再创建
    success, msg = await verify_api_key(provider_name, api_key, base_url, region, model_info)
    if not success:
        return False, msg

    extra_fields = {}
    if base_url:
        extra_fields["base_url"] = base_url
    if region:
        extra_fields["region"] = region
    TenantModelInstanceService.create_instance(provider_id=provider_obj.id,instance_name=instance_name,api_key=api_key_str, extra=json.dumps(extra_fields))
    if model_info:
        msg = ""
        for model in model_info:
            success, _msg = add_model_to_instance(tenant_id, provider_name, instance_name, **model)
            if not success:
                msg += _msg
        if msg:
            return False, msg

    return True, "success"


def list_provider_instances(tenant_id: str, provider_name: str):
    """列出指定供应商下的所有实例。"""
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_name}'"
    provider_id = provider_obj.id
    instance_objs = TenantModelInstanceService.get_all_by_provider_id(provider_id)
    if not instance_objs:
        return True, []
    instances = []
    for instance_obj in instance_objs:
        extra_fields = json.loads(instance_obj.extra) if instance_obj.extra else {}
        instances.append({
            "id": instance_obj.id,
            "instance_name": instance_obj.instance_name,
            "provider_id": provider_id,
            "region": extra_fields.get("region", ""),
            "status": instance_obj.status,
        })

    return True, instances


async def verify_api_key(provider_name: str, api_key: str|dict, base_url: str=None, region: str=None, model_info: list[dict]=None):
    """验证供应商 API Key 是否有效。

    这是整个 Provider 系统中最核心的验证函数。对供应商工厂中定义的每种
    模型类型，依次用提供的 API Key 尝试调用对应模型的能力接口：

    - **Embedding 模型** → 调用 ``encode()``，检查返回向量非空
    - **Chat 模型** → 流式调用 ``async_chat_streamly()``，检查有非错误响应
    - **Rerank 模型** → 调用 ``similarity()``，检查返回分数有效
    - **OCR 模型** → 调用 ``check_available()``，检查可用性
    - **TTS 模型** → 调用 ``tts()``，检查无异常抛出

    只要任一种模型类型验证通过，整个 API Key 被视为有效（``any()`` 判断）。

    特殊处理：
    - SiliconFlow 的 ``region="intl"`` 映射到 ``siliconflow_intl`` 工厂
    - BaiduYiyan 的 API Key 如为纯字符串则转换为 ``{"yiyan_ak": ..., "yiyan_sk": ""}`` 格式
    - 超时时间由环境变量 ``LLM_TIMEOUT_SECONDS`` 控制，默认 10 秒
    """
    if not provider_name:
        return False, "Provider name is required"

    base_url = _normalize_provider_base_url(provider_name, base_url)

    # SiliconFlow 国际站路由
    if region and region == "intl" and provider_name.lower() == "siliconflow":
        target_factory_name = "siliconflow_intl"
    else:
        target_factory_name = provider_name

    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == target_factory_name]
    if not factory_info:
        return False, f"Provider '{provider_name}' not found"

    factory_llms = factory_info[0]["llm"]
    if not factory_llms:
        if not model_info:
            return False, f"No models found for provider '{provider_name}'"
        # 从用户传入的 model_info 构造临时模型列表用于验证
        factory_llms = [{
            "model_type": _type,
            "llm_name": model.get("model_name", ""),
        } for model in model_info if model for _type in model.get("model_type", []) ]
        if not factory_llms:
            return False, f"No valid models found for provider '{provider_name}'"

    # 依次尝试每种模型类型的 API 调用
    chat_passed, embd_passed, rerank_passed, ocr_passed, tts_passed = False, False, False, False, False
    timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", 10))
    extra = {"provider": provider_name}
    msg = ""
    # BaiduYiyan 的 API Key 格式转换
    if provider_name == "BaiduYiyan":
        if isinstance(api_key, str):
            try:
                json.loads(api_key)
            except (json.JSONDecodeError, TypeError):
                api_key = {"yiyan_ak": api_key, "yiyan_sk": ""}
    api_key_str = api_key if isinstance(api_key, str) else json.dumps(api_key)
    for llm in factory_llms:
        model_types = _factory_model_types(llm)
        # ---- Embedding 模型验证 ----
        if not embd_passed and LLMType.EMBEDDING.value in model_types:
            assert provider_name in EmbeddingModel, f"Embedding model from {provider_name} is not supported yet."
            mdl = EmbeddingModel[provider_name](api_key_str, llm["llm_name"], base_url=base_url)
            try:
                arr, tc = await asyncio.wait_for(
                    asyncio.to_thread(mdl.encode, ["Test if the api key is available"]),
                    timeout=timeout_seconds,
                )
                if len(arr[0]) == 0:
                    raise Exception("Fail")
                embd_passed = True
            except Exception as e:
                logging.exception(
                    "Fail to access embedding model for provider=%s model=%s",
                    provider_name,
                    llm["llm_name"],
                )
                msg += f"\nFail to access embedding model({llm['llm_name']}) using this api key." + str(e)
        # ---- Chat 模型验证 ----
        elif not chat_passed and LLMType.CHAT.value in model_types:
            assert provider_name in ChatModel, f"Chat model from {provider_name} is not supported yet."
            mdl = ChatModel[provider_name](api_key_str, llm["llm_name"], base_url=base_url, **extra)
            try:
                async def check_streamly():
                    async for chunk in mdl.async_chat_streamly(
                            None,
                            [{"role": "user", "content": "Hi"}],
                            {"temperature": 0.9},
                    ):
                        if chunk and isinstance(chunk, str) and chunk.find("**ERROR**") < 0:
                            return True
                    return False

                result = await asyncio.wait_for(check_streamly(), timeout=timeout_seconds)
                if result:
                    chat_passed = True
                else:
                    raise Exception("No valid response received")
            except Exception as e:
                logging.exception(
                    "Fail to access chat model for provider=%s model=%s",
                    provider_name,
                    llm["llm_name"],
                )
                msg += f"\nFail to access model({provider_name}/{llm['llm_name']}) using this api key." + str(e)
        # ---- Rerank 模型验证 ----
        elif not rerank_passed and LLMType.RERANK.value in model_types:
            if provider_name not in RerankModel:
                unsupported_msg = f"Rerank model from {provider_name} is not supported yet."
                logging.warning(unsupported_msg)
                msg += f"\n{unsupported_msg}"
                continue
            mdl = RerankModel[provider_name](api_key_str, llm["llm_name"], base_url=base_url)
            try:
                arr, tc = await asyncio.wait_for(
                    asyncio.to_thread(mdl.similarity, "What's the weather?", ["Is it sunny today?"]),
                    timeout=timeout_seconds,
                )
                if len(arr) == 0 or tc == 0:
                    raise Exception("Fail")
                rerank_passed = True
                logging.debug(f"passed model rerank {llm['llm_name']}")
            except Exception as e:
                logging.exception(
                    "Fail to access rerank model for provider=%s model=%s",
                    provider_name,
                    llm["llm_name"],
                )
                msg += f"\nFail to access model({provider_name}/{llm['llm_name']}) using this api key." + str(e)
        # ---- OCR 模型验证 ----
        elif not ocr_passed and LLMType.OCR.value in model_types:
            assert provider_name in OcrModel, f"OCR model from {provider_name} is not supported yet."
            mdl = OcrModel[provider_name](key=api_key_str, model_name=llm["llm_name"], base_url=base_url)
            try:
                ok, reason = await asyncio.wait_for(
                    asyncio.to_thread(mdl.check_available),
                    timeout=timeout_seconds,
                )
                if not ok:
                    raise RuntimeError(reason or "Model not available")
                ocr_passed = True
            except Exception as e:
                logging.exception(
                    "Fail to access OCR model for provider=%s model=%s",
                    provider_name,
                    llm["llm_name"],
                )
                msg += f"\nFail to access model({provider_name}/{llm['llm_name']})." + str(e)
        # ---- TTS 模型验证 ----
        elif not tts_passed and LLMType.TTS.value in model_types:
            assert provider_name in TTSModel, f"TTS model from {provider_name} is not supported yet."
            mdl = TTSModel[provider_name](key=api_key_str, model_name=llm["llm_name"], base_url=base_url)
            try:
                def drain_tts():
                    for _ in mdl.tts("Hello~ RAGFlower!"):
                        pass

                await asyncio.wait_for(
                    asyncio.to_thread(drain_tts),
                    timeout=timeout_seconds,
                )
                tts_passed = True
            except Exception as e:
                logging.exception(
                    "Fail to access TTS model for provider=%s model=%s",
                    provider_name,
                    llm["llm_name"],
                )
                msg += f"\nFail to access model({provider_name}/{llm['llm_name']})." + str(e)
        # 只要任一类型通过验证，清除错误消息并提前退出循环
        if any([embd_passed, chat_passed, rerank_passed, ocr_passed, tts_passed]):
            msg = ""
            break

    success = any([embd_passed, chat_passed, rerank_passed, ocr_passed, tts_passed])
    return success, "success" if success else msg


def show_provider_instance(tenant_id: str, provider_name: str, instance_name: str):
    """查看指定实例的详细信息（API Key 脱敏后返回）。"""
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_name}'"
    provider_id = provider_obj.id
    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_id, instance_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_name}' and instance '{instance_name}'"

    extra_fields = json.loads(instance_obj.extra) if instance_obj.extra else {}
    return True, {
        "id": instance_obj.id,
        "instance_name": instance_obj.instance_name,
        "provider_id": provider_id,
        "region": extra_fields.get("region", ""),
        "status": instance_obj.status
    }


def drop_provider_instances(tenant_id: str, provider_name: str, instance_names: list):
    """批量删除供应商实例及其下所有模型。

    先校验所有传入的 instance_name 都存在，再执行级联删除：
    模型 → 实例。任何一个实例不存在则返回错误，不执行任何删除。
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_name}'"
    provider_id = provider_obj.id
    not_exist_instances = []
    instance_ids = []
    for instance_name in instance_names:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_id, instance_name)
        if not instance_obj:
            not_exist_instances.append(instance_name)
            continue
        instance_ids.append(instance_obj.id)
    if not_exist_instances:
        return False, f"No instance found for provider '{provider_name}' and instance '{not_exist_instances}'"
    delete_models_by_instance_ids(instance_ids)
    TenantModelInstanceService.delete_by_ids(instance_ids)
    return True, None


# =============================================================================
# Instance Model（实例内模型配置）业务逻辑
# =============================================================================

def list_instance_models(tenant_id: str, provider_name: str, instance_name: str, supported_only: bool = False):
    """列出实例中的模型配置（包含启用/禁用状态）。

    模型的启用/禁用遵循"白名单 + 黑名单"混合逻辑：
    - 默认所有工厂中的模型都是 active（启用）
    - 如果 tenant_model 表中存在某模型记录，则按其 status 字段显示状态
    - 工厂模型列表中不存在的自定义模型也会被列出（从 tenant_model 中读取）

    注：这与旧版 Go 代码的逻辑保持一致。

    :param supported_only: True 时仅返回工厂中定义的模型名称列表（不含状态）。
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_name}'"

    if supported_only:
        # 仅返回供应商支持的模型名列表
        factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
        if not factory_info:
            return False, f"Provider '{provider_name}' not found"
        llms = factory_info[0].get("llm", [])
        models = [{"name": llm["llm_name"]} for llm in llms]
        models.sort(key=lambda x: x["name"])
        return True, models

    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_name}' and instance '{instance_name}'"

    # 从 tenant_model 表读取该实例的模型记录
    model_records = TenantModelService.get_models_by_instance_id(instance_obj.id)
    # 构建 model_name → {status, model_type, extra} 的映射
    model_info_map: dict = {}
    for model_record in model_records:
        if model_info_map.get(model_record.model_name):
            model_info_map[model_record.model_name]["model_type"].append(model_record.model_type)
        else:
            model_info_map[model_record.model_name] = {
                "status": model_record.status,
                "model_type": [model_record.model_type],
                "extra": model_record.extra
            }

    # 列出工厂中定义的所有模型，合并数据库中的状态
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
    if not factory_info:
        return False, f"Provider '{provider_name}' not found"

    llms = factory_info[0].get("llm", [])
    models = []
    for llm in llms:
        models.append({
            "name": llm["llm_name"],
            "model_type": list(
                dict.fromkeys(_factory_model_types(llm) + model_info_map.get(llm["llm_name"], {}).get("model_type", []))
            ),
            "max_tokens": llm.get("max_tokens"),
            "status": model_info_map.get(llm["llm_name"], {}).get("status", "active"),
        })
    # 添加工厂列表中不存在的自定义模型（来自 tenant_model 但不在工厂定义中）
    factory_models = [m["name"] for m in models]
    for model_name, model_info_dict in model_info_map.items():
        if model_name not in factory_models:
            extra_fields = json.loads(model_info_dict["extra"]) if model_info_dict["extra"] else {}
            models.append({
                "name": model_name,
                "model_type": model_info_dict["model_type"],
                "max_tokens": extra_fields.get("max_tokens", 8192),
                "status": model_info_dict["status"],
            })
    return True, models


def add_model_to_instance(tenant_id: str, provider_name: str, instance_name: str, model_name: str, model_type: str|list[str], max_tokens: int=8192, extra: dict=None):
    """向实例添加模型配置。

    支持 ``model_type`` 为字符串或列表——为每种模型类型创建一条独立的
    tenant_model 记录。每类模型可在工厂定义中查找 ``is_tools`` 等额外属性，
    并与用户传入的 ``extra`` 合并存储。
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_name}'"
    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_name}' and instance '{instance_name}'"
    # 检查同名模型是否已存在
    model_obj = TenantModelService.get_by_provider_id_and_instance_id_and_model_name(provider_obj.id, instance_obj.id, model_name)
    if model_obj:
        return False, f"Model '{model_name}' already exists for provider '{provider_name}' and instance '{instance_name}'"
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
    if not factory_info:
        return False, f"Provider '{provider_name}' not found"
    llms = factory_info[0].get("llm", [])
    if isinstance(model_type, str):
        model_type = [model_type]

    for _type in model_type:
        extra_fields = {"max_tokens": max_tokens}
        # 从工厂定义中查找模型的额外属性（如 is_tools）
        target_model = [llm for llm in llms if _type in _factory_model_types(llm) and llm["llm_name"] == model_name]
        if target_model:
            extra_fields.update({"is_tools": target_model[0].get("is_tools", False)})
        if extra:
            extra_fields.update(extra)
        TenantModelService.insert(
            model_name=model_name,
            provider_id=provider_obj.id,
            instance_id=instance_obj.id,
            model_type=_type,
            extra=json.dumps(extra_fields)
        )

    return True, "success"


def update_model_status(tenant_id: str, provider_name: str, instance_name: str, model_name: str, status: str):
    """启用或禁用实例中的模型。

    逻辑（与旧版 Go 代码一致）：
    - tenant_model 表中已存在记录 → 直接更新 status
    - 不存在记录：
      - 设为 ``active`` → 无需操作（默认就是启用）
      - 设为 ``inactive`` → 从工厂定义中查找模型配置，创建一条 status=inactive 的记录

    每个模型可能有多种类型（如同时是 chat + embedding），会为每种类型创建/更新记录。
    """
    if status not in (ActiveStatusEnum.ACTIVE.value, ActiveStatusEnum.INACTIVE.value):
        return False, f"status must be '{ActiveStatusEnum.ACTIVE.value}' or '{ActiveStatusEnum.INACTIVE.value}'"

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_name}'"

    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_name}' and instance '{instance_name}'"

    # 查找 tenant_model 表中的现有记录
    model_obj_list = TenantModelService.get_by_provider_id_and_instance_id_and_model_name(
        provider_obj.id, instance_obj.id, model_name
    )

    if model_obj_list:
        # 已有记录 → 批量更新状态
        TenantModelService.batch_update_model_status([m.id for m in model_obj_list], status)
    else:
        # 无记录
        if status == ActiveStatusEnum.ACTIVE.value:
            # 默认即启用，无需写入记录
            return True, None
        # 设为 inactive → 创建禁用记录，需从工厂定义中查找模型元信息
        factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
        if not factory_info:
            return False, f"Provider '{provider_name}' not found"
        llms = factory_info[0].get("llm", [])
        target_llm = [llm for llm in llms if llm["llm_name"] == model_name]
        if not target_llm:
            return False, f"provider {provider_name} model {model_name} not found"

        for model_type in _factory_model_types(target_llm[0]):
            TenantModelService.insert(
                id=get_uuid(),
                model_name=model_name,
                model_type=model_type,
                provider_id=provider_obj.id,
                instance_id=instance_obj.id,
                status=status,
                extra=json.dumps({"max_tokens": target_llm[0].get("max_tokens", 8192), "is_tools": target_llm[0].get("is_tools", False)})
            )

    return True, None


# =============================================================================
# 模型聊天测试
# =============================================================================

async def chat_to_model(tenant_id: str, provider_name: str, instance_name: str, model_name: str, message: str, stream: bool = False, thinking: bool = False):
    """向指定模型发送聊天消息（测试/调试用）。

    使用复合名称 ``model_name@instance_name@provider_name`` 查找模型配置，
    通过 LLMBundle 创建 LLM 实例进行调用。

    :param stream: True 时不在此处消费流，返回 llm 对象和配置供上层使用 SSE 推送。
    :param thinking: 是否启用深度推理模式（传给模型配置，具体由 LLMBundle 处理）。
    """
    from api.db.services.llm_service import LLMBundle

    # 用复合名称查找模型配置
    composite_name = f"{model_name}@{instance_name}@{provider_name}"
    try:
        model_config = get_model_config_from_provider_instance(tenant_id, LLMType.CHAT.value, composite_name)
    except LookupError:
        return False, f"Model '{composite_name}' not authorized"

    if not model_config:
        return False, f"Model '{composite_name}' not found"

    llm = LLMBundle(tenant_id, model_config)

    if stream:
        # 流式模式：返回 LLM 实例供上层进行 SSE 推送
        return True, {"type": "stream", "llm": llm, "model_config": model_config}

    # 非流式模式：直接返回完整回复
    try:
        response = await llm.async_chat(
            None,
            [{"role": "user", "content": message}],
            {"temperature": 0.9},
        )
        result = {
            "answer": response,
            "reasoning_content": "",
        }
        return True, result
    except Exception as e:
        logging.exception(f"Chat to model failed: {e}")
        return False, str(e)
