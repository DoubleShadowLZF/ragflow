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
租户模型联合服务 —— 跨多个数据表的聚合查询和模型配置解析。

本模块是 Provider API 系统的"胶水层"，联合三个独立 Service 层
（Provider / Instance / Model）完成跨表操作，包括：

- 复合模型名称 ``model@instance@provider`` 的解析与配置组装
- 从环境变量自动创建 OCR 供应商实例（MinerU / PaddleOCR / OpenDataLoader）
- 租户默认模型查询
- API Key 解析和结构化提取
"""

import logging
import os
import enum
import json
from common import settings
from common.constants import ActiveStatusEnum, LLMType, MINERU_DEFAULT_CONFIG, MINERU_ENV_KEYS, OPENDATALOADER_DEFAULT_CONFIG, OPENDATALOADER_ENV_KEYS, PADDLEOCR_DEFAULT_CONFIG, PADDLEOCR_ENV_KEYS
from api.db.services.tenant_llm_service import TenantService
from api.db.services.tenant_model_provider_service import TenantModelProviderService
from api.db.services.tenant_model_instance_service import TenantModelInstanceService
from api.db.services.tenant_model_service import TenantModelService

logger = logging.getLogger(__name__)


# =============================================================================
# 内部辅助函数
# =============================================================================

def _factory_model_types(llm: dict) -> list[str]:
    """从 LLM 工厂条目中提取模型类型列表。

    兼容 ``model_type`` 为字符串或列表两种配置格式。
    """
    model_type = llm.get("model_type")
    if isinstance(model_type, list):
        return model_type
    return [model_type] if model_type else []


def _decode_api_key_config(raw_api_key: str) -> tuple[str, bool | None, str | None]:
    """解析 API Key 配置，提取结构化信息。

    返回三元组：
    - ``api_key`` —— 纯密钥字符串
    - ``is_tools`` —— 是否启用工具调用（tools/function calling），
      未设置时返回 None
    - ``api_key_payload`` —— 完整的原始 JSON（当 api_key 字段包含
      复杂结构时保留），简单结构时返回 None

    处理三种情况：
    1. 无法解析为 JSON → 整串作为 api_key 返回
    2. JSON dict，键仅含 api_key/is_tools → 返回纯密钥 + is_tools 标志
    3. JSON dict，包含其他复杂字段 → 返回密钥 + 保留原始 payload
    """
    if not raw_api_key:
        return raw_api_key, None, None

    try:
        parsed = json.loads(raw_api_key)
    except Exception:
        return raw_api_key, None, None

    if not isinstance(parsed, dict):
        return raw_api_key, None, None

    is_tools = bool(parsed["is_tools"]) if "is_tools" in parsed else None
    # 仅含 api_key 和 is_tools → 简单模式，无需保留 payload
    if set(parsed.keys()) <= {"api_key", "is_tools"}:
        return parsed.get("api_key", ""), is_tools, None

    # 复杂模式 → 保留原始 payload 供下游使用
    return parsed.get("api_key", raw_api_key), is_tools, raw_api_key


# =============================================================================
# 模型查询与解析
# =============================================================================

def get_first_provider_model_name(tenant_id: str, provider_name: str, model_type: str | enum.Enum) -> str | None:
    """获取指定供应商下第一个活跃模型的复合名称。

    遍历该供应商的所有活跃实例和模型，返回第一个匹配类型的
    活跃模型，格式为 ``model_name@instance_name@provider_name``。

    常用于需要"随便选一个可用模型"的默认回退场景。
    """
    model_type_val = model_type if isinstance(model_type, str) else model_type.value
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return None

    for instance_obj in TenantModelInstanceService.get_all_by_provider_id(provider_obj.id):
        if instance_obj.status != ActiveStatusEnum.ACTIVE.value:
            continue
        for model_obj in TenantModelService.get_models_by_instance_id(instance_obj.id):
            if model_obj.model_type == model_type_val and model_obj.status == ActiveStatusEnum.ACTIVE.value:
                return f"{model_obj.model_name}@{instance_obj.instance_name}@{provider_name}"
    return None


# =============================================================================
# OCR 供应商初始化 —— 从环境变量自动创建
# =============================================================================

def _collect_env_config(env_keys: list[str], default_config: dict) -> dict | None:
    """从环境变量收集配置，合并默认值。

    至少找到一个环境变量时才返回有效配置；否则返回 None（表示未配置环境变量，
    不自动创建供应商）。

    Args:
        env_keys: 要读取的环境变量键名列表。
        default_config: 默认配置字典（不含环境变量覆盖值）。

    Returns:
        合并后的配置字典，或 None（未设置任何环境变量）。
    """
    config = dict(default_config)
    found = False
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            found = True
            config[key] = value
    return config if found else None


def _ensure_ocr_provider_from_env(tenant_id: str, provider_name: str, model_name: str, config: dict | None) -> str | None:
    """从环境变量确保 OCR 供应商/实例/模型存在（幂等操作）。

    这是 MinerU、PaddleOCR、OpenDataLoader 三个 OCR 引擎共用的创建逻辑：
    1. 如果供应商不存在 → 创建租户-供应商关联
    2. 如果实例不存在（按 api_key 查找）→ 创建新实例
    3. 如果模型配置不存在 → 创建 OCR 类型的模型记录

    幂等性：已存在的记录不会重复创建。

    Returns:
        复合模型名称 ``model_name@instance_name@provider_name``，或 None。
    """
    if not config:
        return None

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        TenantModelProviderService.insert(tenant_id=tenant_id, provider_name=provider_name)
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)

    api_key = json.dumps(config)
    instance_obj = TenantModelInstanceService.get_by_provider_id_and_api_key(provider_obj.id, api_key)
    if not instance_obj:
        instance_obj = TenantModelInstanceService.create_instance(
            provider_id=provider_obj.id,
            instance_name=model_name,
            api_key=api_key,
            extra="{}",
        )

    model_obj = TenantModelService.get_by_provider_id_and_instance_id_and_model_type_and_model_name(
        provider_obj.id,
        instance_obj.id,
        LLMType.OCR.value,
        model_name,
    )
    if not model_obj:
        TenantModelService.insert(
            model_name=model_name,
            provider_id=provider_obj.id,
            instance_id=instance_obj.id,
            model_type=LLMType.OCR.value,
            extra=json.dumps({"max_tokens": 0}),
        )

    return f"{model_name}@{instance_obj.instance_name}@{provider_name}"


def ensure_mineru_from_env(tenant_id: str) -> str | None:
    """从环境变量自动创建 MinerU OCR 供应商/实例/模型。

    读取 ``MINERU_ENV_KEYS`` 指定的环境变量（如 MINERU_API_KEY、MINERU_BASE_URL），
    与 ``MINERU_DEFAULT_CONFIG`` 合并后自动创建。
    """
    return _ensure_ocr_provider_from_env(
        tenant_id,
        "MinerU",
        "mineru-from-env",
        _collect_env_config(MINERU_ENV_KEYS, MINERU_DEFAULT_CONFIG),
    )


def ensure_paddleocr_from_env(tenant_id: str) -> str | None:
    """从环境变量自动创建 PaddleOCR 供应商/实例/模型。"""
    return _ensure_ocr_provider_from_env(
        tenant_id,
        "PaddleOCR",
        "paddleocr-from-env",
        _collect_env_config(PADDLEOCR_ENV_KEYS, PADDLEOCR_DEFAULT_CONFIG),
    )


def ensure_opendataloader_from_env(tenant_id: str) -> str | None:
    """从环境变量自动创建 OpenDataLoader 供应商/实例/模型。"""
    return _ensure_ocr_provider_from_env(
        tenant_id,
        "OpenDataLoader",
        "opendataloader-from-env",
        _collect_env_config(OPENDATALOADER_ENV_KEYS, OPENDATALOADER_DEFAULT_CONFIG),
    )


# =============================================================================
# 租户默认模型与复合名称解析
# =============================================================================

def get_tenant_default_model_by_type(tenant_id: str, model_type: str|enum.Enum):
    """获取租户指定类型的默认模型配置。

    从 Tenant 表中读取该租户设定的各类默认模型 ID
    （llm_id、embd_id、rerank_id、asr_id、img2txt_id、tts_id），
    然后调用 get_model_config_from_provider_instance 解析为完整配置。

    Raises:
        LookupError: 租户不存在。
        Exception: OCR 类型需要显式指定模型名；未知的模型类型。
        Exception: 未设置该类型的默认模型。
    """
    exist, tenant = TenantService.get_by_id(tenant_id)
    if not exist:
        raise LookupError("Tenant not found")
    model_type_val = model_type if isinstance(model_type, str) else model_type.value
    model_name: str = ""
    match model_type_val:
        case LLMType.EMBEDDING.value:
            model_name = tenant.embd_id
        case LLMType.SPEECH2TEXT.value:
            model_name = tenant.asr_id
        case LLMType.IMAGE2TEXT.value:
            model_name = tenant.img2txt_id
        case LLMType.CHAT.value:
            model_name = tenant.llm_id
        case LLMType.RERANK.value:
            model_name = tenant.rerank_id
        case LLMType.TTS.value:
            model_name = tenant.tts_id
        case LLMType.OCR.value:
            raise Exception("OCR model name is required")
        case _:
            raise Exception(f"Unknown model type {model_type}")
    if not model_name:
        raise Exception(f"No default {model_type} model is set.")
    return get_model_config_from_provider_instance(tenant_id, model_type, model_name)


def split_model_name(model_name: str):
    """解析复合模型名称，拆分为三个组成部分。

    支持三种格式：
    - ``"model_name"`` → (model_name, "", "")
    - ``"model_name@provider"`` → (model_name, "default", provider)
    - ``"model_name@instance@provider"`` → (model_name, instance, provider)

    Returns:
        (pure_model_name, instance_name, provider_name) 三元组。
    """
    parts = model_name.split("@")
    if len(parts) == 1:
        pure_model_name = parts[0]
        provider_name = ""
        instance_name = ""
    elif len(parts) == 2:
        pure_model_name = parts[0]
        provider_name = parts[1]
        instance_name = "default"
    else:
        pure_model_name = parts[0]
        instance_name = parts[1]
        provider_name = parts[2]
    return pure_model_name, instance_name, provider_name


# =============================================================================
# 核心函数 —— 模型配置组装
# =============================================================================

def get_model_config_from_provider_instance(tenant_id, model_type: str|enum.Enum, model_name: str):
    """根据复合模型名称解析并返回完整的模型配置字典。

    这是整个 Provider/Instance/Model 系统的**核心组装函数**。绝大多数
    下游消费者（LLMBundle、聊天接口、文档处理等）通过此函数获取可用的
    模型配置。配置组装流程：

    **0. 快速通道** —— Builtin TEI 本地 Embedding 模型。
       如果 ``COMPOSE_PROFILES`` 包含 "tei-" 且模型名匹配 ``TEI_MODEL``
       环境变量，直接返回本地 Embedding 配置（无需查询数据库）。

    **1. 数据库查找** —— 在 tenant_model 表中查找精确匹配的记录。
       找到则从 model.extra 和 instance.extra 中提取 max_tokens、is_tools、
       api_base 等字段组装配置。

    **2. 工厂回退** —— 数据库不存在记录时，从 FACTORY_LLM_INFOS 工厂定义中
       查找匹配模型。支持 SiliconFlow 国际站路由（region="intl"）。

    Args:
        tenant_id: 租户 ID。
        model_type: 模型类型（如 LLMType.CHAT / LLMType.EMBEDDING）。
        model_name: 复合模型名称（``model@instance@provider``）。

    Returns:
        dict —— 包含 ``llm_factory``, ``api_key``, ``llm_name``, ``api_base``,
        ``model_type``, ``is_tools``, ``max_tokens`` 等字段的配置字典。

    Raises:
        LookupError: 供应商/实例/模型未找到，或模型已被禁用。
    """
    pure_model_name, instance_name, provider_name = split_model_name(model_name)
    model_type_val = model_type if isinstance(model_type, str) else model_type.value

    # ---- 快速通道：Builtin TEI 本地 Embedding 模型 ----
    compose_profiles = os.getenv("COMPOSE_PROFILES", "")
    is_tei_builtin_embedding = (
            model_type_val == LLMType.EMBEDDING.value
            and "tei-" in compose_profiles
            and pure_model_name == os.getenv("TEI_MODEL", "")
            and (provider_name == "Builtin" or not provider_name)
    )
    if is_tei_builtin_embedding:
        # 直接从全局配置返回本地 Embedding 模型配置，无需查库
        embedding_cfg = settings.EMBEDDING_CFG
        return {
            "llm_factory": "Builtin",
            "api_key": embedding_cfg["api_key"],
            "llm_name": pure_model_name,
            "api_base": embedding_cfg["base_url"],
            "model_type": LLMType.EMBEDDING.value,
        }

    # ---- 步骤1: 查找供应商和实例 ----
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        raise LookupError(f"Provider {provider_name} not found for model {model_name}.")
    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_name)
    if not instance_obj:
        raise LookupError(f"Instance {instance_name} not found for model {model_name}.")

    # ---- 步骤2: 尝试从数据库精确查找模型记录 ----
    model_obj = TenantModelService.get_by_provider_id_and_instance_id_and_model_type_and_model_name(provider_obj.id, instance_obj.id, model_type_val, pure_model_name)

    api_key, is_tool, api_key_payload = _decode_api_key_config(instance_obj.api_key)
    extra_fields = json.loads(instance_obj.extra) if instance_obj.extra else {}

    if model_obj:
        # 数据库中已有记录 → 从 model.extra 组装配置
        if model_obj.status == ActiveStatusEnum.INACTIVE.value:
            raise LookupError(f"Model {model_name} is disabled.")

        model_extra = json.loads(model_obj.extra) if model_obj.extra else {}
        model_config = {
            "llm_factory": provider_obj.provider_name,
            "api_key": api_key,
            "llm_name": model_obj.model_name,
            "api_base": extra_fields.get("base_url", ""),
            "model_type": model_obj.model_type,
            "is_tools": model_extra.get("is_tools", is_tool),
            "max_tokens": model_extra.get("max_tokens", 8192),
        }
        if api_key_payload is not None:
            model_config["api_key_payload"] = api_key_payload

        return model_config
    else:
        # ---- 步骤3: 工厂回退 —— 从 FACTORY_LLM_INFOS 查找 ----
        region = extra_fields.get("region", "default")
        # SiliconFlow 国际站路由切换
        if region == "intl" and provider_name.lower() == "siliconflow":
            target_factory_name = "siliconflow_intl"
        else:
            target_factory_name = provider_name
        fac_list = [f for f in settings.FACTORY_LLM_INFOS if f["name"] == target_factory_name]
        if not fac_list:
            raise LookupError(f"Model provider config not found: {provider_name}")
        llm_list = [llm for llm in fac_list[0]["llm"] if llm["llm_name"] == pure_model_name]
        if not llm_list:
            raise LookupError(f"Model config not found: {model_name}")
        llm_info = llm_list[0]
        # 校验模型类型是否匹配
        if model_type_val not in _factory_model_types(llm_info):
            raise LookupError(f"Model {model_name} is not a {model_type_val} model.")
        model_config = {
            "llm_factory": provider_obj.provider_name,
            "api_key": api_key,
            "llm_name": llm_info["llm_name"],
            "api_base": extra_fields.get("base_url", ""),
            "model_type": model_type_val,
            "is_tools": llm_info.get("is_tools", is_tool),
            "max_tokens": llm_info.get("max_tokens", 8192),
        }
        if api_key_payload is not None:
            model_config["api_key_payload"] = api_key_payload
        return model_config


# =============================================================================
# 辅助查询函数
# =============================================================================

def get_api_key(tenant_id: str, model_name: str):
    """根据复合模型名称获取实例的 API Key。

    解析 ``model@instance@provider`` 格式的名称，查找对应实例
    并返回其存储的 api_key。
    """
    _, instance_name, provider_name = split_model_name(model_name)

    if not provider_name:
        raise LookupError("Provider name is required.")
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        raise LookupError(f"Provider {provider_name} not found.")
    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_name)
    if not instance_obj:
        raise LookupError(f"Instance {instance_name} not found.")
    return instance_obj.api_key


def get_model_type_by_name(tenant_id: str, model_name: str):
    """根据复合模型名称获取该模型的类型列表。

    优先从数据库 tenant_model 表查询；如果无记录则回退到
    FACTORY_LLM_INFOS 工厂定义中查找模型类型。

    Returns:
        list[str] —— 模型类型列表（如 ["chat", "embedding"]）。
    """
    pure_model_name, instance_name, provider_name = split_model_name(model_name)
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        raise LookupError(f"Provider {provider_name} not found for model {model_name}.")
    instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_name)
    if not instance_obj:
        raise LookupError(f"Instance {instance_name} not found for model {model_name}.")
    model_objs = TenantModelService.get_by_provider_id_and_instance_id_and_model_name(provider_obj.id, instance_obj.id, pure_model_name)
    if not model_objs:
        # 数据库无记录 → 从工厂定义查找
        extra_fields = json.loads(instance_obj.extra) if instance_obj.extra else {}
        region = extra_fields.get("region", "default")
        if region == "intl" and provider_name.lower() == "siliconflow":
            target_factory_name = "siliconflow_intl"
        else:
            target_factory_name = provider_name
        fac_list = [f for f in settings.FACTORY_LLM_INFOS if f["name"] == target_factory_name]
        if not fac_list:
            raise LookupError(f"Model provider config not found: {provider_name}")
        llm_list = [llm for llm in fac_list[0]["llm"] if llm["llm_name"] == pure_model_name]
        if not llm_list:
            raise LookupError(f"Model {pure_model_name} not found for model {model_name}.")
        return _factory_model_types(llm_list[0])
    return [model_obj.model_type for model_obj in model_objs]


# =============================================================================
# 批量删除
# =============================================================================

def delete_models_by_instance_ids(instance_ids: list[str]):
    """按实例 ID 列表批量删除模型记录。"""
    return TenantModelService.delete_by_instance_ids(instance_ids)


def delete_instances_by_provider_ids(provider_ids: list[str]):
    """按供应商 ID 列表批量删除实例记录。"""
    return TenantModelInstanceService.delete_by_provider_ids(provider_ids)


# =============================================================================
# 跨实例查询
# =============================================================================

def get_models_by_tenant_and_provider_and_model_type(tenant_id: str, provider_name: str, model_type: str):
    """按租户、供应商和模型类型查询所有匹配的模型记录。

    遍历该供应商下的所有实例，汇总指定类型的所有模型。
    返回 TenantModel 对象列表（包含 id、model_name、status 等字段）。
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return []
    instances = TenantModelInstanceService.get_all_by_provider_id(provider_obj.id)
    if not instances:
        return []
    results = []
    for inst in instances:
        models = TenantModelService.get_by_provider_id_and_instance_id_and_model_type(provider_obj.id, inst.id, model_type)
        if models:
            results.extend(models)
    return results
