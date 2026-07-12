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
import os
import json
import secrets
import logging
from datetime import date

from common.constants import RAG_FLOW_SERVICE_NAME
from common.file_utils import get_project_base_directory
from common.config_utils import get_base_config, decrypt_database_config
from common.misc_utils import pip_install_torch
from common.constants import SVR_QUEUE_NAME, Storage

# 文档检索引擎连接器（Elasticsearch / Infinity / OpenSearch / OceanBase）
import rag.utils
import rag.utils.es_conn
import rag.utils.infinity_conn
import rag.utils.ob_conn
import rag.utils.opensearch_conn

# 对象存储连接器（Azure Blob / S3 / MinIO / GCS / OSS / OpenDAL）
from rag.utils.azure_sas_conn import RAGFlowAzureSasBlob
from rag.utils.azure_spn_conn import RAGFlowAzureSpnBlob
from rag.utils.gcs_conn import RAGFlowGCS
from rag.utils.minio_conn import RAGFlowMinio
from rag.utils.opendal_conn import OpenDALStorage
from rag.utils.redis_conn import REDIS_CONN
from rag.utils.s3_conn import RAGFlowS3
from rag.utils.oss_conn import RAGFlowOSS

from rag.nlp import search

# Memory 模块的检索引擎连接器（用于消息存储）
import memory.utils.es_conn as memory_es_conn
import memory.utils.infinity_conn as memory_infinity_conn
import memory.utils.ob_conn as memory_ob_conn

# =============================================================================
#  基础运行环境配置
# =============================================================================

# 时区设置，默认使用 Asia/Shanghai
TIMEZONE = os.getenv("TZ", "Asia/Shanghai")

# =============================================================================
#  LLM / 模型相关配置（由 init_settings() 动态初始化）
# =============================================================================

LLM = None                       # LLM 实例（延迟初始化）
LLM_FACTORY = None               # LLM 厂商名称（如 "OpenAI", "DeepSeek"）
LLM_BASE_URL = None              # LLM API 基础 URL
CHAT_MDL = ""                    # 默认聊天模型
EMBEDDING_MDL = ""               # 默认嵌入模型
RERANK_MDL = ""                  # 默认重排序模型
ASR_MDL = ""                     # 默认语音识别模型
IMAGE2TEXT_MDL = ""              # 默认图像理解模型

CHAT_CFG = ""                    # 聊天模型完整配置（含 api_key、base_url）
EMBEDDING_CFG = ""               # 嵌入模型完整配置
RERANK_CFG = ""                  # 重排序模型完整配置
ASR_CFG = ""                     # 语音识别模型完整配置
IMAGE2TEXT_CFG = ""              # 图像理解模型完整配置
API_KEY = None                   # LLM API 密钥
PARSERS = None                   # 文档解析器列表
HOST_IP = None                   # 服务主机 IP
HOST_PORT = None                 # 服务端口
SECRET_KEY = None                # JWT / 会话加密密钥
FACTORY_LLM_INFOS = None         # 所有可用 LLM 厂商信息
ALLOWED_LLM_FACTORIES = None     # 允许使用的 LLM 厂商列表

# =============================================================================
#  数据库与检索引擎配置
# =============================================================================

DATABASE_TYPE = os.getenv("DB_TYPE", "mysql")                 # 数据库类型（mysql / postgresql）
DATABASE = decrypt_database_config(name=DATABASE_TYPE)        # 数据库连接配置（解密后）

# 认证配置
AUTHENTICATION_CONF = None

# 客户端认证
CLIENT_AUTHENTICATION = None     # 是否启用客户端认证
HTTP_APP_KEY = None              # HTTP 应用密钥
GITHUB_OAUTH = None              # GitHub OAuth 配置
FEISHU_OAUTH = None              # 飞书 OAuth 配置
OAUTH_CONFIG = None              # OAuth 通用配置

# 文档检索引擎类型（elasticsearch / infinity / opensearch / oceanbase）
DOC_ENGINE = os.getenv('DOC_ENGINE', 'elasticsearch')
DOC_ENGINE_INFINITY = (DOC_ENGINE.lower() == "infinity")
DOC_ENGINE_OCEANBASE = (DOC_ENGINE.lower() == "oceanbase")

docStoreConn = None              # 文档检索引擎连接器实例
msgStoreConn = None              # 消息检索引擎连接器实例

retriever = None                 # 文本检索器实例
kg_retriever = None              # 知识图谱检索器实例

# =============================================================================
#  用户注册与认证配置
# =============================================================================

# 用户注册开关：1=开启，0=关闭
REGISTER_ENABLED = 1

# SSO 专属模式：为 True 时隐藏密码登录表单，仅允许 SSO 登录
DISABLE_PASSWORD_LOGIN = False

# =============================================================================
#  沙箱执行器与邮件配置
# =============================================================================

# 沙箱执行器管理器地址
SANDBOX_HOST = None
# 强测试迭代次数
STRONG_TEST_COUNT = int(os.environ.get("STRONG_TEST_COUNT", "8"))

# SMTP 邮件配置
SMTP_CONF = None
MAIL_SERVER = ""
MAIL_PORT = 000
MAIL_USE_SSL = True
MAIL_USE_TLS = False
MAIL_USERNAME = ""
MAIL_PASSWORD = ""
MAIL_DEFAULT_SENDER = ()
MAIL_FRONTEND_URL = ""

# =============================================================================
#  外部服务连接配置（由 init_settings() 动态填充）
# =============================================================================

ES = {}                          # Elasticsearch 配置
INFINITY = {}                    # Infinity 向量数据库配置
AZURE = {}                       # Azure Blob 存储配置
S3 = {}                          # AWS S3 配置
MINIO = {}                       # MinIO 对象存储配置
OB = {}                          # OceanBase 配置
OSS = {}                         # 阿里云 OSS 配置
OS = {}                          # OpenSearch 配置
GCS = {}                         # Google Cloud Storage 配置

# =============================================================================
#  文档处理与嵌入批处理配置
# =============================================================================

DOC_MAXIMUM_SIZE: int = 128 * 1024 * 1024   # 文档最大大小（默认 128MB）
DOC_BULK_SIZE: int = 4                       # 文档批量处理大小
EMBEDDING_BATCH_SIZE: int = 16               # 嵌入向量批处理大小

PARALLEL_DEVICES: int = 0                    # 可用 GPU 数量（由 check_and_install_torch 检测）

# =============================================================================
#  对象存储配置
# =============================================================================

STORAGE_IMPL_TYPE = os.getenv('STORAGE_IMPL', 'MINIO')  # 对象存储类型
STORAGE_IMPL = None                                       # 对象存储实例（延迟初始化）

def get_svr_queue_name(priority: int, suffix: str = "common") -> str:
    """
    生成任务执行器的 Redis 队列名称，支持优先级和类型两个维度。

    队列命名规则：{前缀}.{优先级}.common
    目前 suffix 参数预留，实际仅使用 "common" 类型。

    Args:
        priority: 任务优先级（0=低优先级, 1=高优先级）
        suffix: 任务类型后缀（common/resume/graphrag/raptor/mindmap）
                当前仅 "common" 在使用，其余类型为预留字段。

    Returns:
        str: 队列名称，如 "te.0.common" 或 "te.1.common"

    Examples:
        get_svr_queue_name(0, "common") -> "te.0.common"
        get_svr_queue_name(1, "common") -> "te.1.common"
        get_svr_queue_name(0) -> "te.0.common"  # 默认 suffix="common"
    """
    return f"{SVR_QUEUE_NAME}.{priority}.common"


def get_svr_queue_names(suffix: str):
    """获取按优先级排序的队列名称列表（高优先级在前）。

    Task Executor 会按照此顺序消费队列，优先处理高优先级任务。
    """
    return [get_svr_queue_name(priority, suffix) for priority in [1, 0]]

def init_secret_key():
    """初始化密钥：优先从环境变量读取，其次从配置文件读取。

    密钥长度至少需要 32 字符，用于 JWT 签名和会话加密。

    Returns:
        str | None: 有效的密钥字符串，或 None 表示需自动生成
    """
    secret_key = os.environ.get("RAGFLOW_SECRET_KEY")
    if secret_key and len(secret_key) >= 32:
        return secret_key

    # 检查配置文件中是否已设置非默认的密钥
    configured_key = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("secret_key")
    if configured_key and configured_key != str(date.today()) and len(configured_key) >= 32:
        return configured_key
    return None


def get_secret_key():
    """获取密钥（懒加载模式）。"""
    global SECRET_KEY
    if SECRET_KEY is None:
        return _get_or_create_secret_key()
    return SECRET_KEY


def _get_or_create_secret_key():
    """生成或从 Redis 获取系统级密钥。

    密钥生成策略：
    1. 生成一个新的安全随机密钥
    2. 尝试将密钥存入 Redis（若已存在则使用已有值，保证多实例共享同一密钥）
    3. 如果是新生成的密钥，记录安全警告日志
    """
    import logging

    generated_key = secrets.token_hex(32)
    secret_key = REDIS_CONN.get_or_create_secret_key("ragflow:system:secret_key", generated_key)
    if generated_key == secret_key:
        logging.warning("SECURITY WARNING: Using auto-generated SECRET_KEY.")
    return secret_key

class StorageFactory:
    """对象存储工厂类，根据 Storage 枚举创建对应的存储连接器实例。

    支持的存储类型：
    - MINIO: MinIO 对象存储
    - AZURE_SPN / AZURE_SAS: Azure Blob（服务主体 / SAS 令牌两种认证方式）
    - AWS_S3: Amazon S3
    - OSS: 阿里云对象存储
    - OPENDAL: Apache OpenDAL 统一存储抽象
    - GCS: Google Cloud Storage
    """
    storage_mapping = {
        Storage.MINIO: RAGFlowMinio,
        Storage.AZURE_SPN: RAGFlowAzureSpnBlob,
        Storage.AZURE_SAS: RAGFlowAzureSasBlob,
        Storage.AWS_S3: RAGFlowS3,
        Storage.OSS: RAGFlowOSS,
        Storage.OPENDAL: OpenDALStorage,
        Storage.GCS: RAGFlowGCS,
    }

    @classmethod
    def create(cls, storage: Storage):
        """根据存储类型枚举创建对应的连接器实例。"""
        return cls.storage_mapping[storage]()


def init_settings():
    """初始化所有全局配置。在服务启动时调用，完成以下初始化步骤：

    1. 数据库连接配置
    2. LLM 模型默认配置（聊天/嵌入/重排序/语音/图像）
    3. 用户注册与认证配置
    4. 文档检索引擎连接器（ES / Infinity / OpenSearch / OceanBase）
    5. 消息检索引擎连接器
    6. 对象存储连接器（MinIO / S3 / Azure / OSS / GCS）
    7. 检索器（文本 + 知识图谱）
    8. 沙箱执行器、SMTP 邮件、文档处理参数
    """
    # ---- 数据库配置 ----
    global DATABASE_TYPE, DATABASE
    DATABASE_TYPE = os.getenv("DB_TYPE", "mysql")
    DATABASE = decrypt_database_config(name=DATABASE_TYPE)

    # ---- LLM 厂商及默认模型配置 ----
    global ALLOWED_LLM_FACTORIES, LLM_FACTORY, LLM_BASE_URL
    llm_settings = get_base_config("user_default_llm", {}) or {}
    llm_default_models = llm_settings.get("default_models", {}) or {}
    LLM_FACTORY = llm_settings.get("factory", "") or ""
    LLM_BASE_URL = llm_settings.get("base_url", "") or ""
    ALLOWED_LLM_FACTORIES = llm_settings.get("allowed_factories", None)

    # ---- 用户注册开关 ----
    global REGISTER_ENABLED
    try:
        REGISTER_ENABLED = int(os.environ.get("REGISTER_ENABLED", "1"))
    except Exception:
        pass

    # ---- SSO / 密码登录开关 ----
    global DISABLE_PASSWORD_LOGIN
    try:
        env_val = os.environ.get("DISABLE_PASSWORD_LOGIN", "").lower()
        if env_val in ("1", "true", "yes"):
            DISABLE_PASSWORD_LOGIN = True
        else:
            authentication_conf = get_base_config("authentication", {})
            DISABLE_PASSWORD_LOGIN = bool(authentication_conf.get("disable_password_login", False))
    except Exception:
        pass

    # ---- 加载所有 LLM 厂商信息 ----
    global FACTORY_LLM_INFOS
    try:
        with open(os.path.join(get_project_base_directory(), "conf", "llm_factories.json"), "r") as f:
            FACTORY_LLM_INFOS = json.load(f)["factory_llm_infos"]
    except Exception:
        FACTORY_LLM_INFOS = []

    global API_KEY
    API_KEY = llm_settings.get("api_key")

    # ---- 文档解析器列表 ----
    global PARSERS
    PARSERS = llm_settings.get(
        "parsers", "naive:General,qa:Q&A,resume:Resume,manual:Manual,table:Table,paper:Paper,book:Book,laws:Laws,presentation:Presentation,picture:Picture,one:One,audio:Audio,email:Email,tag:Tag"
    )

    # ---- 解析各模型的配置条目 ----
    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    chat_entry = _parse_model_entry(llm_default_models.get("chat_model", CHAT_MDL))
    embedding_entry = _parse_model_entry(llm_default_models.get("embedding_model", EMBEDDING_MDL))
    rerank_entry = _parse_model_entry(llm_default_models.get("rerank_model", RERANK_MDL))
    asr_entry = _parse_model_entry(llm_default_models.get("asr_model", ASR_MDL))
    image2text_entry = _parse_model_entry(llm_default_models.get("image2text_model", IMAGE2TEXT_MDL))

    # ---- 合并模型名称、厂商、API Key、Base URL 生成完整配置 ----
    global CHAT_CFG, EMBEDDING_CFG, RERANK_CFG, ASR_CFG, IMAGE2TEXT_CFG
    CHAT_CFG = _resolve_per_model_config(chat_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    EMBEDDING_CFG = _resolve_per_model_config(embedding_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    RERANK_CFG = _resolve_per_model_config(rerank_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    ASR_CFG = _resolve_per_model_config(asr_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    IMAGE2TEXT_CFG = _resolve_per_model_config(image2text_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)

    CHAT_MDL = CHAT_CFG.get("model", "") or ""
    EMBEDDING_MDL = EMBEDDING_CFG.get("model", "") or ""
    # 若使用 TEI（Text Embeddings Inference）容器，使用指定的嵌入模型
    compose_profiles = os.getenv("COMPOSE_PROFILES", "")
    if "tei-" in compose_profiles:
        EMBEDDING_MDL = os.getenv("TEI_MODEL", EMBEDDING_MDL or "BAAI/bge-small-en-v1.5")
    RERANK_MDL = RERANK_CFG.get("model", "") or ""
    ASR_MDL = ASR_CFG.get("model", "") or ""
    IMAGE2TEXT_MDL = IMAGE2TEXT_CFG.get("model", "") or ""

    # ---- 服务主机与端口 ----
    global HOST_IP, HOST_PORT
    HOST_IP = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HOST_PORT = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("http_port")

    # ---- 密钥初始化 ----
    global SECRET_KEY
    SECRET_KEY = init_secret_key()

    # ---- 认证配置（OAuth / SSO） ----
    authentication_conf = get_base_config("authentication", {})

    global CLIENT_AUTHENTICATION, HTTP_APP_KEY, GITHUB_OAUTH, FEISHU_OAUTH, OAUTH_CONFIG
    CLIENT_AUTHENTICATION = authentication_conf.get("client", {}).get("switch", False)
    HTTP_APP_KEY = authentication_conf.get("client", {}).get("http_app_key")
    GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
    FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")
    OAUTH_CONFIG = get_base_config("oauth", {})

    # ---- 文档检索引擎连接器初始化 ----
    # 根据 DOC_ENGINE 环境变量选择对应的检索引擎
    global DOC_ENGINE, DOC_ENGINE_INFINITY, DOC_ENGINE_OCEANBASE, docStoreConn, ES, OB, OS, INFINITY
    DOC_ENGINE = os.environ.get("DOC_ENGINE", "elasticsearch").strip()
    DOC_ENGINE_INFINITY = (DOC_ENGINE.lower() == "infinity")
    DOC_ENGINE_OCEANBASE = (DOC_ENGINE.lower() == "oceanbase")
    lower_case_doc_engine = DOC_ENGINE.lower()
    if lower_case_doc_engine == "elasticsearch":
        ES = get_base_config("es", {})
        docStoreConn = rag.utils.es_conn.ESConnection()
    elif lower_case_doc_engine == "infinity":
        INFINITY = get_base_config("infinity", {
            "uri": "infinity:23817",
            "postgres_port": 5432,
            "db_name": "default_db"
        })
        docStoreConn = rag.utils.infinity_conn.InfinityConnection()
    elif lower_case_doc_engine == "opensearch":
        OS = get_base_config("os", {})
        docStoreConn = rag.utils.opensearch_conn.OSConnection()
    elif lower_case_doc_engine == "oceanbase":
        OB = get_base_config("oceanbase", {})
        docStoreConn = rag.utils.ob_conn.OBConnection()
    elif lower_case_doc_engine == "seekdb":
        OB = get_base_config("seekdb", {})
        docStoreConn = rag.utils.ob_conn.OBConnection()
    else:
        raise Exception(f"Not supported doc engine: {DOC_ENGINE}")

    # ---- 消息检索引擎连接器（与文档引擎共用同一引擎类型） ----
    global msgStoreConn
    if DOC_ENGINE == "elasticsearch":
        ES = get_base_config("es", {})
        msgStoreConn = memory_es_conn.ESConnection()
    elif DOC_ENGINE == "infinity":
        INFINITY = get_base_config("infinity", {
            "uri": "infinity:23817",
            "postgres_port": 5432,
            "db_name": "default_db"
        })
        msgStoreConn = memory_infinity_conn.InfinityConnection()
    elif lower_case_doc_engine in ["oceanbase", "seekdb"]:
        msgStoreConn = memory_ob_conn.OBConnection()

    # ---- 对象存储连接器初始化 ----
    global AZURE, S3, MINIO, OSS, GCS
    if STORAGE_IMPL_TYPE in ['AZURE_SPN', 'AZURE_SAS']:
        AZURE = get_base_config("azure", {})
    elif STORAGE_IMPL_TYPE == 'AWS_S3':
        S3 = get_base_config("s3", {})
    elif STORAGE_IMPL_TYPE == 'MINIO':
        MINIO = decrypt_database_config(name="minio")
    elif STORAGE_IMPL_TYPE == 'OSS':
        OSS = get_base_config("oss", {})
    elif STORAGE_IMPL_TYPE == 'GCS':
        GCS = get_base_config("gcs", {})

    global STORAGE_IMPL
    storage_impl = StorageFactory.create(Storage[STORAGE_IMPL_TYPE])

    # ---- 存储加密（可选，通过 RAGFLOW_CRYPTO_ENABLED 控制） ----
    crypto_enabled = os.environ.get("RAGFLOW_CRYPTO_ENABLED", "false").lower() == "true"

    if crypto_enabled:
        try:
            from rag.utils.encrypted_storage import create_encrypted_storage
            algorithm = os.environ.get("RAGFLOW_CRYPTO_ALGORITHM", "aes-256-cbc")
            crypto_key = os.environ.get("RAGFLOW_CRYPTO_KEY")

            STORAGE_IMPL = create_encrypted_storage(storage_impl,
                algorithm=algorithm,
                key=crypto_key,
                encryption_enabled=crypto_enabled)
        except Exception as e:
            logging.error(f"Failed to initialize encrypted storage: {e}")
            STORAGE_IMPL = storage_impl
    else:
        STORAGE_IMPL = storage_impl

    # ---- 检索器初始化（文本检索 + 知识图谱检索） ----
    global retriever, kg_retriever
    retriever = search.Dealer(docStoreConn)
    from rag.graphrag import search as kg_search

    kg_retriever = kg_search.KGSearch(docStoreConn)

    # ---- 沙箱执行器 ----
    global SANDBOX_HOST
    if int(os.environ.get("SANDBOX_ENABLED", "0")):
        SANDBOX_HOST = os.environ.get("SANDBOX_HOST", "sandbox-executor-manager")

    # ---- SMTP 邮件配置 ----
    global SMTP_CONF
    SMTP_CONF = get_base_config("smtp", {})

    global MAIL_SERVER, MAIL_PORT, MAIL_USE_SSL, MAIL_USE_TLS, MAIL_USERNAME, MAIL_PASSWORD, MAIL_DEFAULT_SENDER, MAIL_FRONTEND_URL
    MAIL_SERVER = SMTP_CONF.get("mail_server", "")
    MAIL_PORT = SMTP_CONF.get("mail_port", 000)
    MAIL_USE_SSL = SMTP_CONF.get("mail_use_ssl", True)
    MAIL_USE_TLS = SMTP_CONF.get("mail_use_tls", False)
    MAIL_USERNAME = SMTP_CONF.get("mail_username", "")
    MAIL_PASSWORD = SMTP_CONF.get("mail_password", "")
    mail_default_sender = SMTP_CONF.get("mail_default_sender", [])
    if mail_default_sender and len(mail_default_sender) >= 2:
        MAIL_DEFAULT_SENDER = (mail_default_sender[0], mail_default_sender[1])
    MAIL_FRONTEND_URL = SMTP_CONF.get("mail_frontend_url", "")

    # ---- 文档处理参数 ----
    global DOC_MAXIMUM_SIZE, DOC_BULK_SIZE, EMBEDDING_BATCH_SIZE
    DOC_MAXIMUM_SIZE = int(os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024))
    DOC_BULK_SIZE = int(os.environ.get("DOC_BULK_SIZE", 4))
    EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 16))

    # 禁用 .NET 全球化支持，避免在某些环境下抛出异常
    os.environ["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "1"


def check_and_install_torch():
    """检测并安装 PyTorch，统计可用 GPU 数量。

    仅在调用时动态导入 torch，避免在不需要 GPU 的场景下引入重依赖。
    """
    global PARALLEL_DEVICES
    try:
        pip_install_torch()
        import torch.cuda
        PARALLEL_DEVICES = torch.cuda.device_count()
        logging.info(f"found {PARALLEL_DEVICES} gpus")
    except Exception:
        logging.info("can't import package 'torch'")


def _parse_model_entry(entry):
    """将模型配置条目标准化为 dict 格式。

    支持两种输入格式：
    - str: 仅模型名称，如 "gpt-4"
    - dict: 完整配置，含 name/model、factory、api_key、base_url

    Returns:
        dict: {"name": str, "factory": str|None, "api_key": str|None, "base_url": str|None}
    """
    if isinstance(entry, str):
        return {"name": entry, "factory": None, "api_key": None, "base_url": None}
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model") or ""
        return {
            "name": name,
            "factory": entry.get("factory"),
            "api_key": entry.get("api_key"),
            "base_url": entry.get("base_url"),
        }
    return {"name": "", "factory": None, "api_key": None, "base_url": None}


def _resolve_per_model_config(entry_dict, backup_factory, backup_api_key, backup_base_url):
    """将单个模型的配置与全局默认值合并，生成最终配置。

    合并规则（模型级优先于全局级）：
    1. 模型名 + 厂商拼接为 "模型名@厂商" 格式（防止不同厂商同名模型冲突）
    2. API Key 和 Base URL 优先使用模型级配置，缺失时回退到全局配置

    Returns:
        dict: {"model": str, "factory": str, "api_key": str, "base_url": str}
    """
    name = (entry_dict.get("name") or "").strip()
    m_factory = entry_dict.get("factory") or backup_factory or ""
    m_api_key = entry_dict.get("api_key") or backup_api_key or ""
    m_base_url = entry_dict.get("base_url") or backup_base_url or ""

    # 以 "模型名@厂商" 格式区分不同厂商的相同名称模型
    if name and "@" not in name and m_factory:
        name = f"{name}@{m_factory}"

    return {
        "model": name,
        "factory": m_factory,
        "api_key": m_api_key,
        "base_url": m_base_url,
    }


def print_rag_settings():
    """打印当前 RAG 系统关键配置参数到日志。"""
    logging.info(f"MAX_CONTENT_LENGTH: {DOC_MAXIMUM_SIZE}")
    logging.info(f"MAX_FILE_COUNT_PER_USER: {int(os.environ.get('MAX_FILE_NUM_PER_USER', 0))}")

