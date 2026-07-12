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
# =============================================================================
# 标准库导入
# =============================================================================
import logging  # 日志记录
import os  # 环境变量读取
import sys  # 系统路径操作（用于动态注册模块）
import time  # 性能计时（API 耗时统计）
from importlib.util import module_from_spec, spec_from_file_location  # 运行时动态加载 Python 模块
from pathlib import Path  # 路径操作（扫描蓝图文件）

# =============================================================================
# 第三方库导入
# =============================================================================
from quart import Blueprint, Quart, request, g, current_app, session, jsonify  # Quart 异步 Web 框架核心
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer  # JWT 令牌序列化/反序列化
from quart_cors import cors  # 跨域资源共享（CORS）支持
from quart_auth import Unauthorized as QuartAuthUnauthorized  # Quart-Auth 未授权异常
from werkzeug.exceptions import Unauthorized as WerkzeugUnauthorized  # Werkzeug 未授权异常
from quart_schema import QuartSchema  # OpenAPI/Swagger 文档自动生成

# =============================================================================
# 项目内部模块导入
# =============================================================================
from common import settings  # 全局配置管理（Redis、数据库、密钥等）
from common.constants import StatusEnum, RetCode  # 状态码和返回码枚举
from api.db.db_models import close_connection, APIToken  # 数据库连接管理、API 令牌模型
from api.db.services import UserService  # 用户服务（查询、认证）
from api.utils.json_encode import CustomJSONEncoder  # 自定义 JSON 编码器
from api.utils import commands  # Flask CLI 命令注册
from api.utils.api_utils import server_error_response, get_json_result  # 统一 API 响应格式
from api.constants import API_VERSION  # API 版本号常量
from common.exceptions import ModelException  # 模型相关业务异常
from common.misc_utils import get_uuid  # UUID 生成工具

# 初始化全局配置（读取 service_conf.yaml、解密密码等）
settings.init_settings()

# 模块公开接口：仅暴露 app 实例
__all__ = ["app"]

# 未授权访问的统一错误消息
UNAUTHORIZED_MESSAGE = "<Unauthorized '401: Unauthorized'>"


def _unauthorized_message(error):
    """从异常对象中提取可读的未授权错误消息。

    尝试按优先级获取：error.description → repr(error) → 默认消息。
    用于 401 错误处理器中构造响应体。

    Args:
        error: 异常对象（可为 None）。

    Returns:
        str —— 错误描述字符串。
    """
    if error is None:
        return UNAUTHORIZED_MESSAGE

    description = getattr(error, "description", None)
    if description:
        return description

    try:
        return repr(error)
    except Exception:
        return UNAUTHORIZED_MESSAGE


# =============================================================================
# Quart 应用实例创建与基础配置
# =============================================================================
app = Quart(__name__)
# 启用跨域资源共享，允许所有来源访问
app = cors(app, allow_origin="*")

# 启用 OpenAPI / Swagger 文档支持
QuartSchema(app)

# 关闭严格斜杠模式：/api/v1/users 和 /api/v1/users/ 视为同一路由
app.url_map.strict_slashes = False
# 使用自定义 JSON 编码器
app.json_encoder = CustomJSONEncoder
# 注册全局异常处理器 —— 所有未捕获的 Exception 返回统一错误格式
app.errorhandler(Exception)(server_error_response)

# 配置 Quart 响应超时 —— 适配 LLM 慢响应场景（如本地 CPU 运行的 Ollama）
# Quart 默认响应超时为 60 秒，对于大模型推理来说太短
app.config["RESPONSE_TIMEOUT"] = int(os.environ.get("QUART_RESPONSE_TIMEOUT", 600))
app.config["BODY_TIMEOUT"] = int(os.environ.get("QUART_BODY_TIMEOUT", 600))

## 开发调试用：取消下面注释可跳过登录验证
# app.config["LOGIN_DISABLED"] = True
# Session 配置：使用 Redis 存储，不持久化
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = settings.decrypt_database_config(name="redis")
# 上传文件最大大小限制，默认 1GB
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 1024 * 1024 * 1024)
)
# 应用密钥 —— 用于 JWT 签名和 Session 加密
app.config['SECRET_KEY'] = settings.get_secret_key()
app.secret_key = settings.get_secret_key()
# 注册自定义 Flask CLI 命令
commands.register_commands(app)

# =============================================================================
# 认证系统类型定义（装饰器/类型提示用）
# =============================================================================
from functools import wraps  # 装饰器辅助工具
from typing import ParamSpec, TypeVar  # 泛型参数规范
from collections.abc import Awaitable, Callable, Iterable  # 抽象类型
from werkzeug.local import LocalProxy  # 线程/协程安全的惰性代理对象

T = TypeVar("T")
P = ParamSpec("P")

# 三种认证方式常量
AUTH_JWT = "JWT"    # JWT 令牌认证（Web 登录）
AUTH_API = "API"    # API 密钥认证（程序化访问）
AUTH_BETA = "BETA"  # Beta 令牌认证（特殊用途）
DEFAULT_AUTH_TYPES = (AUTH_JWT, AUTH_API)  # 默认同时支持 JWT 和 API 两种方式


def _normalize_auth_types(auth_types=None):
    """规范化认证类型参数，统一转为大写集合。

    支持多种调用方式：
    - None → 返回默认认证类型集合 {JWT, API}
    - "jwt" → {"JWT"}
    - ["jwt", "api"] → {"JWT", "API"}

    Args:
        auth_types: 认证类型字符串、可迭代对象或 None。

    Returns:
        set —— 大写认证类型字符串的集合。
    """
    if auth_types is None:
        return set(DEFAULT_AUTH_TYPES)
    if isinstance(auth_types, str):
        return {auth_types.upper()}
    if isinstance(auth_types, Iterable):
        return {str(auth_type).upper() for auth_type in auth_types}
    return {str(auth_types).upper()}


def _load_user_from_session():
    """从 Session Cookie 中恢复当前登录用户（会话回退机制）。

    背景：
    OAuth/OIDC 回调调用 ``login_user(user)`` 后会将 ``_user_id`` 写入
    Session。前端在收到 401 响应时会清除 localStorage 中的 Authorization
    请求头，导致后续请求不带任何认证头 —— 此时仍需通过服务端 Session 来
    识别用户身份。

    安全策略：
    与 JWT 路径使用相同的 access_token 有效性校验规则：
    - 拒绝空 token
    - 拒绝长度 < 32 的 token
    - 拒绝以 "INVALID_" 开头的已注销 token

    这样确保登出后被重写的 token 或数据损坏导致的短 token 无法继续维持
    过期的会话认证。

    Returns:
        User 对象（认证成功）或 None（未登录 / token 无效）。
    """
    user_id = session.get("_user_id")
    if not user_id:
        return None
    try:
        users = UserService.query(id=user_id, status=StatusEnum.VALID.value)
    except Exception:
        logging.exception("load_user from session failed")
        return None
    if not users:
        return None
    user = users[0]
    # 对 Session 中用户的 access_token 做与 JWT 路径一致的安全校验
    access_token = str(user.access_token or "").strip()
    if not access_token or len(access_token) < 32 or access_token.startswith("INVALID_"):
        return None
    logging.debug("Authenticated request via session fallback for user_id=%s", user_id)
    g.auth_type = AUTH_JWT
    g.user = user
    return user


def _load_user(auth_types=None):
    """核心认证逻辑 —— 按优先级链式尝试多种方式加载当前用户。

    这是整个认证系统的核心函数，被 login_required 装饰器和 current_user
    代理调用。认证按以下优先级依次尝试：

    **0. 缓存命中** —— 如果 g.user 已存在且 auth_type 符合要求，直接返回。

    **1. 无 Authorization 头** → 回退到 Session Cookie 认证。

    **2. Bearer Token 解析** —— 从 Authorization 头中提取 token：
       - 以 "bearer " 开头 → 取第二部分作为 token
       - 其他格式 → 整个值作为 token

    **3. Beta 令牌认证** —— 查询 APIToken.beta 字段。

    **4. JWT 令牌认证** —— 反序列化 JWT 获取 access_token，查询用户表。
       包含多层安全检查（空 token / 短 token / 数据库中 access_token 为空）。

    **5. API 令牌认证** —— 直接查询 APIToken 表。

    **6. 最终回退** —— 所有认证方式失败后，最后一次尝试 Session。

    Args:
        auth_types: 允许的认证类型集合（None 表示使用默认值）。

    Returns:
        User 对象（认证成功）或 None（认证失败）。
    """
    # 记录调用方是否显式指定了 auth_types
    explicit_auth_types = auth_types is not None
    auth_types = _normalize_auth_types(auth_types)
    # 如果 g 上下文中已有认证用户且类型匹配，直接复用（避免重复查询数据库）
    if getattr(g, "user", None) and (not explicit_auth_types or getattr(g, "auth_type", None) in auth_types):
        return g.user

    # 没有 Authorization 请求头时，尝试从 Session Cookie 恢复用户
    authorization = request.headers.get("Authorization")
    if not authorization:
        return _load_user_from_session() if AUTH_JWT in auth_types else None

    # 解析认证令牌 —— 支持 "Bearer <token>" 和裸 token 两种格式
    if authorization[:7].lower() == "bearer ":
        parts = authorization.split(maxsplit=1)
        if len(parts) < 2:
            logging.warning("Authorization header has invalid bearer format")
            return _load_user_from_session() if AUTH_JWT in auth_types else None
        auth_token = parts[1]
    else:
        auth_token = authorization

    # 初始化认证状态
    g.user = None
    g.auth_type = None
    g.auth_error_message = None

    # ---- 方式1: Beta 令牌认证 ----
    if AUTH_BETA in auth_types:
        try:
            objs = APIToken.query(beta=auth_token)
            if objs:
                user = UserService.query(id=objs[0].tenant_id, status=StatusEnum.VALID.value)
                if user:
                    g.auth_type = AUTH_BETA
                    g.user = user[0]
                    return user[0]
            g.auth_error_message = 'Authentication error: API key is invalid! '
        except Exception as e_beta:
            logging.warning(f"load_user from beta token got exception {e_beta}")
            g.auth_error_message = 'Authentication error: API key is invalid!'

    # ---- 方式2: JWT 令牌认证 ----
    if AUTH_JWT in auth_types:
        try:
            # 使用应用密钥反序列化 JWT，提取 access_token
            jwt = Serializer(secret_key=settings.get_secret_key())
            access_token = str(jwt.loads(auth_token))

            # 安全检查：拒绝空 token
            if not access_token or not access_token.strip():
                logging.warning("Authentication attempt with empty access token")
                return _load_user_from_session()

            # 安全检查：拒绝格式不正常的短 token
            if len(access_token.strip()) < 32:
                logging.warning(f"Authentication attempt with invalid token format: {len(access_token)} chars")
                return _load_user_from_session()

            # 用 access_token 查询用户（UserService.query 内部会做进一步校验）
            user = UserService.query(access_token=access_token, status=StatusEnum.VALID.value)
            if user:
                # 安全检查：数据库中 access_token 为空，拒绝认证
                if not user[0].access_token or not user[0].access_token.strip():
                    logging.warning(f"User {user[0].email} has empty access_token in database")
                    return _load_user_from_session()
                g.auth_type = AUTH_JWT
                g.user = user[0]
                return user[0]
            return _load_user_from_session()
        except Exception as e_jwt:
            logging.warning(f"load_user from jwt got exception {e_jwt}")

    # ---- 方式3: API 令牌认证（JWT 解码失败后才尝试）----
    if AUTH_API in auth_types:
        try:
            objs = APIToken.query(token=auth_token)
            if objs:
                user = UserService.query(id=objs[0].tenant_id, status=StatusEnum.VALID.value)
                if user:
                    if not user[0].access_token or not user[0].access_token.strip():
                        logging.warning(f"User {user[0].email} has empty access_token in database")
                        return _load_user_from_session() if AUTH_JWT in auth_types else None
                    g.auth_type = AUTH_API
                    g.user = user[0]
                    return user[0]
                logging.warning(f"load_user: No user found for tenant_id={objs[0].tenant_id} from APIToken")
            else:
                logging.warning(f"load_user: No APIToken found for token={auth_token[:10]}...")
        except Exception as e_api_token:
            logging.warning(f"load_user from api token got exception {e_api_token}")

    # ---- 最终回退: Session Cookie ----
    return _load_user_from_session() if AUTH_JWT in auth_types else None


# 线程/协程安全的惰性用户代理 —— 每次访问时动态调用 _load_user()
current_user = LocalProxy(_load_user)


def login_required(func: Callable[P, Awaitable[T]] = None, auth_types=None) -> Callable[P, Awaitable[T]]:
    """登录认证装饰器 —— 限制路由只能被已认证用户访问。

    使用方式（注意装饰器顺序：@app.route 必须在最外层）：

    .. code-block:: python

        @app.route('/api/v1/users')
        @login_required
        async def get_users():
            ...

    认证流程：
    1. 调用 _load_user() 按 JWT → API Token → Session 的优先级链认证
    2. 如果设置了 RAGFLOW_API_TIMING 环境变量，记录认证耗时
    3. 纯 BETA 认证模式下，失败返回 JSON 错误而非抛出异常
    4. 其他模式下，未认证抛出 QuartAuthUnauthorized（触发 401 响应）

    Args:
        func: 被装饰的路由处理函数（可选，支持无参数调用 @login_required）。
        auth_types: 允许的认证类型，None 表示默认 {JWT, API}。

    Returns:
        装饰后的 async 函数。
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # 可选：认证耗时统计（通过环境变量 RAGFLOW_API_TIMING 开启）
            timing_enabled = os.getenv("RAGFLOW_API_TIMING")
            t_start = time.perf_counter() if timing_enabled else None
            user = _load_user(auth_types)
            if timing_enabled:
                logging.info(
                    "api_timing login_required auth_ms=%.2f path=%s",
                    (time.perf_counter() - t_start) * 1000,
                    request.path,
                )
            if not user:
                # 纯 BETA 认证模式下不抛异常，返回 JSON 格式的错误信息
                if _normalize_auth_types(auth_types) == {AUTH_BETA}:
                    return get_json_result(
                        code=RetCode.DATA_ERROR,
                        message=getattr(g, "auth_error_message", None) or "Authorization is not valid!",
                    )
                # 其他模式抛出未授权异常，触发 401 错误处理器
                raise QuartAuthUnauthorized()
            return await current_app.ensure_async(func)(*args, **kwargs)

        return wrapper

    # 支持 @login_required 和 @login_required() 两种调用方式
    if func is None:
        return decorator
    return decorator(func)


def login_user(user, remember=False, duration=None, force=False, fresh=True):
    """登录用户 —— 将用户信息写入 Session。

    将用户 ID 写入 Redis Session，完成服务端登录状态持久化。
    如果用户的 ``is_active`` 为 False，除非设置 ``force=True``，
    否则登录会被拒绝。

    :param user: 要登录的用户对象。
    :type user: object
    :param remember: 是否记住登录状态（会话过期后仍保持登录）。默认 False。
    :type remember: bool
    :param duration: remember cookie 的有效时长。None 表示使用默认配置。
    :type duration: :class:`datetime.timedelta`
    :param force: 即使用户未激活（is_active=False），也强制登录。默认 False。
    :type force: bool
    :param fresh: 标记 Session 是否为"新鲜"登录。设为 False 表示非新鲜会话。默认 True。
    :type fresh: bool
    """
    if not force and not user.is_active:
        return False

    session["_user_id"] = user.id
    session["_fresh"] = fresh
    session["_id"] = get_uuid()
    return True


def logout_user():
    """登出用户 —— 清除 Session 中的用户信息和记住我 Cookie。

    无需传入用户对象，直接清除当前 Session 中的所有认证相关数据：
    - _user_id: 用户标识
    - _fresh: 新鲜会话标记
    - _id: 会话唯一 ID
    - remember_token Cookie 的清理标记
    """
    if "_user_id" in session:
        session.pop("_user_id")

    if "_fresh" in session:
        session.pop("_fresh")

    if "_id" in session:
        session.pop("_id")

    # 清除"记住我"Cookie
    COOKIE_NAME = "remember_token"
    cookie_name = current_app.config.get("REMEMBER_COOKIE_NAME", COOKIE_NAME)
    if cookie_name in request.cookies:
        session["_remember"] = "clear"
        if "_remember_seconds" in session:
            session.pop("_remember_seconds")

    return True


def search_pages_path(page_path):
    """扫描指定目录下的所有 API 蓝图模块文件。

    按三种模式匹配：
    1. *_app.py —— 传统蓝图（如 user_app.py, chat_app.py）
    2. *sdk/*.py —— SDK API 模块（如 sdk/dataset.py）
    3. *restful_apis/*.py —— RESTful API 模块（如 restful_apis/user_api.py）

    自动排除以 "." 开头的隐藏文件。

    Args:
        page_path: 要扫描的目录 Path 对象。

    Returns:
        Path 对象列表 —— 找到的所有蓝图模块文件路径。
    """
    app_path_list = [path for path in page_path.glob("*_app.py") if not path.name.startswith(".")]
    api_path_list = [path for path in page_path.glob("*sdk/*.py") if not path.name.startswith(".")]
    app_path_list.extend(api_path_list)
    restful_api_path_list = [path for path in page_path.glob("*restful_apis/*.py") if not path.name.startswith(".")]
    app_path_list.extend(restful_api_path_list)
    return app_path_list


def register_page(page_path):
    """动态加载并注册一个蓝图模块到 Quart 应用。

    这是 RAGFlow 自动发现 API 路由的核心机制。流程：
    1. 从文件路径推导 Python 模块名
    2. 运行时动态加载模块（importlib）
    3. 为模块创建 Blueprint 并注入 app 实例
    4. 根据模块路径确定 URL 前缀：
       - restful_apis → /api/v1（新版 RESTful 风格）
       - 其他 → /v1/<page_name>（传统风格）

    Args:
        page_path: 蓝图文件的 Path 对象。

    Returns:
        str —— 注册后的 URL 前缀。
    """
    path = f"{page_path}"

    # 从文件名推导页面名称：user_app.py → user
    page_name = page_path.stem.removesuffix("_app")
    # 构建完整的模块路径：api.apps.user
    module_name = ".".join(page_path.parts[page_path.parts.index("api") : -1] + (page_name,))

    # 运行时动态加载模块
    spec = spec_from_file_location(module_name, page_path)
    page = module_from_spec(spec)
    page.app = app  # 注入 Quart 应用实例
    page.manager = Blueprint(page_name, module_name)  # 创建蓝图
    sys.modules[module_name] = page  # 注册到全局模块缓存
    spec.loader.exec_module(page)  # 执行模块代码

    page_name = getattr(page, "page_name", page_name)
    # 判断是否为 RESTful API 路径（兼容 Windows 和 Linux 分隔符）
    restful_api_path = "\\restful_apis\\" if sys.platform.startswith("win") else "/restful_apis/"
    url_prefix = f"/api/{API_VERSION}" if restful_api_path in path else f"/{API_VERSION}/{page_name}"

    # 将蓝图注册到 Quart 应用
    app.register_blueprint(page.manager, url_prefix=url_prefix)
    return url_prefix


# =============================================================================
# 自动发现并注册所有蓝图模块
# =============================================================================
# 扫描的目录列表 —— 包含传统蓝图、RESTful API 和 SDK API 三个位置
pages_dir = [
    Path(__file__).parent,                                    # api/apps/
    Path(__file__).parent.parent / "api" / "apps",            # api/apps/（冗余路径，保证覆盖）
    Path(__file__).parent.parent / "api" / "apps" / "restful_apis",  # api/apps/restful_apis/
    Path(__file__).parent.parent / "api" / "apps" / "sdk",    # api/apps/sdk/
]

# 一次性扫描所有目录，加载并注册所有蓝图模块
client_urls_prefix = [register_page(path) for directory in pages_dir for path in search_pages_path(directory)]

# 注册向后兼容路由（旧版 API 路径映射到新版）
from api.apps.backward_compat import register_backward_compat_routes

register_backward_compat_routes(app)


# =============================================================================
# 全局错误处理器
# =============================================================================

@app.errorhandler(404)
async def not_found(error):
    """404 错误处理器 —— 路由未找到时返回统一 JSON 格式。"""
    logging.error(f"The requested URL {request.path} was not found")
    message = f"Not Found: {request.path}"
    response = {
        "code": RetCode.NOT_FOUND,
        "message": message,
        "data": None,
        "error": "Not Found",
    }
    return jsonify(response), RetCode.NOT_FOUND


@app.errorhandler(401)
async def unauthorized(error):
    """401 错误处理器 —— 通用未授权错误，返回统一 JSON 格式。"""
    logging.warning("Unauthorized request")
    return get_json_result(code=RetCode.UNAUTHORIZED, message=_unauthorized_message(error)), RetCode.UNAUTHORIZED


@app.errorhandler(QuartAuthUnauthorized)
async def unauthorized_quart_auth(error):
    """Quart-Auth 未授权异常专用的 401 处理器。"""
    logging.warning("Unauthorized request (quart_auth)")
    return get_json_result(code=RetCode.UNAUTHORIZED, message=repr(error)), RetCode.UNAUTHORIZED


@app.errorhandler(WerkzeugUnauthorized)
async def unauthorized_werkzeug(error):
    """Werkzeug 未授权异常专用的 401 处理器。"""
    logging.warning("Unauthorized request (werkzeug)")
    return get_json_result(code=error.code, message=error.description), RetCode.UNAUTHORIZED


@app.errorhandler(ModelException)
async def handle_model_exception(error):
    """模型异常处理器 —— LLM/Embedding 等模型相关错误。"""
    logging.warning("Forbidden request")
    return get_json_result(code=RetCode.BAD_REQUEST, message=repr(error)), 200


@app.teardown_request
def _db_close(exception):
    """请求结束时自动关闭数据库连接。

    注册为 Quart 的 teardown_request 钩子，确保每个请求结束后
    Peewee 数据库连接被正确归还到连接池，避免连接泄漏。
    如果请求过程中发生异常，记录详细日志。
    """
    if exception:
        logging.exception(f"Request failed: {exception}")
    close_connection()
