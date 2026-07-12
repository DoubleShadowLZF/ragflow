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
用户认证与租户管理 RESTful API 端点。

本模块提供用户注册、登录/登出、OAuth 第三方登录、个人信息管理、
租户模型配置、以及忘记密码（图形验证码 + 邮箱 OTP）等流程的 HTTP 端点。

路由前缀: ``/api/v1``（由 ``api/apps/restful_apis/__init__.py`` 注册）。

主要端点分组:

- **认证** — ``/auth/login``、``/auth/logout``、``/auth/login/channels``、``/auth/login/<channel>``
- **OAuth 回调** — ``/auth/oauth/<channel>/callback``
- **用户管理** — ``/users``（注册）、``/users/me``（查询/修改个人信息）
- **租户配置** — ``/users/me/models``（查询/更新租户模型设置）
- **密码重置** — ``/auth/password/forgot/captcha`` → ``/otp`` → ``/otp/verify`` → ``/auth/password/reset``
"""
import logging
import string
import os
import re
import secrets
import time
from datetime import datetime
import base64

from quart import make_response, redirect, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from api.apps.auth import get_auth_client
from api.db import FileType, UserTenantRole
from api.db.services.file_service import FileService
from api.db.services.user_service import TenantService, UserService, UserTenantService
from common.time_utils import current_timestamp, datetime_format, get_format_time
from common.misc_utils import download_img, get_uuid
from common.constants import RetCode
from common.connection_utils import construct_response
from api.utils.api_utils import (
    get_data_error_result,
    get_json_result,
    get_request_json,
    server_error_response,
    validate_request,
)
from api.utils.nickname_validation import validate_nickname
from api.utils.crypt import decrypt
from rag.utils.redis_conn import REDIS_CONN
from api.apps import login_required, current_user, login_user, logout_user
from api.utils.web_utils import (
    send_email_html,
    OTP_LENGTH,
    OTP_TTL_SECONDS,
    ATTEMPT_LIMIT,
    ATTEMPT_LOCK_SECONDS,
    RESEND_COOLDOWN_SECONDS,
    otp_keys,
    hash_code,
    captcha_key,
)
from common import settings


@manager.route("/auth/login", methods=["POST"])  # noqa: F821
async def login():
    """
    用户登录端点。

    支持邮箱+密码登录，密码经 base64 加密传输。
    登录成功后返回用户信息及 access_token。
    """
    json_body = await get_request_json()
    if not json_body:
        logging.warning("Login failed: invalid or empty JSON body")
        return get_json_result(data=False, code=RetCode.AUTHENTICATION_ERROR, message="Unauthorized!")

    email = json_body.get("email", "")

    users = UserService.query(email=email)
    if not users:
        logging.warning("Login failed: email not registered")
        return get_json_result(
            data=False,
            code=RetCode.AUTHENTICATION_ERROR,
            message=f"Email: {email} is not registered!",
        )

    password = json_body.get("password")
    try:
        password = decrypt(password)
    except BaseException:
        logging.warning("Login failed: password decryption error")
        return get_json_result(data=False, code=RetCode.SERVER_ERROR, message="Fail to crypt password")

    user = UserService.query_user(email, password)

    if user and hasattr(user, 'is_active') and user.is_active == "0":
        logging.warning("Login failed: disabled account for user_id=%s", user.id)
        return get_json_result(
            data=False,
            code=RetCode.FORBIDDEN,
            message="This account has been disabled, please contact the administrator!",
        )
    elif user:
        user.access_token = get_uuid()
        login_user(user)
        user.update_time = current_timestamp()
        user.update_date = datetime_format(datetime.now())
        user.save()
        logging.info("Login successful: user_id=%s", user.id)
        msg = "Welcome back!"

        return await construct_response(data=user.to_safe_dict(for_self=True), auth=user.get_id(), message=msg)
    else:
        logging.warning("Login failed: wrong credentials")
        return get_json_result(
            data=False,
            code=RetCode.AUTHENTICATION_ERROR,
            message="Email and password do not match!",
        )


@manager.route("/auth/login/channels", methods=["GET"])  # noqa: F821
async def get_login_channels():
    """获取所有已配置的第三方认证渠道（OAuth/OIDC）。"""
    try:
        channels = []
        for channel, config in settings.OAUTH_CONFIG.items():
            channels.append(
                {
                    "channel": channel,
                    "display_name": config.get("display_name", channel.title()),
                    "icon": config.get("icon", "sso"),
                }
            )
        return get_json_result(data=channels)
    except Exception as e:
        logging.exception(e)
        return get_json_result(data=[], message=f"Load channels failure, error: {str(e)}", code=RetCode.EXCEPTION_ERROR)


@manager.route("/auth/login/<channel>", methods=["GET"])  # noqa: F821
async def oauth_login(channel):
    """OAuth/OIDC 第三方登录入口，重定向到认证提供方的授权页面。"""
    channel_config = settings.OAUTH_CONFIG.get(channel)
    if not channel_config:
        raise ValueError(f"Invalid channel name: {channel}")
    auth_cli = get_auth_client(channel_config)

    state = get_uuid()
    session["oauth_state"] = state
    auth_url = auth_cli.get_authorization_url(state)
    logging.info("OAuth login initiated: channel='%s', state='%s'", channel, state)
    return redirect(auth_url)


@manager.route("/auth/oauth/<channel>/callback", methods=["GET"])  # noqa: F821
async def oauth_callback(channel):
    """OAuth/OIDC 回调处理 — 验证 state、换取 token、获取用户信息并登录或自动注册。"""
    try:
        channel_config = settings.OAUTH_CONFIG.get(channel)
        if not channel_config:
            raise ValueError(f"Invalid channel name: {channel}")
        auth_cli = get_auth_client(channel_config)

        # Check the state
        state = request.args.get("state")
        if not state or state != session.get("oauth_state"):
            return redirect("/?error=invalid_state")
        session.pop("oauth_state", None)

        # Obtain the authorization code
        code = request.args.get("code")
        if not code:
            return redirect("/?error=missing_code")

        # Exchange authorization code for access token
        if hasattr(auth_cli, "async_exchange_code_for_token"):
            token_info = await auth_cli.async_exchange_code_for_token(code)
        else:
            token_info = auth_cli.exchange_code_for_token(code)
        access_token = token_info.get("access_token")
        if not access_token:
            return redirect("/?error=token_failed")

        id_token = token_info.get("id_token")

        # Fetch user info
        if hasattr(auth_cli, "async_fetch_user_info"):
            user_info = await auth_cli.async_fetch_user_info(access_token, id_token=id_token)
        else:
            user_info = auth_cli.fetch_user_info(access_token, id_token=id_token)
        if not user_info.email:
            return redirect("/?error=email_missing")

        # Login or register
        users = UserService.query(email=user_info.email)
        user_id = get_uuid()

        if not users:
            try:
                try:
                    avatar = await download_img(user_info.avatar_url)
                except Exception as e:
                    logging.exception(e)
                    avatar = ""

                users = user_register(
                    user_id,
                    {
                        "access_token": get_uuid(),
                        "email": user_info.email,
                        "avatar": avatar,
                        "nickname": user_info.nickname,
                        "login_channel": channel,
                        "last_login_time": get_format_time(),
                        "is_superuser": False,
                    },
                )

                if not users:
                    raise Exception(f"Failed to register {user_info.email}")
                if len(users) > 1:
                    raise Exception(f"Same email: {user_info.email} exists!")

                # Try to log in
                user = users[0]
                login_user(user)
                return redirect(f"/?auth={user.get_id()}")

            except Exception as e:
                rollback_user_registration(user_id)
                logging.exception(e)
                return redirect(f"/?error={str(e)}")

        # User exists, try to log in
        user = users[0]
        user.access_token = get_uuid()
        if user and hasattr(user, 'is_active') and user.is_active == "0":
            return redirect("/?error=user_inactive")

        login_user(user)
        user.save()
        return redirect(f"/?auth={user.get_id()}")
    except Exception as e:
        logging.exception(e)
        return redirect(f"/?error={str(e)}")


@manager.route("/auth/logout", methods=["POST"])  # noqa: F821
@login_required
async def log_out():
    """用户登出 — 使 access_token 失效并清除登录会话。"""
    user = current_user._get_current_object() if hasattr(current_user, "_get_current_object") else current_user
    user_id = user.id
    user.access_token = f"INVALID_{secrets.token_hex(16)}"
    saved = user.save()
    if saved == 0:
        logging.error("Logout failed to persist access token update: user_id=%s", user_id)
        return get_json_result(code=RetCode.SERVER_ERROR, data=False, message="Failed to update access token")
    logout_user()
    logging.info("Logout: user_id=%s, access_token invalidated", user_id)
    return get_json_result(data=True)


@manager.route("/users/me", methods=["PATCH"])  # noqa: F821
@login_required
async def setting_user():
    """更新当前用户设置（昵称、密码等），密码修改需先验证旧密码。"""
    update_dict = {}
    request_data = await get_request_json()
    if request_data.get("password"):
        new_password = request_data.get("new_password")
        if not check_password_hash(current_user.password, decrypt(request_data["password"])):
            return get_json_result(
                data=False,
                code=RetCode.AUTHENTICATION_ERROR,
                message="Password error!",
            )

        if new_password:
            update_dict["password"] = generate_password_hash(decrypt(new_password))

    for k in request_data.keys():
        if k in [
            "password",
            "new_password",
            "email",
            "status",
            "is_superuser",
            "login_channel",
            "is_anonymous",
            "is_active",
            "is_authenticated",
            "last_login_time",
        ]:
            continue
        update_dict[k] = request_data[k]

    if "nickname" in update_dict:
        error_message, error_code = validate_nickname(update_dict["nickname"])
        if error_message:
            return get_json_result(data=False, message=error_message, code=error_code)
        update_dict["nickname"] = update_dict["nickname"].strip()

    try:
        UserService.update_by_id(current_user.id, update_dict)
        return get_json_result(data=True)
    except Exception as e:
        logging.exception(e)
        return get_json_result(data=False, message="Update failure!", code=RetCode.EXCEPTION_ERROR)


@manager.route("/users/me", methods=["GET"])  # noqa: F821
@login_required
async def user_profile():
    """获取当前用户的个人信息（ID、昵称、邮箱等）。"""
    return get_json_result(data=current_user.to_safe_dict(for_self=True))


def rollback_user_registration(user_id):
    """注册失败时回滚：依次删除已创建的 User、Tenant、UserTenant 记录。"""
    try:
        UserService.delete_by_id(user_id)
    except Exception:
        pass
    try:
        TenantService.delete_by_id(user_id)
    except Exception:
        pass
    try:
        u = UserTenantService.query(tenant_id=user_id)
        if u:
            UserTenantService.delete_by_id(u[0].id)
    except Exception:
        pass


def user_register(user_id, user):
    """创建新用户及其个人租户，初始化文件根目录和用户-租户关联。

    返回 UserService 查询到的用户列表；失败时调用方应调用
    ``rollback_user_registration`` 清理残留数据。
    """
    user["id"] = user_id
    tenant = {
        "id": user_id,
        "name": user["nickname"] + "‘s Kingdom",
        "llm_id": settings.CHAT_MDL,
        "embd_id": settings.EMBEDDING_MDL,
        "asr_id": settings.ASR_MDL,
        "parser_ids": settings.PARSERS,
        "img2txt_id": settings.IMAGE2TEXT_MDL,
        "rerank_id": settings.RERANK_MDL,
    }
    usr_tenant = {
        "tenant_id": user_id,
        "user_id": user_id,
        "invited_by": user_id,
        "role": UserTenantRole.OWNER,
    }
    file_id = get_uuid()
    file = {
        "id": file_id,
        "parent_id": file_id,
        "tenant_id": user_id,
        "created_by": user_id,
        "name": "/",
        "type": FileType.FOLDER.value,
        "size": 0,
        "location": "",
    }

    # tenant_llm = get_init_tenant_llm(user_id)

    if not UserService.save(**user):
        return None
    TenantService.insert(**tenant)
    UserTenantService.insert(**usr_tenant)
    # TenantLLMService.insert_many(tenant_llm)
    FileService.insert(file)
    return UserService.query(email=user["email"])


@manager.route("/users", methods=["POST"])  # noqa: F821
@validate_request("nickname", "email", "password")
async def user_add():
    """注册新用户 — 校验邮箱格式和唯一性，创建用户及个人租户，自动登录。"""

    if not settings.REGISTER_ENABLED:
        return get_json_result(
            data=False,
            message="User registration is disabled!",
            code=RetCode.OPERATING_ERROR,
        )

    req = await get_request_json()
    email_address = req["email"]

    # Validate the email address
    if not re.match(r"^[\w\._-]+@([\w_-]+\.)+[\w-]{2,}$", email_address):
        return get_json_result(
            data=False,
            message=f"Invalid email address: {email_address}!",
            code=RetCode.OPERATING_ERROR,
        )

    # Check if the email address is already used
    if UserService.query(email=email_address):
        return get_json_result(
            data=False,
            message=f"Email: {email_address} has already registered!",
            code=RetCode.OPERATING_ERROR,
        )

    # Construct user info data
    nickname = req["nickname"]
    error_message, error_code = validate_nickname(nickname)
    if error_message:
        return get_json_result(data=False, message=error_message, code=error_code)
    nickname = nickname.strip()

    user_dict = {
        "access_token": get_uuid(),
        "email": email_address,
        "nickname": nickname,
        "password": decrypt(req["password"]),
        "login_channel": "password",
        "last_login_time": get_format_time(),
        "is_superuser": False,
    }

    user_id = get_uuid()
    try:
        users = user_register(user_id, user_dict)
        if not users:
            raise Exception(f"Fail to register {email_address}.")
        if len(users) > 1:
            raise Exception(f"Same email: {email_address} exists!")
        user = users[0]
        login_user(user)
        return await construct_response(
            data=user.to_safe_dict(for_self=True),
            auth=user.get_id(),
            message=f"{nickname}, welcome aboard!",
        )
    except Exception as e:
        rollback_user_registration(user_id)
        logging.exception(e)
        return get_json_result(
            data=False,
            message=f"User registration failure, error: {str(e)}",
            code=RetCode.EXCEPTION_ERROR,
        )


@manager.route("/users/me/models", methods=["GET"])  # noqa: F821
@login_required
async def tenant_info():
    """获取当前用户所属租户的模型配置信息（LLM、Embedding、ASR 等）。"""
    try:
        tenants = TenantService.get_info_by(current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")
        return get_json_result(data=tenants[0])
    except Exception as e:
        return server_error_response(e)


@manager.route("/users/me/models", methods=["PATCH"])  # noqa: F821
@login_required
@validate_request("tenant_id", "asr_id", "embd_id", "img2txt_id", "llm_id")
async def set_tenant_info():
    """更新租户的模型配置（LLM、Embedding、ASR、Image2Text、Rerank 等）。"""
    req = await get_request_json()
    try:
        tid = req.pop("tenant_id")
        TenantService.update_by_id(tid, req)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/auth/password/forgot/captcha", methods=["POST"])  # noqa: F821
async def forget_get_captcha():
    """忘记密码第一步：生成图形验证码图片并缓存到 Redis（key 为 captcha:{email}，有效期 60 秒）。

    返回 JPEG 图片。
    """
    email = (request.args.get("email") or "")
    if not email:
        return get_json_result(data=False, code=RetCode.ARGUMENT_ERROR, message="email is required")

    users = UserService.query(email=email)
    if not users:
        return get_json_result(data=False, code=RetCode.DATA_ERROR, message="invalid email")

    # Generate captcha text
    allowed = string.ascii_uppercase + string.digits
    captcha_text = "".join(secrets.choice(allowed) for _ in range(OTP_LENGTH))
    REDIS_CONN.set(captcha_key(email), captcha_text, 60) # Valid for 60 seconds

    from captcha.image import ImageCaptcha
    image = ImageCaptcha(width=300, height=120, font_sizes=[50, 60, 70])
    img_bytes = image.generate(captcha_text).read()
    response = await make_response(img_bytes)
    response.headers.set("Content-Type", "image/JPEG")
    return response


@manager.route("/auth/password/forgot/otp", methods=["POST"])  # noqa: F821
async def forget_send_otp():
    """忘记密码第二步：校验图形验证码，通过后生成邮箱 OTP 验证码并发送邮件。

    流程：
    1. 验证图形验证码（大小写不敏感），验证后删除防止重用。
    2. 检查重发冷却时间。
    3. 生成大写字母 OTP，存储 hash+salt 到 Redis。
    4. 通过邮件发送 OTP。
    """
    req = await get_request_json()
    email = req.get("email") or ""
    captcha = (req.get("captcha") or "").strip()

    if not email or not captcha:
        return get_json_result(data=False, code=RetCode.ARGUMENT_ERROR, message="email and captcha required")

    users = UserService.query(email=email)
    if not users:
        return get_json_result(data=False, code=RetCode.DATA_ERROR, message="invalid email")

    stored_captcha = REDIS_CONN.get(captcha_key(email))
    if not stored_captcha:
        return get_json_result(data=False, code=RetCode.NOT_EFFECTIVE, message="invalid or expired captcha")
    if (stored_captcha or "").strip().lower() != captcha.lower():
        return get_json_result(data=False, code=RetCode.AUTHENTICATION_ERROR, message="invalid or expired captcha")

    # Delete captcha to prevent reuse
    REDIS_CONN.delete(captcha_key(email))

    k_code, k_attempts, k_last, k_lock = otp_keys(email)
    now = int(time.time())
    last_ts = REDIS_CONN.get(k_last)
    if last_ts:
        try:
            elapsed = now - int(last_ts)
        except Exception:
            elapsed = RESEND_COOLDOWN_SECONDS
        remaining = RESEND_COOLDOWN_SECONDS - elapsed
        if remaining > 0:
            return get_json_result(data=False, code=RetCode.NOT_EFFECTIVE, message=f"you still have to wait {remaining} seconds")

    # Generate OTP (uppercase letters only) and store hashed
    otp = "".join(secrets.choice(string.ascii_uppercase) for _ in range(OTP_LENGTH))
    salt = os.urandom(16)
    code_hash = hash_code(otp, salt)
    REDIS_CONN.set(k_code, f"{code_hash}:{salt.hex()}", OTP_TTL_SECONDS)
    REDIS_CONN.set(k_attempts, 0, OTP_TTL_SECONDS)
    REDIS_CONN.set(k_last, now, OTP_TTL_SECONDS)
    REDIS_CONN.delete(k_lock)

    ttl_min = OTP_TTL_SECONDS // 60

    try:
        await send_email_html(
            subject="Your Password Reset Code",
            to_email=email,
            template_key="reset_code",
            code=otp,
            ttl_min=ttl_min,
        )

    except Exception as e:
        logging.exception(e)
        return get_json_result(data=False, code=RetCode.SERVER_ERROR, message="failed to send email")

    return get_json_result(data=True, code=RetCode.SUCCESS, message="verification passed, email sent")


def _verified_key(email: str) -> str:
    """返回 OTP 验证通过后 Redis 标记的键名。"""
    return f"otp:verified:{email}"


@manager.route("/auth/password/forgot/otp/verify", methods=["POST"])  # noqa: F821
async def forget_verify_otp():
    """忘记密码第三步：验证邮箱 OTP。

    成功后：
    - 删除 OTP 和尝试计数器。
    - 在 Redis 中设置已验证标记（``_verified_key``），供重置密码步骤使用。
    失败次数达到上限时锁定一段时间。

    Request JSON: { email, otp }
    """
    req = await get_request_json()
    email = req.get("email") or ""
    otp = (req.get("otp") or "").strip()

    if not all([email, otp]):
        return get_json_result(data=False, code=RetCode.ARGUMENT_ERROR, message="email and otp are required")

    users = UserService.query(email=email)
    if not users:
        return get_json_result(data=False, code=RetCode.DATA_ERROR, message="invalid email")

    # Verify OTP from Redis
    k_code, k_attempts, k_last, k_lock = otp_keys(email)
    if REDIS_CONN.get(k_lock):
        return get_json_result(data=False, code=RetCode.NOT_EFFECTIVE, message="too many attempts, try later")

    stored = REDIS_CONN.get(k_code)
    if not stored:
        return get_json_result(data=False, code=RetCode.NOT_EFFECTIVE, message="expired otp")

    try:
        stored_hash, salt_hex = str(stored).split(":", 1)
        salt = bytes.fromhex(salt_hex)
    except Exception:
        return get_json_result(data=False, code=RetCode.EXCEPTION_ERROR, message="otp storage corrupted")

    calc = hash_code(otp.upper(), salt)
    if calc != stored_hash:
        # bump attempts
        try:
            attempts = int(REDIS_CONN.get(k_attempts) or 0) + 1
        except Exception:
            attempts = 1
        REDIS_CONN.set(k_attempts, attempts, OTP_TTL_SECONDS)
        if attempts >= ATTEMPT_LIMIT:
            REDIS_CONN.set(k_lock, int(time.time()), ATTEMPT_LOCK_SECONDS)
        return get_json_result(data=False, code=RetCode.AUTHENTICATION_ERROR, message="expired otp")

    # Success: consume OTP and attempts; mark verified
    REDIS_CONN.delete(k_code)
    REDIS_CONN.delete(k_attempts)
    REDIS_CONN.delete(k_last)
    REDIS_CONN.delete(k_lock)

    # set verified flag with limited TTL, reuse OTP_TTL_SECONDS or smaller window
    try:
        REDIS_CONN.set(_verified_key(email), "1", OTP_TTL_SECONDS)
    except Exception:
        return get_json_result(data=False, code=RetCode.SERVER_ERROR, message="failed to set verification state")

    return get_json_result(data=True, code=RetCode.SUCCESS, message="otp verified")


@manager.route("/auth/password/reset", methods=["POST"])  # noqa: F821
async def forget_reset_password():
    """忘记密码第四步：重置密码。

    要求 OTP 已验证（Redis 中存在 ``_verified_key`` 标记）。
    步骤：
    1. 校验 Redis 中的已验证标记。
    2. 解密并比对两次输入的新密码。
    3. 更新密码。
    4. 清除已验证标记，自动登录。

    Request JSON: { email, new_password, confirm_new_password }
    """
    
    req = await get_request_json()
    email = req.get("email") or ""
    new_pwd = req.get("new_password")
    new_pwd2 = req.get("confirm_new_password")

    if not all([email, new_pwd, new_pwd2]):
        return get_json_result(data=False, code=RetCode.ARGUMENT_ERROR, message="email and passwords are required")

    if not REDIS_CONN.get(_verified_key(email)):
        return get_json_result(data=False, code=RetCode.AUTHENTICATION_ERROR, message="email not verified")

    new_pwd_base64 = decrypt(new_pwd)
    new_pwd_string = base64.b64decode(new_pwd_base64).decode('utf-8')
    new_pwd2_string = base64.b64decode(decrypt(new_pwd2)).decode('utf-8')

    if new_pwd_string != new_pwd2_string:
        return get_json_result(data=False, code=RetCode.ARGUMENT_ERROR, message="passwords do not match")

    users = UserService.query_user_by_email(email=email)
    if not users:
        return get_json_result(data=False, code=RetCode.DATA_ERROR, message="invalid email")
    
    user = users[0]
    try:
        UserService.update_user_password(user.id, new_pwd_base64)
    except Exception as e:
        logging.exception(e)
        return get_json_result(data=False, code=RetCode.EXCEPTION_ERROR, message="failed to reset password")

    # clear verified flag
    try:
        REDIS_CONN.delete(_verified_key(email))
    except Exception:
        pass

    msg = "Password reset successful. Logged in."
    return await construct_response(data=user.to_safe_dict(for_self=True), auth=user.get_id(), message=msg)


