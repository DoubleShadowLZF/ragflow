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
# 标准库导入
import hashlib  # 哈希算法，用于生成网关路由
from datetime import datetime  # 日期时间处理
import logging  # 日志记录

import peewee  # 轻量级 ORM 框架
from werkzeug.security import generate_password_hash, check_password_hash  # 密码哈希与验证

# 项目内部模块导入
from api.db import UserTenantRole  # 用户-租户角色枚举
from api.db.db_models import DB, UserTenant  # 数据库连接代理、用户-租户关联模型
from api.db.db_models import User, Tenant  # 用户模型、租户模型
from api.db.services.common_service import CommonService  # 通用服务基类
from common.misc_utils import get_uuid  # UUID 生成工具
from common.time_utils import current_timestamp, datetime_format  # 时间戳和日期格式化工具
from common.constants import StatusEnum  # 状态枚举
from common import settings  # 全局配置


class UserService(CommonService):
    """用户服务类 —— 管理用户相关的数据库操作。

    继承自 CommonService，提供用户管理的专业功能，包括：
    - 用户认证（邮箱 + 密码登录）
    - 用户 CRUD（创建、查询、更新、删除）
    - 访问令牌（access_token）安全校验
    - 管理员权限判断

    Attributes:
        model: 绑定的 Peewee ORM 模型类 —— User 表。
    """
    model = User

    @classmethod
    @DB.connection_context()
    def query(cls, cols=None, reverse=None, order_by=None, **kwargs):
        """查询用户 —— 带 access_token 安全校验。

        对 access_token 参数做了三层安全过滤，防止恶意查询：
        1. 拒绝空值 / 纯空白字符串
        2. 拒绝长度不足 32 字符（合法 UUID 至少 32 位）
        3. 拒绝以 "INVALID_" 开头的已注销 token

        安全校验通过后，委托父类 CommonService.query() 执行实际查询。

        Returns:
            合法的 Peewee SelectQuery 对象；非法 token 返回一个必然为空的查询。
        """
        if 'access_token' in kwargs:
            access_token = kwargs['access_token']

            # 拒绝空值、None 或纯空白字符串的 access_token
            if not access_token or not str(access_token).strip():
                logging.warning("UserService.query: Rejecting empty access_token query")
                return cls.model.select().where(cls.model.id == "INVALID_EMPTY_TOKEN")  # 返回空结果

            # 拒绝长度不足的 token（合法访问令牌应为 UUID，至少 32 字符）
            if len(str(access_token).strip()) < 32:
                logging.warning(f"UserService.query: Rejecting short access_token query: {len(str(access_token))} chars")
                return cls.model.select().where(cls.model.id == "INVALID_SHORT_TOKEN")  # 返回空结果

            # 拒绝已失效的 token（以 "INVALID_" 开头，来自登出操作）
            if str(access_token).startswith("INVALID_"):
                logging.warning("UserService.query: Rejecting invalidated access_token")
                return cls.model.select().where(cls.model.id == "INVALID_LOGOUT_TOKEN")  # 返回空结果

        # 安全校验通过，调用父类方法执行实际查询
        return super().query(cols=cols, reverse=reverse, order_by=order_by, **kwargs)

    @classmethod
    @DB.connection_context()
    def filter_by_id(cls, user_id):
        """根据用户 ID 查询单个用户。

        Args:
            user_id: 用户的唯一标识符。

        Returns:
            找到则返回 User 对象，否则返回 None。
        """
        try:
            user = cls.model.select().where(cls.model.id == user_id).get()
            return user
        except peewee.DoesNotExist:
            return None

    @classmethod
    @DB.connection_context()
    def query_user(cls, email, password):
        """通过邮箱和密码认证用户（登录验证）。

        先按邮箱和有效状态查找用户，再使用 werkzeug 的密码哈希校验
        对比明文密码与数据库中存储的哈希值。

        Args:
            email: 用户注册邮箱。
            password: 明文密码。

        Returns:
            认证成功返回 User 对象，失败返回 None。
        """
        user = cls.model.select().where((cls.model.email == email),
                                        (cls.model.status == StatusEnum.VALID.value)).first()
        if user and check_password_hash(str(user.password), password):
            return user
        else:
            return None

    @classmethod
    @DB.connection_context()
    def query_user_by_email(cls, email):
        """根据邮箱查询所有匹配的用户。

        Args:
            email: 要查询的邮箱地址。

        Returns:
            匹配的用户列表（可能包含多个或空列表）。
        """
        users = cls.model.select().where((cls.model.email == email))
        return list(users)

    @classmethod
    @DB.connection_context()
    def save(cls, **kwargs):
        """创建新用户。

        自动处理：
        - 生成 UUID 作为用户 ID（如果未提供）
        - 对明文密码进行哈希处理
        - 填充创建时间和更新时间戳

        Args:
            **kwargs: 用户属性字典，可包含 nickname, email, password 等字段。

        Returns:
            新创建的 User 对象。
        """
        # 未提供 ID 时自动生成 UUID
        if "id" not in kwargs:
            kwargs["id"] = get_uuid()
        # 对密码进行哈希加密存储
        if "password" in kwargs:
            kwargs["password"] = generate_password_hash(
                str(kwargs["password"]))

        # 设置创建时间和更新时间
        current_ts = current_timestamp()
        current_date = datetime_format(datetime.now())

        kwargs["create_time"] = current_ts
        kwargs["create_date"] = current_date
        kwargs["update_time"] = current_ts
        kwargs["update_date"] = current_date
        # force_insert=True 强制执行 INSERT 而非 UPDATE
        obj = cls.model(**kwargs).save(force_insert=True)
        return obj

    @classmethod
    @DB.connection_context()
    def delete_user(cls, user_ids, update_user_dict):
        """软删除用户 —— 将状态标记为 0。

        使用数据库事务（atomic）保证操作的原子性。
        注意：此方法名为"删除"，实际是逻辑删除（status=0），不会物理删除记录。

        Args:
            user_ids: 要删除的用户 ID 列表。
            update_user_dict: 额外的更新字段（当前方法未使用该参数）。
        """
        with DB.atomic():
            cls.model.update({"status": 0}).where(
                cls.model.id.in_(user_ids)).execute()

    @classmethod
    @DB.connection_context()
    def update_user(cls, user_id, user_dict):
        """更新用户信息。

        在事务中更新指定用户的字段，并自动刷新 update_time 和 update_date。

        Args:
            user_id: 目标用户的唯一标识符。
            user_dict: 要更新的字段字典（如 nickname, email 等）。
        """
        with DB.atomic():
            if user_dict:
                user_dict["update_time"] = current_timestamp()
                user_dict["update_date"] = datetime_format(datetime.now())
                cls.model.update(user_dict).where(
                    cls.model.id == user_id).execute()

    @classmethod
    @DB.connection_context()
    def update_user_password(cls, user_id, new_password):
        """更新用户密码。

        对新密码进行哈希处理后，在事务中更新密码和修改时间。

        Args:
            user_id: 目标用户的唯一标识符。
            new_password: 新的明文密码。
        """
        with DB.atomic():
            update_dict = {
                "password": generate_password_hash(str(new_password)),
                "update_time": current_timestamp(),
                "update_date": datetime_format(datetime.now())
            }
            cls.model.update(update_dict).where(cls.model.id == user_id).execute()

    @classmethod
    @DB.connection_context()
    def is_admin(cls, user_id):
        """判断指定用户是否为超级管理员。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            bool —— 如果 is_superuser == 1 则返回 True，否则 False。
        """
        return cls.model.select().where(
            cls.model.id == user_id,
            cls.model.is_superuser == 1).count() > 0

    @classmethod
    @DB.connection_context()
    def get_all_users(cls):
        """获取所有用户列表，按邮箱排序。

        Returns:
            所有用户的列表。
        """
        users = cls.model.select().order_by(cls.model.email)
        return list(users)


class TenantService(CommonService):
    """租户服务类 —— 管理租户相关的数据库操作。

    继承自 CommonService，提供租户管理功能，包括：
    - 查询用户拥有/加入的租户信息
    - 租户积分扣减
    - MinIO 网关路由计算（基于 tenant_id 哈希）
    - 模型配置缺失的租户排查

    Attributes:
        model: 绑定的 Peewee ORM 模型类 —— Tenant 表。
    """
    model = Tenant

    @classmethod
    @DB.connection_context()
    def get_info_by(cls, user_id):
        """获取用户作为 OWNER（拥有者）的租户详细信息。

        通过 JOIN UserTenant 表，查询用户在有效状态下以 OWNER
        角色关联的租户，返回租户的完整模型配置信息。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            租户信息字典列表，包含租户 ID、名称、各类模型 ID（LLM/Embedding/
            Rerank/ASR/图片转文本/TTS/OCR/Parser）及用户角色。
        """
        fields = [
            cls.model.id.alias("tenant_id"),
            cls.model.name,
            cls.model.llm_id,
            cls.model.embd_id,
            cls.model.rerank_id,
            cls.model.asr_id,
            cls.model.img2txt_id,
            cls.model.tts_id,
            cls.model.ocr_id,
            cls.model.parser_ids,
            UserTenant.role]
        return list(cls.model.select(*fields)
                    .join(UserTenant, on=((cls.model.id == UserTenant.tenant_id) & (UserTenant.user_id == user_id) & (UserTenant.status == StatusEnum.VALID.value) & (UserTenant.role == UserTenantRole.OWNER)))
                    .where(cls.model.status == StatusEnum.VALID.value).dicts())

    @classmethod
    @DB.connection_context()
    def get_joined_tenants_by_user_id(cls, user_id):
        """获取用户作为 NORMAL 成员加入的租户列表。

        与 get_info_by 的区别：此方法查询角色为 NORMAL（普通成员）
        的租户，用于获取用户被邀请加入的团队/工作空间。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            租户信息字典列表，包含租户 ID、名称、LLM/Embedding/ASR/
            图片转文本模型 ID 及用户角色。
        """
        fields = [
            cls.model.id.alias("tenant_id"),
            cls.model.name,
            cls.model.llm_id,
            cls.model.embd_id,
            cls.model.asr_id,
            cls.model.img2txt_id,
            UserTenant.role]
        return list(cls.model.select(*fields)
                    .join(UserTenant, on=((cls.model.id == UserTenant.tenant_id) & (UserTenant.user_id == user_id) & (UserTenant.status == StatusEnum.VALID.value) & (UserTenant.role == UserTenantRole.NORMAL)))
                    .where(cls.model.status == StatusEnum.VALID.value).dicts())

    @classmethod
    @DB.connection_context()
    def decrease(cls, user_id, num):
        """扣减租户的可用积分（credit）。

        使用 Peewee 的原子更新表达式 `credit = credit - num`，
        避免并发场景下的读写竞争问题。

        Args:
            user_id: 租户 ID。
            num: 要扣减的积分数量。

        Raises:
            LookupError: 如果受影响行数为 0（租户不存在）。
        """
        num = cls.model.update(credit=cls.model.credit - num).where(
            cls.model.id == user_id).execute()
        if num == 0:
            raise LookupError("Tenant not found which is supposed to be there")

    @classmethod
    @DB.connection_context()
    def user_gateway(cls, tenant_id):
        """计算租户对应的 MinIO 存储网关索引。

        通过对 tenant_id 做 SHA256 哈希，取模后分配到配置的
        MinIO 实例之一，实现多 MinIO 实例间的负载均衡。

        Args:
            tenant_id: 租户唯一标识符。

        Returns:
            int —— MinIO 实例的索引（0 到 len(settings.MINIO)-1）。
        """
        hash_obj = hashlib.sha256(tenant_id.encode("utf-8"))
        return int(hash_obj.hexdigest(), 16)%len(settings.MINIO)

    @classmethod
    @DB.connection_context()
    def get_null_tenant_model_id_rows(cls):
        """查询任意模型 ID 字段为 NULL 的租户记录。

        用于排查配置不完整的租户 —— 当 LLM、Embedding、ASR、TTS、
        Rerank 或图片转文本模型 ID 任一为空时，该租户被返回。

        Returns:
            模型配置不完整的租户对象列表。
        """
        objs = cls.model.select().orwhere(cls.model.tenant_llm_id.is_null(), cls.model.tenant_embd_id.is_null(), cls.model.tenant_asr_id.is_null(), cls.model.tenant_tts_id.is_null(), cls.model.tenant_rerank_id.is_null(), cls.model.tenant_img2txt_id.is_null())
        return list(objs)


class UserTenantService(CommonService):
    """用户-租户关联服务类 —— 管理用户与租户的多对多关系。

    继承自 CommonService，处理用户在不同租户中的成员身份和角色管理。
    UserTenant 是 User 和 Tenant 之间的中间表，每条记录代表一个用户
    在某个租户中的成员关系及其角色（OWNER / NORMAL）。

    Attributes:
        model: 绑定的 Peewee ORM 模型类 —— UserTenant 表。
    """
    model = UserTenant

    @classmethod
    @DB.connection_context()
    def filter_by_id(cls, user_tenant_id):
        """根据关联记录 ID 查询有效的用户-租户关系。

        Args:
            user_tenant_id: UserTenant 记录的唯一标识符。

        Returns:
            找到则返回 UserTenant 对象，否则返回 None。
        """
        try:
            user_tenant = cls.model.select().where((cls.model.id == user_tenant_id) & (cls.model.status == StatusEnum.VALID.value)).get()
            return user_tenant
        except peewee.DoesNotExist:
            return None

    @classmethod
    @DB.connection_context()
    def save(cls, **kwargs):
        """创建用户-租户关联记录。

        自动生成 UUID 作为记录 ID（如果未提供），然后强制插入新行。

        Args:
            **kwargs: 关联记录属性，需包含 user_id, tenant_id, role 等字段。

        Returns:
            新创建的 UserTenant 对象。
        """
        if "id" not in kwargs:
            kwargs["id"] = get_uuid()
        obj = cls.model(**kwargs).save(force_insert=True)
        return obj

    @classmethod
    @DB.connection_context()
    def get_by_tenant_id(cls, tenant_id):
        """获取指定租户下的所有成员列表（不含 OWNER）。

        JOIN User 表获取成员详细信息（昵称、邮箱、头像、认证状态等），
        排除角色为 OWNER 的用户（OWNER 由 get_info_by 单独查询）。

        Args:
            tenant_id: 租户唯一标识符。

        Returns:
            成员信息字典列表，包含用户 ID、角色、昵称、邮箱、头像、
            认证状态、激活状态、是否为超级管理员等字段。
        """
        fields = [
            cls.model.id,
            cls.model.user_id,
            cls.model.status,
            cls.model.role,
            User.nickname,
            User.email,
            User.avatar,
            User.is_authenticated,
            User.is_active,
            User.is_anonymous,
            User.status,
            User.update_date,
            User.is_superuser]
        return list(cls.model.select(*fields)
                    .join(User, on=((cls.model.user_id == User.id) & (cls.model.status == StatusEnum.VALID.value) & (cls.model.role != UserTenantRole.OWNER)))
                    .where(cls.model.tenant_id == tenant_id)
                    .dicts())

    @classmethod
    @DB.connection_context()
    def get_tenants_by_user_id(cls, user_id):
        """获取指定用户所属的所有租户列表。

        通过 JOIN User 表，按 user_id 查询该用户所有有效关联的租户。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            租户信息字典列表，包含租户 ID、用户角色、昵称、邮箱、
            头像和更新时间。
        """
        fields = [
            cls.model.tenant_id,
            cls.model.role,
            User.nickname,
            User.email,
            User.avatar,
            User.update_date
        ]
        return list(cls.model.select(*fields)
                    .join(User, on=((cls.model.tenant_id == User.id) & (UserTenant.user_id == user_id) & (UserTenant.status == StatusEnum.VALID.value)))
                    .where(cls.model.status == StatusEnum.VALID.value).dicts())

    @classmethod
    @DB.connection_context()
    def get_user_tenant_relation_by_user_id(cls, user_id):
        """获取用户的所有用户-租户关联记录。

        仅返回 UserTenant 表自身的字段，不 JOIN 其他表。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            关联记录字典列表，包含记录 ID、用户 ID、租户 ID 和角色。
        """
        fields = [
            cls.model.id,
            cls.model.user_id,
            cls.model.tenant_id,
            cls.model.role
        ]
        return list(cls.model.select(*fields).where(cls.model.user_id == user_id).dicts().dicts())

    @classmethod
    @DB.connection_context()
    def get_num_members(cls, user_id: str):
        """统计指定租户的成员数量。

        使用 COUNT 聚合函数统计该租户下所有关联记录的数量。

        Args:
            user_id: 租户 ID（参数名为 user_id，实际语义是 tenant_id）。

        Returns:
            int —— 该租户的成员总数。
        """
        cnt_members = cls.model.select(peewee.fn.COUNT(cls.model.id)).where(cls.model.tenant_id == user_id).scalar()
        return cnt_members

    @classmethod
    @DB.connection_context()
    def filter_by_tenant_and_user_id(cls, tenant_id, user_id):
        """根据租户 ID 和用户 ID 联合查询关联记录。

        用于判断某个用户是否属于某个租户，常用于权限校验场景。

        Args:
            tenant_id: 租户唯一标识符。
            user_id: 用户唯一标识符。

        Returns:
            找到则返回 UserTenant 对象，否则返回 None。
        """
        try:
            user_tenant = cls.model.select().where(
                (cls.model.tenant_id == tenant_id) & (cls.model.status == StatusEnum.VALID.value) &
                (cls.model.user_id == user_id)
            ).first()
            return user_tenant
        except peewee.DoesNotExist:
            return None
