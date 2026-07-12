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
聊天渠道（Chat Channel）持久化服务

封装 ChatChannel 模型的 CRUD 操作，并提供权限校验逻辑。
聊天渠道用于将 RAGFlow 对话助手对接到外部即时通讯平台（如 Telegram、Discord 等）。
"""

import logging

from peewee import JOIN

from api.db.db_models import DB, ChatChannel, Dialog
from api.db.services.common_service import CommonService

LOGGER = logging.getLogger(__name__)


class ChatChannelService(CommonService):
    """聊天渠道服务层，继承 CommonService 提供基础的增删改查能力。"""

    model = ChatChannel

    @classmethod
    @DB.connection_context()
    def list(cls, tenant_id):
        """查询租户的所有聊天渠道，LEFT JOIN 关联的 Dialog 获取对话名称。

        Returns:
            list[dict]: 渠道列表（不含敏感凭证），按创建时间倒序排列
        """
        fields = [
            cls.model.id,
            cls.model.name,
            cls.model.channel,
            cls.model.dialog_id,
            cls.model.status,
            Dialog.name.alias("dialog_name"),
        ]
        return list(
            cls.model.select(*fields)
            .join(
                Dialog,
                join_type=JOIN.LEFT_OUTER,
                on=(Dialog.id == cls.model.dialog_id),
            )
            .where(cls.model.tenant_id == tenant_id)
            .order_by(cls.model.create_time.desc())
            .dicts()
        )

    @classmethod
    @DB.connection_context()
    def list_active(cls):
        """获取所有启用状态（status="1"）的聊天渠道，跨租户查询。

        返回完整的渠道记录（包含 config 中的凭证信息），供后台任务轮询用。
        """
        return list(cls.model.select().where(cls.model.status == "1"))

    @classmethod
    @DB.connection_context()
    def accessible(cls, channel_id: str, user_id: str) -> bool:
        """检查用户是否有权访问指定聊天渠道。

        访问判定规则：
        1. 渠道不存在 → 拒绝，记录警告日志
        2. 用户的 ID 等于渠道的 tenant_id（渠道所有者） → 允许
        3. 用户已加入渠道所属的租户（通过 TenantService 查询） → 允许
        4. 以上条件都不满足 → 拒绝

        Returns:
            bool: 是否允许访问
        """
        e, channel = cls.get_by_id(channel_id)
        if not e:
            LOGGER.warning("chat channel access denied: not found channel_id=%s user_id=%s", channel_id, user_id)
            return False

        # 渠道所有者直接允许
        if channel.tenant_id == user_id:
            return True

        # 检查用户是否已加入该渠道所属的租户
        from api.db.services.user_service import TenantService

        joined_tenants = TenantService.get_joined_tenants_by_user_id(user_id)
        has_access = any(tenant["tenant_id"] == channel.tenant_id for tenant in joined_tenants)
        if not has_access:
            LOGGER.warning(
                "chat channel access denied: tenant mismatch channel_id=%s user_id=%s tenant_id=%s",
                channel_id,
                user_id,
                channel.tenant_id,
            )
        return has_access
