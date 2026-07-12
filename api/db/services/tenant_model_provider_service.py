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
租户模型供应商服务 —— 管理 TenantModelProvider 表的数据库操作。

TenantModelProvider 是 Provider → Instance → Model 三层模型中的顶层，
记录租户与供应商（如 OpenAI、DeepSeek、SiliconFlow）之间的关联关系。
一条记录代表"某租户启用了某供应商"。
"""

from api.db.db_models import DB, TenantModelProvider
from api.db.services.common_service import CommonService


class TenantModelProviderService(CommonService):
    """租户模型供应商服务类。

    继承自 CommonService，管理租户与 LLM 供应商之间的关联。
    租户必须先添加供应商，然后才能在该供应商下创建实例和配置模型。

    Attributes:
        model: TenantModelProvider Peewee ORM 模型类。
    """
    model = TenantModelProvider

    @classmethod
    @DB.connection_context()
    def get_by_tenant_id_and_provider_name(cls, tenant_id, provider_name):
        """按租户 ID 和供应商名精确查询关联记录。

        Args:
            tenant_id: 租户 ID。
            provider_name: 供应商名称（如 "OpenAI"、"DeepSeek"）。

        Returns:
            TenantModelProvider 对象，或 None。
        """
        return cls.model.get_or_none(
            cls.model.tenant_id == tenant_id,
            cls.model.provider_name == provider_name,
        )

    @classmethod
    @DB.connection_context()
    def get_by_tenant_id(cls, tenant_id):
        """获取租户启用的所有供应商列表。

        Args:
            tenant_id: 租户 ID。

        Returns:
            TenantModelProvider 对象列表。
        """
        return list(cls.model.select().where(cls.model.tenant_id == tenant_id))

    @classmethod
    @DB.connection_context()
    def delete_by_tenant_id(cls, tenant_id):
        """删除租户的所有供应商关联（级联删除用）。

        Args:
            tenant_id: 租户 ID。

        Returns:
            删除的记录数。
        """
        return cls.model.delete().where(cls.model.tenant_id == tenant_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_by_tenant_id_and_provider_name(cls, tenant_id, provider_name):
        """删除指定租户的指定供应商关联。

        Args:
            tenant_id: 租户 ID。
            provider_name: 供应商名称。

        Returns:
            删除的记录数。
        """
        return cls.model.delete().where(
            cls.model.tenant_id == tenant_id,
            cls.model.provider_name == provider_name,
        ).execute()

    @classmethod
    @DB.connection_context()
    def list_provider_names_by_tenant_id(cls, tenant_id):
        """获取租户已启用的供应商名称列表（仅返回名称，不含完整对象）。

        用于 provider_api_service.list_providers 等场景，高效获取名称列表。

        Args:
            tenant_id: 租户 ID。

        Returns:
            供应商名称字符串列表。
        """
        return [row.provider_name for row in cls.model.select(cls.model.provider_name).where(cls.model.tenant_id == tenant_id)]