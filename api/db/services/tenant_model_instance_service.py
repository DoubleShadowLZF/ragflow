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
租户模型实例服务 —— 管理 TenantModelInstance 表的数据库操作。

TenantModelInstance（模型实例）是 Provider → Instance → Model 三层模型
中的中间层，每条记录代表租户在某供应商下用特定 API Key 创建的一个具体配置。
一个供应商可以有多个实例（如不同的区域或不同的 API 端点）。
"""

from common.misc_utils import get_uuid
from api.db.db_models import DB, TenantModelInstance
from api.db.services.common_service import CommonService
from api.db.services import duplicate_name


class TenantModelInstanceService(CommonService):
    """租户模型实例服务类。

    继承自 CommonService，提供租户模型实例（Provider Instance）的增删查操作。
    Provider Instance 存储了供应商实例的 API Key、Base URL 等连接信息。

    Attributes:
        model: TenantModelInstance Peewee ORM 模型类。
    """
    model = TenantModelInstance

    @classmethod
    @DB.connection_context()
    def create_instance(cls, provider_id: str, instance_name: str, api_key: str, extra: str):
        """创建模型实例。

        自动处理实例名称去重 —— 如果同名实例已存在，会在名称后追加 "(1)"、
        "(2)" 等后缀，确保实例名称在同一个 provider 下唯一。

        Args:
            provider_id: 供应商 ID。
            instance_name: 实例名称（如 "production"、"staging"）。
            api_key: API 密钥字符串。
            extra: JSON 字符串，存储 base_url、region 等附加配置。

        Returns:
            新创建的 TenantModelInstance 对象。
        """
        # 名称去重后再插入
        unique_instance_name = duplicate_name(cls.query, name_field="instance_name", provider_id=provider_id, instance_name=instance_name)
        return cls.insert(id=get_uuid(), provider_id=provider_id, instance_name=unique_instance_name, api_key=api_key, extra=extra)

    @classmethod
    @DB.connection_context()
    def get_all_by_provider_id(cls, provider_id):
        """获取指定供应商下的所有实例列表。

        Args:
            provider_id: 供应商 ID。

        Returns:
            TenantModelInstance 对象列表。
        """
        return list(cls.model.select().where(cls.model.provider_id == provider_id))

    @classmethod
    @DB.connection_context()
    def get_by_provider_ids(cls, provider_ids):
        """批量获取多个供应商下的所有实例。

        Args:
            provider_ids: 供应商 ID 列表。

        Returns:
            TenantModelInstance 对象列表。
        """
        return list(cls.model.select().where(cls.model.provider_id.in_(provider_ids)))

    @classmethod
    @DB.connection_context()
    def get_by_provider_id_and_instance_name(cls, provider_id, instance_name):
        """按供应商 ID 和实例名称精确查询单个实例。

        Args:
            provider_id: 供应商 ID。
            instance_name: 实例名称。

        Returns:
            TenantModelInstance 对象，或 None。
        """
        return cls.model.get_or_none(
            cls.model.provider_id == provider_id,
            cls.model.instance_name == instance_name,
        )

    @classmethod
    @DB.connection_context()
    def get_by_provider_id_and_api_key(cls, provider_id, api_key):
        """按供应商 ID 和 API Key 查询实例（用于检查相同 Key 的重复实例）。

        Args:
            provider_id: 供应商 ID。
            api_key: API 密钥字符串。

        Returns:
            TenantModelInstance 对象，或 None。
        """
        return cls.model.get_or_none(
            cls.model.provider_id == provider_id,
            cls.model.api_key == api_key
        )

    @classmethod
    @DB.connection_context()
    def delete_by_provider_id_and_instance_name(cls, provider_id, instance_name):
        """按供应商 ID 和实例名称删除单个实例。

        Args:
            provider_id: 供应商 ID。
            instance_name: 实例名称。

        Returns:
            删除的记录数。
        """
        return cls.model.delete().where(
            cls.model.provider_id == provider_id,
            cls.model.instance_name == instance_name,
        ).execute()

    @classmethod
    @DB.connection_context()
    def delete_by_provider_ids(cls, provider_ids):
        """按供应商 ID 列表批量删除实例。

        Args:
            provider_ids: 供应商 ID 列表。

        Returns:
            删除的记录数。
        """
        return cls.model.delete().where(
            cls.model.provider_id.in_(provider_ids)
        ).execute()
