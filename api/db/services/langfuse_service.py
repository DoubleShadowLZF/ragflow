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

"""
Langfuse API 密钥持久化服务

封装 TenantLangfuse 模型（Peewee ORM）的 CRUD 操作，供 langfuse_api.py 调用。
每个租戶（tenant）最多配置一组 Langfuse 密钥（secret_key + public_key + host）。
"""

from datetime import datetime

import peewee

from api.db.db_models import DB, TenantLangfuse
from api.db.services.common_service import CommonService
from common.time_utils import current_timestamp, datetime_format


class TenantLangfuseService(CommonService):
    """
    租户 Langfuse 密钥服务层。

    注意：所有修改状态的操作应在 DB.atomic() 上下文内调用，确保事务原子性。
    """

    model = TenantLangfuse

    @classmethod
    @DB.connection_context()
    def filter_by_tenant(cls, tenant_id):
        """按租户 ID 查询 Langfuse 密钥记录，返回 Peewee Model 实例或 None。"""
        fields = [cls.model.tenant_id, cls.model.host, cls.model.secret_key, cls.model.public_key]
        try:
            keys = cls.model.select(*fields).where(cls.model.tenant_id == tenant_id).first()
            return keys
        except peewee.DoesNotExist:
            return None

    @classmethod
    @DB.connection_context()
    def filter_by_tenant_with_info(cls, tenant_id):
        """按租户 ID 查询 Langfuse 密钥记录，返回 dict 格式（含字段名），方便直接序列化为 JSON。"""
        fields = [cls.model.tenant_id, cls.model.host, cls.model.secret_key, cls.model.public_key]
        try:
            keys = cls.model.select(*fields).where(cls.model.tenant_id == tenant_id).dicts().first()
            return keys
        except peewee.DoesNotExist:
            return None

    @classmethod
    @DB.connection_context()
    def delete_ty_tenant_id(cls, tenant_id):
        """按租户 ID 删除 Langfuse 密钥记录。"""
        return cls.model.delete().where(cls.model.tenant_id == tenant_id).execute()

    @classmethod
    def update_by_tenant(cls, tenant_id, langfuse_keys):
        """更新指定租户的 Langfuse 密钥，自动填充 update_time 和 update_date。"""
        langfuse_keys["update_time"] = current_timestamp()
        langfuse_keys["update_date"] = datetime_format(datetime.now())
        return cls.model.update(**langfuse_keys).where(cls.model.tenant_id == tenant_id).execute()

    @classmethod
    def save(cls, **kwargs):
        """新建 Langfuse 密钥记录，自动填充 create_time/update_time。"""
        current_ts = current_timestamp()
        current_date = datetime_format(datetime.now())

        kwargs["create_time"] = current_ts
        kwargs["create_date"] = current_date
        kwargs["update_time"] = current_ts
        kwargs["update_date"] = current_date
        obj = cls.model.create(**kwargs)
        return obj

    @classmethod
    def delete_model(cls, langfuse_model):
        """删除指定 Langfuse Model 实例。"""
        langfuse_model.delete_instance()
