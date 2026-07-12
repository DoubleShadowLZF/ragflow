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
通用数据库服务基类。

本模块提供所有服务类的基类 ``CommonService``，封装了基于 Peewee ORM 的标准 CRUD
操作和通用数据库查询模式。所有业务服务类（如 ``ConnectorService``、``DocumentService``
等）均继承自此基类。

同时提供数据库操作的重试机制：

- ``retry_db_operation`` — 基于 tenacity 库的指数退避重试装饰器。
- ``retry_deadlock_operation`` — 专门处理 MySQL/OceanBase 死锁（错误码 1213）的重试。
- ``_is_deadlock_error`` — 判断 OperationalError 是否为死锁错误。
"""
import logging
import time
from datetime import datetime
from functools import wraps

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import peewee
from peewee import InterfaceError, OperationalError

from api.db.db_models import DB
from common.misc_utils import get_uuid
from common.time_utils import current_timestamp, datetime_format


def _is_deadlock_error(exc: OperationalError) -> bool:
    """判断 OperationalError 是否为 MySQL/OceanBase 死锁错误（错误码 1213）。"""
    return isinstance(exc, OperationalError) and bool(getattr(exc, "args", ())) and exc.args[0] == 1213


def retry_deadlock_operation(max_retries=3, retry_delay=0.1):
    """当 MySQL/OceanBase 因死锁中止操作时，自动重试整个数据库操作。

    重试延迟按指数增长（2^n），最多重试 ``max_retries`` 次。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    if not _is_deadlock_error(e) or attempt >= max_retries - 1:
                        raise
                    current_delay = retry_delay * (2**attempt)
                    logging.warning(
                        "%s failed due to DB deadlock, retrying (%s/%s): %s",
                        func.__qualname__,
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    time.sleep(current_delay)

        return wrapper

    return decorator


def retry_db_operation(func):
    """数据库操作重试装饰器。

    基于 tenacity 库，在遇到 InterfaceError 或 OperationalError 时
    自动重试，最多 3 次，采用指数退避策略（1s ~ 5s）。
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((InterfaceError, OperationalError)),
        before_sleep=lambda retry_state: print(f"RETRY {retry_state.attempt_number} TIMES"),
        reraise=True,
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class CommonService:
    """通用服务基类 — 提供标准的 CRUD 操作和通用数据库查询模式。

    所有业务服务类均继承自此基类，使用 Peewee ORM 与数据库交互。
    子类须设置 ``model`` 属性以指定操作的 Peewee 模型。

    Attributes:
        model: 该服务所操作的 Peewee 模型类，子类必须覆盖此属性。
    """

    model = None

    @classmethod
    @DB.connection_context()
    def query(cls, cols=None, reverse=None, order_by=None, **kwargs):
        """执行数据库查询，支持列选择与排序。

        Args:
            cols: 要选择的列名列表，为 None 则选择全部列。
            reverse: True 降序，False 升序。
            order_by: 排序字段名。
            **kwargs: 额外的过滤条件。

        Returns:
            peewee.ModelSelect: 匹配记录的查询结果。
        """
        return cls.model.query(cols=cols, reverse=reverse, order_by=order_by, **kwargs)

    @classmethod
    @DB.connection_context()
    def get_all(cls, cols=None, reverse=None, order_by=None):
        """获取表中全部记录，支持列选择和排序。

        若未指定 order_by 但指定了 reverse，则默认按 create_time 排序。

        Args:
            cols: 要选择的列名列表，为 None 则选择全部列。
            reverse: True 降序，False 升序。
            order_by: 排序字段名，未指定时默认为 create_time。

        Returns:
            peewee.ModelSelect: 包含全部匹配记录的查询。
        """
        if cols:
            query_records = cls.model.select(*cols)
        else:
            query_records = cls.model.select()
        if reverse is not None:
            if not order_by or not hasattr(cls.model, order_by):
                order_by = "create_time"
            if reverse is True:
                query_records = query_records.order_by(cls.model.getter_by(order_by).desc())
            elif reverse is False:
                query_records = query_records.order_by(cls.model.getter_by(order_by).asc())
        return query_records

    @classmethod
    @DB.connection_context()
    def get(cls, **kwargs):
        """获取匹配条件的单条记录。

        Args:
            **kwargs: 过滤条件。

        Returns:
            Model instance: 匹配的单条记录。

        Raises:
            peewee.DoesNotExist: 未找到匹配记录时抛出。
        """
        return cls.model.get(**kwargs)

    @classmethod
    @DB.connection_context()
    def get_or_none(cls, **kwargs):
        """获取匹配条件的单条记录，未找到时返回 None 而不抛出异常。

        Args:
            **kwargs: 过滤条件。

        Returns:
            Model instance or None: 匹配记录或 None。
        """
        try:
            return cls.model.get(**kwargs)
        except peewee.DoesNotExist:
            return None

    @classmethod
    @DB.connection_context()
    def save(cls, **kwargs):
        """保存新记录到数据库（强制 INSERT 而非 UPDATE）。

        Args:
            **kwargs: 记录字段值。

        Returns:
            Model instance: 创建后的记录对象。
        """
        sample_obj = cls.model(**kwargs).save(force_insert=True)
        return sample_obj

    @classmethod
    @DB.connection_context()
    def insert(cls, **kwargs):
        """插入新记录，自动生成 ID 及 create_time/create_date/update_time/update_date 时间戳。

        Args:
            **kwargs: 记录字段值。

        Returns:
            Model instance: 新创建的记录对象。
        """
        if "id" not in kwargs:
            kwargs["id"] = get_uuid()
        timestamp = current_timestamp()
        cur_datetime = datetime_format(datetime.now())
        kwargs["create_time"] = timestamp
        kwargs["create_date"] = cur_datetime
        kwargs["update_time"] = timestamp
        kwargs["update_date"] = cur_datetime
        sample_obj = cls.model(**kwargs).save(force_insert=True)
        return sample_obj

    @classmethod
    @DB.connection_context()
    def insert_many(cls, data_list, batch_size=100):
        """批量插入多条记录，自动设置创建时间戳。

        在事务中以 ``batch_size`` 为一批次分批执行。

        Args:
            data_list: 包含待插入记录数据的字典列表。
            batch_size: 每批次插入的记录数，默认 100。
        """
        current_ts = current_timestamp()
        current_datetime = datetime_format(datetime.now())
        with DB.atomic():
            for d in data_list:
                d["create_time"] = current_ts
                d["create_date"] = current_datetime
                d["update_time"] = current_ts
                d["update_date"] = current_datetime

            for i in range(0, len(data_list), batch_size):
                cls.model.insert_many(data_list[i : i + batch_size]).execute()

    @classmethod
    @DB.connection_context()
    def update_many_by_id(cls, data_list):
        """根据 ID 批量更新多条记录，自动刷新 update_time 和 update_date。

        Args:
            data_list: 包含待更新数据的字典列表，每条字典必须包含 'id' 字段。
        """

        timestamp = current_timestamp()
        cur_datetime = datetime_format(datetime.now())
        for data in data_list:
            data["update_time"] = timestamp
            data["update_date"] = cur_datetime
        with DB.atomic():
            for data in data_list:
                cls.model.update(data).where(cls.model.id == data["id"]).execute()

    @classmethod
    @DB.connection_context()
    @retry_db_operation
    def update_by_id(cls, pid, data):
        """根据 ID 更新单条记录，自动刷新 update_time 和 update_date。

        Args:
            pid: 记录 ID。
            data: 待更新的字段值。

        Returns:
            更新的记录数。
        """
        data["update_time"] = current_timestamp()
        data["update_date"] = datetime_format(datetime.now())
        num = cls.model.update(data).where(cls.model.id == pid).execute()
        return num

    @classmethod
    @DB.connection_context()
    def get_by_id(cls, pid):
        """根据 ID 获取单条记录。

        Args:
            pid: 记录 ID。

        Returns:
            (success, record) 元组。
        """
        try:
            obj = cls.model.get_or_none(cls.model.id == pid)
            if obj:
                return True, obj
        except Exception:
            pass
        return False, None

    @classmethod
    @DB.connection_context()
    def get_by_ids(cls, pids, cols=None):
        """根据 ID 列表批量获取记录。

        Args:
            pids: 记录 ID 列表。
            cols: 要选择的列名列表。

        Returns:
            匹配记录的查询对象。
        """
        if cols:
            objs = cls.model.select(*cols)
        else:
            objs = cls.model.select()
        return objs.where(cls.model.id.in_(pids))

    @classmethod
    @DB.connection_context()
    def delete_by_id(cls, pid):
        """根据 ID 删除单条记录。

        Args:
            pid: 记录 ID。

        Returns:
            删除的记录数。
        """
        return cls.model.delete().where(cls.model.id == pid).execute()

    @classmethod
    @DB.connection_context()
    def delete_by_ids(cls, pids):
        """根据 ID 列表批量删除记录，在事务中执行。

        Args:
            pids: 记录 ID 列表。

        Returns:
            删除的记录数。
        """
        with DB.atomic():
            res = cls.model.delete().where(cls.model.id.in_(pids)).execute()
            return res

    @classmethod
    @DB.connection_context()
    def filter_delete(cls, filters):
        """根据过滤条件删除记录，在事务中执行。

        Args:
            filters: Peewee 过滤条件列表。

        Returns:
            删除的记录数。
        """
        with DB.atomic():
            num = cls.model.delete().where(*filters).execute()
            return num

    @classmethod
    @DB.connection_context()
    def filter_update(cls, filters, update_data):
        """根据过滤条件更新记录，在事务中执行。

        Args:
            filters: Peewee 过滤条件列表。
            update_data: 待更新的字段值。

        Returns:
            更新的记录数。
        """
        with DB.atomic():
            return cls.model.update(update_data).where(*filters).execute()

    @staticmethod
    def cut_list(tar_list, n):
        """将列表按指定大小切分为多个元组。

        Args:
            tar_list: 待切分的列表。
            n: 每个分块的大小。

        Returns:
            包含各分块元组的列表。
        """
        length = len(tar_list)
        arr = range(length)
        result = [tuple(tar_list[x : (x + n)]) for x in arr[::n]]
        return result

    @classmethod
    @DB.connection_context()
    def filter_scope_list(cls, in_key, in_filters_list, filters=None, cols=None):
        """使用 IN 子句批量查询记录，支持可选列选择和额外过滤条件。

        为避免单个 IN 子句过长，内部自动按每 20 个值进行分片查询。

        Args:
            in_key: IN 子句的字段名。
            in_filters_list: IN 子句的值列表。
            filters: 额外的 Peewee 过滤条件。
            cols: 要选择的列名列表。

        Returns:
            匹配记录的列表。
        """
        in_filters_tuple_list = cls.cut_list(in_filters_list, 20)
        if not filters:
            filters = []
        res_list = []
        if cols:
            for i in in_filters_tuple_list:
                query_records = cls.model.select(*cols).where(getattr(cls.model, in_key).in_(i), *filters)
                if query_records:
                    res_list.extend([query_record for query_record in query_records])
        else:
            for i in in_filters_tuple_list:
                query_records = cls.model.select().where(getattr(cls.model, in_key).in_(i), *filters)
                if query_records:
                    res_list.extend([query_record for query_record in query_records])
        return res_list
