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
import hashlib
import inspect
import logging
import operator
import os
import sys
import time
import typing
from datetime import datetime, timezone
from enum import Enum
from functools import wraps

from quart_auth import AuthUser
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer
from peewee import (
    fn,
    InterfaceError,
    OperationalError,
    ProgrammingError,
    BigIntegerField,
    BooleanField,
    CharField,
    CompositeKey,
    DateTimeField,
    Field,
    FloatField,
    IntegerField,
    Metadata,
    Model,
    TextField,
    PrimaryKeyField,
)
from playhouse.migrate import MySQLMigrator, PostgresqlMigrator, migrate
from playhouse.pool import PooledMySQLDatabase, PooledPostgresqlDatabase

from api import utils
from api.db import SerializedType
from api.utils.json_encode import json_dumps, json_loads
from api.utils.configs import deserialize_b64, serialize_b64

from common.time_utils import current_timestamp, timestamp_to_date, date_string_to_timestamp
from common.decorator import singleton
from common.constants import ParserType, MAXIMUM_TASK_PAGE_NUMBER
from common import settings


CONTINUOUS_FIELD_TYPE = {IntegerField, FloatField, DateTimeField}
AUTO_DATE_TIMESTAMP_FIELD_PREFIX = {"create", "start", "end", "update", "read_access", "write_access"}


class TextFieldType(Enum):
    MYSQL = "LONGTEXT"
    OCEANBASE = "LONGTEXT"
    POSTGRES = "TEXT"


class LongTextField(TextField):
    field_type = TextFieldType[settings.DATABASE_TYPE.upper()].value


class JSONField(LongTextField):
    default_value = {}

    def __init__(self, object_hook=None, object_pairs_hook=None, **kwargs):
        self._object_hook = object_hook
        self._object_pairs_hook = object_pairs_hook
        super().__init__(**kwargs)

    def db_value(self, value):
        if value is None:
            value = self.default_value
        return json_dumps(value)

    def python_value(self, value):
        if not value:
            return self.default_value
        return json_loads(value, object_hook=self._object_hook, object_pairs_hook=self._object_pairs_hook)


class ListField(JSONField):
    default_value = []


class SerializedField(LongTextField):
    def __init__(self, serialized_type=SerializedType.PICKLE, object_hook=None, object_pairs_hook=None, **kwargs):
        self._serialized_type = serialized_type
        self._object_hook = object_hook
        self._object_pairs_hook = object_pairs_hook
        super().__init__(**kwargs)

    def db_value(self, value):
        if self._serialized_type == SerializedType.PICKLE:
            return serialize_b64(value, to_str=True)
        elif self._serialized_type == SerializedType.JSON:
            if value is None:
                return None
            return json_dumps(value, with_type=True)
        else:
            raise ValueError(f"the serialized type {self._serialized_type} is not supported")

    def python_value(self, value):
        if self._serialized_type == SerializedType.PICKLE:
            return deserialize_b64(value)
        elif self._serialized_type == SerializedType.JSON:
            if value is None:
                return {}
            return json_loads(value, object_hook=self._object_hook, object_pairs_hook=self._object_pairs_hook)
        else:
            raise ValueError(f"the serialized type {self._serialized_type} is not supported")


def is_continuous_field(cls: typing.Type) -> bool:
    if cls in CONTINUOUS_FIELD_TYPE:
        return True
    for p in cls.__bases__:
        if p in CONTINUOUS_FIELD_TYPE:
            return True
        elif p is not Field and p is not object:
            if is_continuous_field(p):
                return True
    else:
        return False


def auto_date_timestamp_field():
    return {f"{f}_time" for f in AUTO_DATE_TIMESTAMP_FIELD_PREFIX}


def auto_date_timestamp_db_field():
    return {f"f_{f}_time" for f in AUTO_DATE_TIMESTAMP_FIELD_PREFIX}


def remove_field_name_prefix(field_name):
    return field_name[2:] if field_name.startswith("f_") else field_name


class BaseModel(Model):
    create_time = BigIntegerField(null=True, index=True)
    create_date = DateTimeField(null=True, index=True)
    update_time = BigIntegerField(null=True, index=True)
    update_date = DateTimeField(null=True, index=True)

    def to_json(self):
        # This function is obsolete
        return self.to_dict()

    def to_dict(self):
        return self.__dict__["__data__"]

    def to_human_model_dict(self, only_primary_with: list = None):
        model_dict = self.__dict__["__data__"]

        if not only_primary_with:
            return {remove_field_name_prefix(k): v for k, v in model_dict.items()}

        human_model_dict = {}
        for k in self._meta.primary_key.field_names:
            human_model_dict[remove_field_name_prefix(k)] = model_dict[k]
        for k in only_primary_with:
            human_model_dict[k] = model_dict[f"f_{k}"]
        return human_model_dict

    @property
    def meta(self) -> Metadata:
        return self._meta

    @classmethod
    def get_primary_keys_name(cls):
        return cls._meta.primary_key.field_names if isinstance(cls._meta.primary_key, CompositeKey) else [cls._meta.primary_key.name]

    @classmethod
    def getter_by(cls, attr):
        return operator.attrgetter(attr)(cls)

    @classmethod
    def query(cls, reverse=None, order_by=None, **kwargs):
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = "%s" % f_n
            if not hasattr(cls, attr_name) or f_v is None:
                continue
            if type(f_v) in {list, set}:
                f_v = list(f_v)
                if is_continuous_field(type(getattr(cls, attr_name))):
                    if len(f_v) == 2:
                        for i, v in enumerate(f_v):
                            if isinstance(v, str) and f_n in auto_date_timestamp_field():
                                # time type: %Y-%m-%d %H:%M:%S
                                f_v[i] = date_string_to_timestamp(v)
                        lt_value = f_v[0]
                        gt_value = f_v[1]
                        if lt_value is not None and gt_value is not None:
                            filters.append(cls.getter_by(attr_name).between(lt_value, gt_value))
                        elif lt_value is not None:
                            filters.append(operator.attrgetter(attr_name)(cls) >= lt_value)
                        elif gt_value is not None:
                            filters.append(operator.attrgetter(attr_name)(cls) <= gt_value)
                else:
                    filters.append(operator.attrgetter(attr_name)(cls) << f_v)
            else:
                filters.append(operator.attrgetter(attr_name)(cls) == f_v)
        if filters:
            query_records = cls.select().where(*filters)
            if reverse is not None:
                if not order_by or not hasattr(cls, f"{order_by}"):
                    order_by = "create_time"
                if reverse is True:
                    query_records = query_records.order_by(cls.getter_by(f"{order_by}").desc())
                elif reverse is False:
                    query_records = query_records.order_by(cls.getter_by(f"{order_by}").asc())
            return [query_record for query_record in query_records]
        else:
            return []

    @classmethod
    def insert(cls, __data=None, **insert):
        if isinstance(__data, dict) and __data:
            __data[cls._meta.combined["create_time"]] = current_timestamp()
        if insert:
            insert["create_time"] = current_timestamp()

        return super().insert(__data, **insert)

    # update and insert will call this method
    @classmethod
    def _normalize_data(cls, data, kwargs):
        normalized = super()._normalize_data(data, kwargs)
        if not normalized:
            return {}

        normalized[cls._meta.combined["update_time"]] = current_timestamp()

        for f_n in AUTO_DATE_TIMESTAMP_FIELD_PREFIX:
            if {f"{f_n}_time", f"{f_n}_date"}.issubset(cls._meta.combined.keys()) and cls._meta.combined[f"{f_n}_time"] in normalized and normalized[cls._meta.combined[f"{f_n}_time"]] is not None:
                normalized[cls._meta.combined[f"{f_n}_date"]] = timestamp_to_date(normalized[cls._meta.combined[f"{f_n}_time"]])

        return normalized


class JsonSerializedField(SerializedField):
    def __init__(self, object_hook=utils.from_dict_hook, object_pairs_hook=None, **kwargs):
        super(JsonSerializedField, self).__init__(serialized_type=SerializedType.JSON, object_hook=object_hook, object_pairs_hook=object_pairs_hook, **kwargs)


class RetryingPooledMySQLDatabase(PooledMySQLDatabase):
    def __init__(self, *args, **kwargs):
        self.max_retries = kwargs.pop("max_retries", 5)
        self.retry_delay = kwargs.pop("retry_delay", 1)
        super().__init__(*args, **kwargs)

    def execute_sql(self, sql, params=None, commit=True):
        for attempt in range(self.max_retries + 1):
            try:
                return super().execute_sql(sql, params, commit)
            except (OperationalError, InterfaceError) as e:
                error_codes = [2013, 2006]
                error_messages = ['', 'Lost connection']
                should_retry = (
                    (hasattr(e, 'args') and e.args and e.args[0] in error_codes) or
                    (str(e) in error_messages) or
                    (hasattr(e, '__class__') and e.__class__.__name__ == 'InterfaceError')
                )

                if should_retry and attempt < self.max_retries:
                    logging.warning(
                        f"Database connection issue (attempt {attempt+1}/{self.max_retries}): {e}"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logging.error(f"DB execution failure: {e}")
                    raise
        return None

    def _handle_connection_loss(self):
        # self.close_all()
        # self.connect()
        try:
            self.close()
        except Exception:
            pass
        try:
            self.connect()
        except Exception as e:
            logging.error(f"Failed to reconnect: {e}")
            time.sleep(0.1)
            try:
                self.connect()
            except Exception as e2:
                logging.error(f"Failed to reconnect on second attempt: {e2}")
                raise

    def begin(self):
        for attempt in range(self.max_retries + 1):
            try:
                return super().begin()
            except (OperationalError, InterfaceError) as e:
                error_codes = [2013, 2006]
                error_messages = ['', 'Lost connection']

                should_retry = (
                    (hasattr(e, 'args') and e.args and e.args[0] in error_codes) or
                    (str(e) in error_messages) or
                    (hasattr(e, '__class__') and e.__class__.__name__ == 'InterfaceError')
                )

                if should_retry and attempt < self.max_retries:
                    logging.warning(
                        f"Lost connection during transaction (attempt {attempt+1}/{self.max_retries})"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
        return None


class RetryingPooledPostgresqlDatabase(PooledPostgresqlDatabase):
    def __init__(self, *args, **kwargs):
        self.max_retries = kwargs.pop("max_retries", 5)
        self.retry_delay = kwargs.pop("retry_delay", 1)
        super().__init__(*args, **kwargs)

    def execute_sql(self, sql, params=None, commit=True):
        for attempt in range(self.max_retries + 1):
            try:
                return super().execute_sql(sql, params, commit)
            except (OperationalError, InterfaceError) as e:
                # PostgreSQL specific error codes
                # 57P01: admin_shutdown
                # 57P02: crash_shutdown
                # 57P03: cannot_connect_now
                # 08006: connection_failure
                # 08003: connection_does_not_exist
                # 08000: connection_exception
                error_messages = ['connection', 'server closed', 'connection refused',
                                'no connection to the server', 'terminating connection']

                should_retry = any(msg in str(e).lower() for msg in error_messages)

                if should_retry and attempt < self.max_retries:
                    logging.warning(
                        f"PostgreSQL connection issue (attempt {attempt+1}/{self.max_retries}): {e}"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logging.error(f"PostgreSQL execution failure: {e}")
                    raise
        return None

    def _handle_connection_loss(self):
        try:
            self.close()
        except Exception:
            pass
        try:
            self.connect()
        except Exception as e:
            logging.error(f"Failed to reconnect to PostgreSQL: {e}")
            time.sleep(0.1)
            try:
                self.connect()
            except Exception as e2:
                logging.error(f"Failed to reconnect to PostgreSQL on second attempt: {e2}")
                raise

    def begin(self):
        for attempt in range(self.max_retries + 1):
            try:
                return super().begin()
            except (OperationalError, InterfaceError) as e:
                error_messages = ['connection', 'server closed', 'connection refused',
                                'no connection to the server', 'terminating connection']

                should_retry = any(msg in str(e).lower() for msg in error_messages)

                if should_retry and attempt < self.max_retries:
                    logging.warning(
                        f"PostgreSQL connection lost during transaction (attempt {attempt+1}/{self.max_retries})"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
        return None


class RetryingPooledOceanBaseDatabase(PooledMySQLDatabase):
    """Pooled OceanBase database with retry mechanism.

    OceanBase is compatible with MySQL protocol, so we inherit from PooledMySQLDatabase.
    This class provides connection pooling and automatic retry for connection issues.
    """
    def __init__(self, *args, **kwargs):
        self.max_retries = kwargs.pop("max_retries", 5)
        self.retry_delay = kwargs.pop("retry_delay", 1)
        super().__init__(*args, **kwargs)

    def execute_sql(self, sql, params=None, commit=True):
        for attempt in range(self.max_retries + 1):
            try:
                return super().execute_sql(sql, params, commit)
            except (OperationalError, InterfaceError) as e:
                # OceanBase/MySQL specific error codes
                # 2013: Lost connection to MySQL server during query
                # 2006: MySQL server has gone away
                error_codes = [2013, 2006]
                error_messages = ['', 'Lost connection', 'gone away']

                should_retry = (
                    (hasattr(e, 'args') and e.args and e.args[0] in error_codes) or
                    any(msg in str(e).lower() for msg in error_messages) or
                    (hasattr(e, '__class__') and e.__class__.__name__ == 'InterfaceError')
                )

                if should_retry and attempt < self.max_retries:
                    logging.warning(
                        f"OceanBase connection issue (attempt {attempt+1}/{self.max_retries}): {e}"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logging.error(f"OceanBase execution failure: {e}")
                    raise
        return None

    def _handle_connection_loss(self):
        try:
            self.close()
        except Exception:
            pass
        try:
            self.connect()
        except Exception as e:
            logging.error(f"Failed to reconnect to OceanBase: {e}")
            time.sleep(0.1)
            try:
                self.connect()
            except Exception as e2:
                logging.error(f"Failed to reconnect to OceanBase on second attempt: {e2}")
                raise

    def begin(self):
        for attempt in range(self.max_retries + 1):
            try:
                return super().begin()
            except (OperationalError, InterfaceError) as e:
                error_codes = [2013, 2006]
                error_messages = ['', 'Lost connection']

                should_retry = (
                    (hasattr(e, 'args') and e.args and e.args[0] in error_codes) or
                    (str(e) in error_messages) or
                    (hasattr(e, '__class__') and e.__class__.__name__ == 'InterfaceError')
                )

                if should_retry and attempt < self.max_retries:
                    logging.warning(
                        f"Lost connection during transaction (attempt {attempt+1}/{self.max_retries})"
                    )
                    self._handle_connection_loss()
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
        return None


class PooledDatabase(Enum):
    MYSQL = RetryingPooledMySQLDatabase
    OCEANBASE = RetryingPooledOceanBaseDatabase
    POSTGRES = RetryingPooledPostgresqlDatabase


class DatabaseMigrator(Enum):
    MYSQL = MySQLMigrator
    OCEANBASE = MySQLMigrator
    POSTGRES = PostgresqlMigrator


@singleton
class BaseDataBase:
    def __init__(self):
        database_config = settings.DATABASE.copy()
        db_name = database_config.pop("name")

        pool_config = {
            'max_retries': 5,
            'retry_delay': 1,
        }
        database_config.update(pool_config)
        self.database_connection = PooledDatabase[settings.DATABASE_TYPE.upper()].value(
            db_name, **database_config
        )
        # self.database_connection = PooledDatabase[settings.DATABASE_TYPE.upper()].value(db_name, **database_config)
        logging.info("init database on cluster mode successfully")


def with_retry(max_retries=3, retry_delay=1.0):
    """Decorator: Add retry mechanism to database operations

    Args:
        max_retries (int): maximum number of retries
        retry_delay (float): initial retry delay (seconds), will increase exponentially

    Returns:
        decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # get self and method name for logging
                    self_obj = args[0] if args else None
                    func_name = func.__name__
                    lock_name = getattr(self_obj, "lock_name", "unknown") if self_obj else "unknown"

                    if retry < max_retries - 1:
                        current_delay = retry_delay * (2**retry)
                        logging.warning(f"{func_name} {lock_name} failed: {str(e)}, retrying ({retry + 1}/{max_retries})")
                        time.sleep(current_delay)
                    else:
                        logging.error(f"{func_name} {lock_name} failed after all attempts: {str(e)}")

            if last_exception:
                raise last_exception
            return False

        return wrapper

    return decorator


class PostgresDatabaseLock:
    def __init__(self, lock_name, timeout=10, db=None):
        self.lock_name = lock_name
        self.lock_id = int(hashlib.md5(lock_name.encode()).hexdigest(), 16) % (2**31 - 1)
        self.timeout = int(timeout)
        self.db = db if db else DB

    @with_retry(max_retries=3, retry_delay=1.0)
    def lock(self):
        cursor = self.db.execute_sql("SELECT pg_try_advisory_lock(%s)", (self.lock_id,))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"acquire postgres lock {self.lock_name} timeout")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"failed to acquire lock {self.lock_name}")

    @with_retry(max_retries=3, retry_delay=1.0)
    def unlock(self):
        cursor = self.db.execute_sql("SELECT pg_advisory_unlock(%s)", (self.lock_id,))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"postgres lock {self.lock_name} was not established by this thread")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"postgres lock {self.lock_name} does not exist")

    def __enter__(self):
        if isinstance(self.db, PooledPostgresqlDatabase):
            self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.db, PooledPostgresqlDatabase):
            self.unlock()

    def __call__(self, func):
        @wraps(func)
        def magic(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return magic


class MysqlDatabaseLock:
    def __init__(self, lock_name, timeout=10, db=None):
        self.lock_name = lock_name
        self.timeout = int(timeout)
        self.db = db if db else DB

    @with_retry(max_retries=3, retry_delay=1.0)
    def lock(self):
        # SQL parameters only support %s format placeholders
        cursor = self.db.execute_sql("SELECT GET_LOCK(%s, %s)", (self.lock_name, self.timeout))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"acquire mysql lock {self.lock_name} timeout")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"failed to acquire lock {self.lock_name}")

    @with_retry(max_retries=3, retry_delay=1.0)
    def unlock(self):
        cursor = self.db.execute_sql("SELECT RELEASE_LOCK(%s)", (self.lock_name,))
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception(f"mysql lock {self.lock_name} was not established by this thread")
        elif ret[0] == 1:
            return True
        else:
            raise Exception(f"mysql lock {self.lock_name} does not exist")

    def __enter__(self):
        if isinstance(self.db, PooledMySQLDatabase):
            self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.db, PooledMySQLDatabase):
            self.unlock()

    def __call__(self, func):
        @wraps(func)
        def magic(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return magic


class DatabaseLock(Enum):
    MYSQL = MysqlDatabaseLock
    OCEANBASE = MysqlDatabaseLock
    POSTGRES = PostgresDatabaseLock


DB = BaseDataBase().database_connection
DB.lock = DatabaseLock[settings.DATABASE_TYPE.upper()].value


def close_connection():
    try:
        if DB:
            DB.close_stale(age=30)
    except Exception as e:
        logging.exception(e)


class DataBaseModel(BaseModel):
    class Meta:
        database = DB


@DB.connection_context()
@DB.lock("init_database_tables", 60)
def init_database_tables(alter_fields=[]):
    members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    table_objs = []
    create_failed_list = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):
            table_objs.append(obj)

            if not obj.table_exists():
                logging.debug(f"start create table {obj.__name__}")
                try:
                    obj.create_table(safe=True)
                    logging.debug(f"create table success: {obj.__name__}")
                except Exception as e:
                    logging.exception(e)
                    create_failed_list.append(obj.__name__)
            else:
                logging.debug(f"table {obj.__name__} already exists, skip creation.")

    if create_failed_list:
        logging.error(f"create tables failed: {create_failed_list}")
        raise Exception(f"create tables failed: {create_failed_list}")
    migrate_db()


def fill_db_model_object(model_object, human_model_dict):
    for k, v in human_model_dict.items():
        attr_name = "%s" % k
        if hasattr(model_object.__class__, attr_name):
            setattr(model_object, attr_name, v)
    return model_object


class User(DataBaseModel, AuthUser):
    """
     用户认证和用户信息的核心数据模型。该模型同时继承了 DataBaseModel 和 AuthUser，为系统提供完整的用户管理功能。

     e.g:
     {
        "id": "user_abc123",
        "access_token": "eyJhbGciOiJIUzI1NiIs...",
        "nickname": "张三",
        "password": "$2b$12$...",  // bcrypt 哈希
        "email": "zhangsan@example.com",
        "avatar": "data:image/png;base64,iVBORw0KGgo...",
        "language": "Chinese",
        "color_schema": "Bright",
        "timezone": "UTC+8\tAsia/Shanghai",
        "last_login_time": "2024-01-15T10:00:00Z",
        "is_authenticated": "1",
        "is_active": "1",
        "is_anonymous": "0",
        "is_superuser": false,
        "login_channel": "email",
        "status": "1"
    }
    """
    # 敏感字段保护 : 在序列化时自动过滤，防止泄露
    SENSITIVE_FIELDS = {"password", "access_token", "email"}

    # 认证与安全字段
    id = CharField(max_length=32, primary_key=True)                                          # 主键，用户唯一标识符
    access_token = CharField(max_length=255, null=True, index=True)                          # 访问令牌，用于 API 认证
    password = CharField(max_length=255, null=True, help_text="password", index=True)        # 密码，存储哈希值（不应存储明文）
    email = CharField(max_length=255, null=False, help_text="email", unique=True)            # 邮箱，唯一标识，用于登录

    # 用户基本信息
    nickname = CharField(max_length=100, null=False, help_text="nicky name", index=True)                                                                            # 昵称，用户的显示名称
    avatar = TextField(null=True, help_text="avatar base64 string")                                                                                                 # 头像，Base64 编码的图片
    language = CharField(max_length=32, null=True, help_text="English|Chinese", default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", index=True)   # 语言：Chinese / English
    color_schema = CharField(max_length=32, null=True, help_text="Bright|Dark", default="Bright", index=True)                                                       # 配色方案：Bright / Dark
    timezone = CharField(max_length=64, null=True, help_text="Timezone", default="UTC+8\tAsia/Shanghai", index=True)                                                # 时区，默认 UTC+8 Asia/Shanghai

    # 认证状态字段
    is_authenticated = CharField(max_length=1, null=False, default="1", index=True)           # 是否已认证
    is_active = CharField(max_length=1, null=False, default="1", index=True)                  # 是否激活
    is_anonymous = CharField(max_length=1, null=False, default="0", index=True)               # 是否匿名
    is_superuser = BooleanField(null=True, help_text="is root", default=False, index=True)    # 是否超级用户

    # 审计与状态字段
    login_channel = CharField(null=True, help_text="from which user login", index=True)                                            # 最后登录时间
    last_login_time = DateTimeField(null=True, index=True)                                                                         # 登录渠道（如 email、oauth）
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)       # 状态：0（废弃）/ 1（有效）

    def __str__(self):
        return self.email

    def get_id(self):
        jwt = Serializer(secret_key=settings.get_secret_key())
        return jwt.dumps(str(self.access_token))

    def to_safe_dict(self, *, for_self: bool = False):
        """Return a dict with sensitive fields stripped for API responses.

        Email is treated as sensitive in generic serialization. Pass for_self=True
        when returning the authenticated user's own record (login, profile, etc.).
        """
        result = {k: v for k, v in self.to_dict().items() if k not in self.SENSITIVE_FIELDS}
        if for_self:
            result["email"] = self.email
        logging.debug("User %s serialized safely, filtered fields: %s", self.id, self.SENSITIVE_FIELDS)
        return result

    class Meta:
        db_table = "user"


class Tenant(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    name = CharField(max_length=100, null=True, help_text="Tenant name", index=True)
    public_key = CharField(max_length=255, null=True, index=True)
    llm_id = CharField(max_length=128, null=False, help_text="default llm ID", index=True)
    tenant_llm_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)
    embd_id = CharField(max_length=128, null=False, help_text="default embedding model ID", index=True)
    tenant_embd_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)
    asr_id = CharField(max_length=128, null=False, help_text="default ASR model ID", index=True)
    tenant_asr_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)
    img2txt_id = CharField(max_length=128, null=False, help_text="default image to text model ID", index=True)
    tenant_img2txt_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)
    rerank_id = CharField(max_length=128, null=False, help_text="default rerank model ID", index=True)
    tenant_rerank_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)
    tts_id = CharField(max_length=256, null=True, help_text="default tts model ID", index=True)
    tenant_tts_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)
    ocr_id = CharField(max_length=256, null=True, help_text="default OCR model ID", index=True)
    parser_ids = CharField(max_length=256, null=False, help_text="document processors", index=True)
    credit = IntegerField(default=512, index=True)
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    class Meta:
        db_table = "tenant"


class UserTenant(DataBaseModel):
    """
    建立用户和租户之间的关联关系，实现用户对多个租户的访问权限管理。
    """
    # 标识字段
    id = CharField(max_length=32, primary_key=True) # 主键，关联记录的唯一标识符
    # 关联字段
    user_id = CharField(max_length=32, null=False, index=True) # 用户 ID，关联的用户标识
    tenant_id = CharField(max_length=32, null=False, index=True) # 租户 ID，关联的租户标识

    # 权限字段
    role = CharField(max_length=32, null=False, help_text="UserTenantRole", index=True) # 角色，用户在租户中的权限级别
    invited_by = CharField(max_length=32, null=False, index=True) # 邀请人 ID，记录用户被谁邀请

    # 状态字段
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True) # 状态：0（已废弃）/ 1（有效）

    class Meta:
        db_table = "user_tenant"


class InvitationCode(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    code = CharField(max_length=32, null=False, index=True)
    visit_time = DateTimeField(null=True, index=True)
    user_id = CharField(max_length=32, null=True, index=True)
    tenant_id = CharField(max_length=32, null=True, index=True)
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    class Meta:
        db_table = "invitation_code"


class LLMFactories(DataBaseModel):
    name = CharField(max_length=128, null=False, help_text="LLM factory name", primary_key=True)
    logo = TextField(null=True, help_text="llm logo base64")
    tags = CharField(max_length=255, null=False, help_text="LLM, Text Embedding, Image2Text, ASR", index=True)
    rank = IntegerField(default=0, index=False)
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "llm_factories"


class LLM(DataBaseModel):
    # LLMs dictionary
    llm_name = CharField(max_length=128, null=False, help_text="LLM name", index=True)
    model_type = CharField(max_length=128, null=False, help_text="LLM, Text Embedding, Image2Text, ASR", index=True)
    fid = CharField(max_length=128, null=False, help_text="LLM factory id", index=True)
    max_tokens = IntegerField(default=0)

    tags = CharField(max_length=255, null=False, help_text="LLM, Text Embedding, Image2Text, Chat, 32k...", index=True)
    is_tools = BooleanField(null=False, help_text="support tools", default=False)
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    def __str__(self):
        return self.llm_name

    class Meta:
        primary_key = CompositeKey("fid", "llm_name")
        db_table = "llm"


class TenantLLM(DataBaseModel):
    id = PrimaryKeyField()
    tenant_id = CharField(max_length=32, null=False, index=True)
    llm_factory = CharField(max_length=128, null=False, help_text="LLM factory name", index=True)
    model_type = CharField(max_length=128, null=True, help_text="LLM, Text Embedding, Image2Text, ASR", index=True)
    llm_name = CharField(max_length=128, null=True, help_text="LLM name", default="", index=True)
    api_key = TextField(null=True, help_text="API KEY")
    api_base = CharField(max_length=255, null=True, help_text="API Base")
    max_tokens = IntegerField(default=8192, help_text="Max context token num", index=True)
    used_tokens = IntegerField(default=0, help_text="Used token num", index=True)
    status = CharField(max_length=1, null=False, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    def __str__(self):
        return self.llm_name

    class Meta:
        db_table = "tenant_llm"
        indexes = (
            (("tenant_id", "llm_factory", "llm_name"), True),
        )


class TenantLangfuse(DataBaseModel):
    tenant_id = CharField(max_length=32, null=False, primary_key=True)                                 # 主键，租户 ID
    secret_key = CharField(max_length=2048, null=False, help_text="SECRET KEY", index=True)            # 密钥，Langfuse API Secret Key
    public_key = CharField(max_length=2048, null=False, help_text="PUBLIC KEY", index=True)            # 公钥，Langfuse API Public Key
    host = CharField(max_length=128, null=False, help_text="HOST", index=True)                         # 主机地址，Langfuse 服务地址

    def __str__(self):
        return "Langfuse host" + self.host

    class Meta:
        db_table = "tenant_langfuse"


class Knowledgebase(DataBaseModel):
    """
    RAGFlow 中 知识库（Knowledge Base）的核心数据模型。知识库是 RAG 系统的核心概念，用于组织和管理文档、向量数据以及相关的处理配置。
    核心功能：存储知识库的所有配置信息，包括基本信息、模型配置、解析参数、统计数据和任务状态。
    """


    # 基本信息字段
    id = CharField(max_length=32, primary_key=True) # 主键，知识库的唯一标识符
    tenant_id = CharField(max_length=32, null=False, index=True) # 租户 ID，实现多租户隔离
    name = CharField(max_length=128, null=False, help_text="KB name", index=True)  # 知识库名称，租户内唯一
    avatar = TextField(null=True, help_text="avatar base64 string") # 头像，Base64 编码的图片
    description = TextField(null=True, help_text="KB description") # 描述，知识库的详细说明
    language = CharField(max_length=32, null=True, default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", help_text="English|Chinese", index=True) # 语言，自动检测系统语言

    # 模型配置字段
    embd_id = CharField(max_length=128, null=False, help_text="default embedding model ID", index=True) # Embedding 模型 ID，用于向量化
    tenant_embd_id = IntegerField(null=True, help_text="id in tenant_llm", index=True) # 租户 Embedding ID，在 tenant_llm 表中的 ID

    # 权限与状态字段
    permission = CharField(max_length=16, null=False, help_text="me|team", default="me", index=True) # 权限级别：me（私有）/ team（团队共享）
    created_by = CharField(max_length=32, null=False, index=True) # 创建者 ID
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True) # 状态：0（已废弃）/ 1（有效）

    # 统计字段
    doc_num = IntegerField(default=0, index=True) # 文档数量
    token_num = IntegerField(default=0, index=True) # Token 总量，用于计费
    chunk_num = IntegerField(default=0, index=True) # 分块数量
    pagerank = IntegerField(default=0, index=False) # PageRank 值，用于排序

    # 检索参数字段
    similarity_threshold = FloatField(default=0.2, index=True) # 相似度阈值，过滤低相关度结果
    vector_similarity_weight = FloatField(default=0.3, index=True) # 向量相似度权重，混合检索中向量的权重

    # 解析配置字段
    parser_id = CharField(max_length=32, null=False, help_text="default parser ID", default=ParserType.NAIVE.value, index=True) # 解析器类型：naive、knowledge_graph、table 等
    pipeline_id = CharField(max_length=32, null=True, help_text="Pipeline ID", index=True) # 管道 ID，关联 DataFlow 管道
    parser_config = JSONField(null=False, default={"pages": [[1, 1000000]], "table_context_size": 0, "image_context_size": 0}) # 解析器配置，如页面范围、上下文大小等

    # 异步任务字段
    graphrag_task_id = CharField(max_length=32, null=True, help_text="Graph RAG task ID", index=True) # GraphRAG 任务 ID
    graphrag_task_finish_at = DateTimeField(null=True) # GraphRAG 任务完成时间
    raptor_task_id = CharField(max_length=32, null=True, help_text="RAPTOR task ID", index=True) # RAPTOR 任务 ID
    raptor_task_finish_at = DateTimeField(null=True) # RAPTOR 任务完成时间
    mindmap_task_id = CharField(max_length=32, null=True, help_text="Mindmap task ID", index=True) # 思维导图任务 ID
    mindmap_task_finish_at = DateTimeField(null=True) # 思维导图任务完成时间


    def __str__(self):
        return self.name

    class Meta:
        db_table = "knowledgebase"


class Document(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    thumbnail = TextField(null=True, help_text="thumbnail base64 string")
    kb_id = CharField(max_length=256, null=False, index=True)
    parser_id = CharField(max_length=32, null=False, help_text="default parser ID", index=True)
    pipeline_id = CharField(max_length=32, null=True, help_text="pipeline ID", index=True)
    parser_config = JSONField(null=False, default={"pages": [[1, 1000000]], "table_context_size": 0, "image_context_size": 0})
    source_type = CharField(max_length=128, null=False, default="local", help_text="where dose this document come from", index=True)
    type = CharField(max_length=32, null=False, help_text="file extension", index=True)
    created_by = CharField(max_length=32, null=False, help_text="who created it", index=True)
    name = CharField(max_length=255, null=True, help_text="file name", index=True)
    location = CharField(max_length=255, null=True, help_text="where dose it store", index=True)
    size = BigIntegerField(default=0, index=True)
    token_num = IntegerField(default=0, index=True)
    chunk_num = IntegerField(default=0, index=True)
    progress = FloatField(default=0, index=True)
    progress_msg = TextField(null=True, help_text="process message", default="")
    process_begin_at = DateTimeField(null=True, index=True)
    process_duration = FloatField(default=0)
    suffix = CharField(max_length=32, null=False, help_text="The real file extension suffix", index=True)

    content_hash = CharField(max_length=32, null=True, help_text="xxhash128 of document content for change detection", default="", index=True)

    run = CharField(max_length=1, null=True, help_text="start to run processing or cancel.(1: run it; 2: cancel)", default="0", index=True)
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    class Meta:
        db_table = "document"


class File(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    parent_id = CharField(max_length=32, null=False, help_text="parent folder id", index=True)
    tenant_id = CharField(max_length=32, null=False, help_text="tenant id", index=True)
    created_by = CharField(max_length=32, null=False, help_text="who created it", index=True)
    name = CharField(max_length=255, null=False, help_text="file name or folder name", index=True)
    location = CharField(max_length=255, null=True, help_text="where dose it store", index=True)
    size = BigIntegerField(default=0, index=True)
    type = CharField(max_length=32, null=False, help_text="file extension", index=True)
    source_type = CharField(max_length=128, null=False, default="", help_text="where dose this document come from", index=True)

    class Meta:
        db_table = "file"


class File2Document(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    file_id = CharField(max_length=32, null=True, help_text="file id", index=True)
    document_id = CharField(max_length=32, null=True, help_text="document id", index=True)

    class Meta:
        db_table = "file2document"


class Task(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    doc_id = CharField(max_length=32, null=False, index=True)
    from_page = IntegerField(default=0)
    to_page = IntegerField(default=MAXIMUM_TASK_PAGE_NUMBER)
    task_type = CharField(max_length=32, null=False, default="")
    priority = IntegerField(default=0)

    begin_at = DateTimeField(null=True, index=True)
    process_duration = FloatField(default=0)

    progress = FloatField(default=0, index=True)
    progress_msg = TextField(null=True, help_text="process message", default="")
    retry_count = IntegerField(default=0)
    digest = TextField(null=True, help_text="task digest", default="")
    chunk_ids = LongTextField(null=True, help_text="chunk ids", default="")


class Dialog(DataBaseModel):
    """
    存储对话应用的所有配置，包括知识库选择、模型设置、检索参数、提示词配置等。
    数据存储示例
    {
        "id": "chat_abc123",
        "tenant_id": "tenant_001",
        "name": "技术文档助手",
        "description": "基于技术文档的问答助手",
        "icon": "data:image/png;base64,...",
        "language": "Chinese",
        "llm_id": "gpt-4",
        "tenant_llm_id": 101,
        "llm_setting": {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "max_tokens": 2048
        },
        "prompt_type": "advanced",
        "prompt_config": {
            "system": "你是一个技术文档助手，基于提供的文档回答问题。",                        # 系统提示词
            "prologue": "你好！我是技术文档助手，有什么可以帮助你的？",                       # 开场白
            "parameters": [                                                             # 参数列表
                {"name": "category", "type": "string", "description": "文档分类"}
            ],
            "empty_response": "抱歉，在知识库中没有找到相关内容。"                           # 空结果响应
        },
        "meta_data_filter": {
            "method": "auto",
            "conditions": [
                {"key": "status", "op": "==", "value": "published"}
            ]
        },
        "similarity_threshold": 0.2,
        "vector_similarity_weight": 0.3,
        "top_k": 1024,
        "top_n": 6,
        "rerank_id": "rerank_model_001",
        "tenant_rerank_id": 102,
        "kb_ids": ["kb_001", "kb_002"],
        "do_refer": "1",
        "status": "1"
    }
    """
    # 基本标识字段
    id = CharField(max_length=32, primary_key=True) # 主键，对话的唯一标识符
    tenant_id = CharField(max_length=32, null=False, index=True) # 租户 ID，实现多租户隔离
    name = CharField(max_length=255, null=True, help_text="dialog application name", index=True) # 对话名称，租户内唯一
    description = TextField(null=True, help_text="Dialog description") # 描述，对话的详细说明
    icon = TextField(null=True, help_text="icon base64 string") # 图标，Base64 编码的图片
    language = CharField(max_length=32, null=True, default="Chinese" if "zh_CN" in os.getenv("LANG", "") else "English", help_text="English|Chinese", index=True) # 语言，根据系统环境自动设置

    # 模型配置字段
    llm_id = CharField(max_length=128, null=False, help_text="default llm ID") # LLM 模型 ID，用于生成回答
    tenant_llm_id = IntegerField(null=True, help_text="id in tenant_llm", index=True) # 租户 LLM ID，在 tenant_llm 表中的 ID
    llm_setting = JSONField(null=False, default={"temperature": 0.1, "top_p": 0.3, "frequency_penalty": 0.7, "presence_penalty": 0.4, "max_tokens": 512}) # LLM 参数：temperature、top_p 等
    rerank_id = CharField(max_length=128, null=False, help_text="default rerank model ID") # 重排序模型 ID
    tenant_rerank_id = IntegerField(null=True, help_text="id in tenant_llm", index=True) # 租户重排序 ID

    # 检索参数字段
    top_k = IntegerField(default=1024) # 最大召回数量
    top_n = IntegerField(default=6) # 返回给 LLM 的片段数量
    similarity_threshold = FloatField(default=0.2) # 相似度阈值
    vector_similarity_weight = FloatField(default=0.3) # 向量权重（混合检索）

    # 知识库字段
    kb_ids = JSONField(null=False, default=[]) # 知识库 ID 列表
    meta_data_filter = JSONField(null=True, default={}) # 元数据过滤条件

    # 提示词配置字段
    prompt_type = CharField(max_length=16, null=False, default="simple", help_text="simple|advanced", index=True) # 提示词类型：simple / advanced
    prompt_config = JSONField(
        null=False,
        default={"system": "", "prologue": "Hi! I'm your assistant. What can I do for you?", "parameters": [], "empty_response": "Sorry! No relevant content was found in the knowledge base!"},
    ) # 提示词配置
    do_refer = CharField(max_length=1, null=False, default="1", help_text="it needs to insert reference index into answer or not") # 是否插入引用索引

    # 状态字段
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True) # 状态：0（已废弃）/ 1（有效）

    class Meta:
        db_table = "dialog"


class Conversation(DataBaseModel):
    """
    存储单个会话的完整信息，包括消息历史、引用来源和会话元数据。
    """
    id = CharField(max_length=32, primary_key=True)                                          # 主键，会话的唯一标识符
    dialog_id = CharField(max_length=32, null=False, index=True)                             # 对话 ID，关联 Dialog 模型
    name = CharField(max_length=255, null=True, help_text="conversation name", index=True)   # 会话名称，用户自定义
    message = JSONField(null=True)                                                           # 消息历史，对话的完整消息记录
    reference = JSONField(null=True, default=[])                                             # 引用来源，回答中引用的文档列表
    user_id = CharField(max_length=255, null=True, help_text="user_id", index=True)          # 用户 ID，会话的所有者

    class Meta:
        db_table = "conversation"


class APIToken(DataBaseModel):
    tenant_id = CharField(max_length=32, null=False, index=True)
    token = CharField(max_length=255, null=False, index=True)
    dialog_id = CharField(max_length=32, null=True, index=True)
    source = CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)
    beta = CharField(max_length=255, null=True, index=True)

    class Meta:
        db_table = "api_token"
        primary_key = CompositeKey("tenant_id", "token")


class API4Conversation(DataBaseModel):
    """
     RAGFlow 中用于存储对话（Conversation）会话记录的数据模型。它隶属于一个名为“数据模型”的代码部分，具体来说，这个类是对话历史与交互记录的核心存储结构。
     核心功能:用于持久化存储用户与 RAGFlow 系统交互时的完整对话记录，包括对话内容、引用来源、用户反馈和性能指标等。它是实现会话管理、历史回溯、质量分析和交互优化的数据基础。
     1. 多源支持
    source = CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)
    dialog：普通对话
    agent：Agent 智能体对话
    none：未指定来源

    支持按来源类型索引查询

    2. 完整的追踪能力
    tokens = IntegerField(default=0)      # 成本追踪
    duration = FloatField(default=0)      # 性能追踪
    errors = TextField(null=True)         # 错误追踪
    thumb_up = IntegerField(default=0)    # 质量反馈
    便于进行成本分析、性能优化和质量评估

    3. 版本快照
    version_title = CharField(max_length=255, null=True)
    1.记录对话创建时使用的对话流版本
    2.便于版本回滚和对比分析

    4. 实验支持
    exp_user_id = CharField(max_length=255, null=True, index=True)
    1.支持 A/B 测试场景
    2.可区分真实用户和实验用户

    💡 使用场景
    场景	        使用的字段	说明
    对话历史展示	message, reference	展示完整对话和引用
    使用量统计	tokens, duration, round	统计用户使用情况
    质量分析	    thumb_up, errors	分析回答质量
    A/B 测试	    exp_user_id, source	对比不同版本效果
    成本账单	    tokens, user_id	按用户/租户计费
     e.g:
     {
        "id": "conv_abc123",
        "name": "技术咨询-2024-01-15",
        "dialog_id": "dialog_001",
        "user_id": "user_001",
        "exp_user_id": null,
        "message": [
            {"role": "user", "content": "如何优化Python代码？"},
            {"role": "assistant", "content": "可以从以下几个方面优化..."},
            {"role": "user", "content": "具体如何优化循环？"},
            {"role": "assistant", "content": "循环优化建议：..."}
        ],
        "reference": [
            {"doc_id": "doc_001", "chunk_id": "chunk_001", "content": "Python性能优化指南"},
            {"doc_id": "doc_002", "chunk_id": "chunk_003", "content": "循环优化技巧"}
        ],
        "tokens": 1520,
        "source": "agent",
        "dsl": {"nodes": [...], "edges": [...]},
        "duration": 2.35,
        "round": 4,
        "thumb_up": 1,
        "errors": null,
        "version_title": "v2.0 优化版"
    }
    """
    # 标识与关联字段
    id = CharField(max_length=32, primary_key=True)                                                     # 主键，对话的唯一标识符
    name = CharField(max_length=255, null=True, help_text="conversation name", index=False)             # 对话名称，可为空
    dialog_id = CharField(max_length=32, null=False, index=True)                                        # 关联的对话流/应用 ID，用于关联到具体的对话应用
    user_id = CharField(max_length=255, null=False, help_text="user_id", index=True)                    # 用户 ID，标识对话所属用户
    exp_user_id = CharField(max_length=255, null=True, help_text="exp_user_id", index=True)             # 实验用户 ID，用于 A/B 测试或匿名用户追踪
    # 内容与数据字段
    message = JSONField(null=True)                                                                      # 对话消息记录，存储完整的对话轮次
    reference = JSONField(null=True, default=[])                                                        # 引用来源，记录回答所引用的文档或知识片段
    dsl = JSONField(null=True, default={})                                                              # DSL，记录对话所使用的对话流/应用
    errors = TextField(null=True, help_text="errors")                                                   # 错误信息，记录对话中的异常
    # 状态与元数据字段
    source = CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)             # 来源，记录对话来源，如：none（无来源）、agent（对话助手）、dialog（对话应用）
    tokens = IntegerField(default=0)                                                                    # 对话消耗的 token 数
    duration = FloatField(default=0, index=True)                                                        # 对话耗时，记录对话所花费的时间
    round = IntegerField(default=0, index=True)                                                         # 对话轮次计数
    thumb_up = IntegerField(default=0, index=True)                                                      # 用户点赞数（反馈）
    version_title = CharField(max_length=255, null=True, help_text="canvas version title when session created", index=False)    # 对话流版本标题，记录创建时的版本快照

    class Meta:
        db_table = "api_4_conversation"


class UserCanvas(DataBaseModel):
    """
     Agent（智能体）和 Canvas（画布）的核心存储结构。这个模型用于持久化存储用户创建的 Agent 配置、元数据和状态信息。
     核心功能:存储 Agent/Canvas 的完整信息，包括基本属性、权限控制、发布状态、标签分类和 DSL 配置。
    """
    # 标识与基本信息
    id = CharField(max_length=32, primary_key=True) # 主键，Agent/Canvas 的唯一标识符
    avatar = TextField(null=True, help_text="avatar base64 string") #头像，Base64 编码的图片数据
    user_id = CharField(max_length=255, null=False, help_text="user_id", index=True) # 所有者用户 ID，标识 Agent 的创建者
    title = CharField(max_length=255, null=True, help_text="Canvas title")  # 名称，Agent 的显示名称
    description = TextField(null=True, help_text="Canvas description") # 描述，Agent 的详细说明

    # 权限与状态控制
    permission = CharField(max_length=16, null=False, help_text="me|team", default="me", index=True) # 权限级别：me（私有）/ team（团队共享）
    release = BooleanField(null=False, help_text="is released", default=False, index=True) # 发布状态：True 表示已发布，False 表示草稿

    # 分类与组织
    canvas_type = CharField(max_length=32, null=True, help_text="Canvas type", index=True) # 画布类型，用于扩展分类
    canvas_category = CharField(max_length=32, null=False, default="agent_canvas", help_text="Canvas category: agent_canvas|dataflow_canvas", index=True) # 画布类别：agent_canvas（智能体）/ dataflow_canvas（数据流）
    tags = CharField(max_length=512, null=False, default="", help_text="Comma-separated tags for organizing agents", index=True) # 标签，逗号分隔的标签列表，用于分类和搜索

    # 核心配置
    dsl = JSONField(null=True, default={}) # DSL 配置，定义 Agent 的工作流、节点、工具等

    class Meta:
        db_table = "user_canvas"


class CanvasTemplate(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    avatar = TextField(null=True, help_text="avatar base64 string")
    title = JSONField(null=True, default=dict, help_text="Canvas title")
    description = JSONField(null=True, default=dict, help_text="Canvas description")
    canvas_type = CharField(max_length=32, null=True, help_text="Canvas type", index=True)
    canvas_types = ListField(null=True, default=list, help_text="Canvas types")
    canvas_category = CharField(max_length=32, null=False, default="agent_canvas", help_text="Canvas category: agent_canvas|dataflow_canvas", index=True)
    dsl = JSONField(null=True, default={})

    class Meta:
        db_table = "canvas_template"


class UserCanvasVersion(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    user_canvas_id = CharField(max_length=255, null=False, help_text="user_canvas_id", index=True)

    title = CharField(max_length=255, null=True, help_text="Canvas title")
    description = TextField(null=True, help_text="Canvas description")
    release = BooleanField(null=False, help_text="is released", default=False, index=True)
    dsl = JSONField(null=True, default={})

    class Meta:
        db_table = "user_canvas_version"


class MCPServer(DataBaseModel):
    """
    存储 MCP 服务器的配置，包括连接信息、认证凭证、变量和请求头。
    """
    # 基本标识字段
    id = CharField(max_length=32, primary_key=True)                                                     # 主键，MCP 服务器的唯一标识符
    name = CharField(max_length=255, null=False, help_text="MCP Server name")                           # 名称，MCP 服务器的显示名称
    tenant_id = CharField(max_length=32, null=False, index=True)                                        # 租户 ID，实现多租户隔离
    description = TextField(null=True, help_text="MCP Server description")                              # 描述，服务器的详细说明

    # 连接配置字段
    url = CharField(max_length=2048, null=False, help_text="MCP Server URL")                            # 服务器 URL，MCP 服务器的端点地址
    server_type = CharField(max_length=32, null=False, help_text="MCP Server type")                     # 服务器类型，如 stdio、sse、http 等

    # 认证与扩展字段
    variables = JSONField(null=True, default=dict, help_text="MCP Server variables")                    # 变量，用于模板替换或环境变量
    headers = JSONField(null=True, default=dict, help_text="MCP Server additional request headers")     # 请求头，自定义 HTTP 请求头

    class Meta:
        db_table = "mcp_server"


class Search(DataBaseModel):
    """
    存储搜索配置模板，包括知识库选择、检索参数、重排序设置、LLM 参数等。
    """
    # 基本信息字段
    id = CharField(max_length=32, primary_key=True) # 主键，搜索配置的唯一标识符
    avatar = TextField(null=True, help_text="avatar base64 string") # 头像，Base64 编码的图片
    tenant_id = CharField(max_length=32, null=False, index=True) # 租户 ID，实现多租户隔离
    name = CharField(max_length=128, null=False, help_text="Search name", index=True) # 配置名称，用于标识不同的搜索配置
    description = TextField(null=True, help_text="KB description") # 描述，配置的详细说明
    created_by = CharField(max_length=32, null=False, index=True) # 创建者 ID

    # 搜索配置字段
    search_config = JSONField(
        null=False,
        default={
            # 数据源
            "kb_ids": [], # 知识库 ID 列表
            "doc_ids": [], # 文档 ID 列表（可选）

            # 检索参数
            "similarity_threshold": 0.2, # 相似度阈值，用于过滤掉相似度低于阈值的结果
            "vector_similarity_weight": 0.3, # 向量相似度权重，用于计算向量相似度
            "use_kg": False, # 是否使用知识图谱

            # rerank settings
            "rerank_id": "", #  重排序模型 ID
            "top_k": 1024, # 最大召回数

            # chat settings
            "summary": False, # 是否使用摘要
            "chat_id": "", # 聊天模型 ID
            # Leave it here for reference, don't need to set default values
            "llm_setting": { # LLM 参数
                # "temperature": 0.1,
                # "top_p": 0.3,
                # "frequency_penalty": 0.7,
                # "presence_penalty": 0.4,
            },
            "chat_settingcross_languages": [], # 语言
            "highlight": False, # 是否使用高亮
            "keyword": False, # 是否使用关键词
            "web_search": False, # 是否使用 Web 搜索
            "related_search": False, # 是否使用相关搜索
            "query_mindmap": False, # 是否使用思维导图
        },
    )

    # 状态字段
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True) # 状态：0（已废弃）/ 1（有效）

    def __str__(self):
        return self.name

    class Meta:
        db_table = "search"


class PipelineOperationLog(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    document_id = CharField(max_length=32, index=True)
    tenant_id = CharField(max_length=32, null=False, index=True)
    kb_id = CharField(max_length=32, null=False, index=True)
    pipeline_id = CharField(max_length=32, null=True, help_text="Pipeline ID", index=True)
    pipeline_title = CharField(max_length=32, null=True, help_text="Pipeline title", index=True)
    parser_id = CharField(max_length=32, null=False, help_text="Parser ID", index=True)
    document_name = CharField(max_length=255, null=False, help_text="File name")
    document_suffix = CharField(max_length=255, null=False, help_text="File suffix")
    document_type = CharField(max_length=255, null=False, help_text="Document type")
    source_from = CharField(max_length=255, null=False, help_text="Source")
    progress = FloatField(default=0, index=True)
    progress_msg = TextField(null=True, help_text="process message", default="")
    process_begin_at = DateTimeField(null=True, index=True)
    process_duration = FloatField(default=0)
    dsl = JSONField(null=True, default=dict)
    task_type = CharField(max_length=32, null=False, default="")
    operation_status = CharField(max_length=32, null=False, help_text="Operation status")
    avatar = TextField(null=True, help_text="avatar base64 string")
    status = CharField(max_length=1, null=True, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True)

    class Meta:
        db_table = "pipeline_operation_log"


class Connector(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    tenant_id = CharField(max_length=32, null=False, index=True)
    name = CharField(max_length=128, null=False, help_text="Search name", index=False)
    source = CharField(max_length=128, null=False, help_text="Data source", index=True)
    input_type = CharField(max_length=128, null=False, help_text="poll/event/..", index=True)
    config = JSONField(null=False, default={})
    refresh_freq = IntegerField(default=0, index=False)
    prune_freq = IntegerField(default=0, index=False)
    timeout_secs = IntegerField(default=3600, index=False)
    indexing_start = DateTimeField(null=True, index=True)
    status = CharField(max_length=16, null=True, help_text="schedule", default="schedule", index=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "connector"


class Connector2Kb(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    connector_id = CharField(max_length=32, null=False, index=True)
    kb_id = CharField(max_length=32, null=False, index=True)
    auto_parse = CharField(max_length=1, null=False, default="1", index=False)

    class Meta:
        db_table = "connector2kb"


class ChatChannel(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    tenant_id = CharField(max_length=32, null=False, index=True)
    name = CharField(max_length=128, null=False, help_text="Bot name", index=False)
    channel = CharField(max_length=128, null=False, help_text="Chat channel type", index=True)
    config = JSONField(null=False, default={}, help_text="Channel credential & settings")
    dialog_id = CharField(max_length=32, null=True, default=None, help_text="connected dialog id", index=True)
    status = CharField(max_length=16, null=True, help_text="1: valid, 0: invalid", default="1", index=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "chat_channel"


class DateTimeTzField(CharField):
    field_type = 'VARCHAR'

    def db_value(self, value: datetime|None) -> str|None:
        if value is not None:
            if value.tzinfo is not None:
                return value.isoformat()
            else:
                return value.replace(tzinfo=timezone.utc).isoformat()
        return value

    def python_value(self, value: str|None) -> datetime|None:
        if value is not None:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                import pytz
                return dt.replace(tzinfo=pytz.UTC)
            return dt
        return value


class SyncLogs(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    connector_id = CharField(max_length=32, index=True)
    task_type = CharField(max_length=32, null=False, default="sync", index=True)
    status = CharField(max_length=128, null=False, help_text="Processing status", index=True)
    from_beginning = CharField(max_length=1, null=True, help_text="", default="0", index=False)
    new_docs_indexed = IntegerField(default=0, index=False)
    total_docs_indexed = IntegerField(default=0, index=False)
    docs_removed_from_index = IntegerField(default=0, index=False)
    error_msg = TextField(null=False, help_text="process message", default="")
    error_count = IntegerField(default=0, index=False)
    full_exception_trace = TextField(null=True, help_text="process message", default="")
    time_started = DateTimeField(null=True, index=True)
    poll_range_start = DateTimeTzField(max_length=255, null=True, index=True)
    poll_range_end = DateTimeTzField(max_length=255, null=True, index=True)
    kb_id = CharField(max_length=32, null=False, index=True)

    class Meta:
        db_table = "sync_logs"


class EvaluationDataset(DataBaseModel):
    """Ground truth dataset for RAG evaluation"""
    id = CharField(max_length=32, primary_key=True)
    tenant_id = CharField(max_length=32, null=False, index=True, help_text="tenant ID")
    name = CharField(max_length=255, null=False, index=True, help_text="dataset name")
    description = TextField(null=True, help_text="dataset description")
    kb_ids = JSONField(null=False, help_text="knowledge base IDs to evaluate against")
    created_by = CharField(max_length=32, null=False, index=True, help_text="creator user ID")
    create_time = BigIntegerField(null=False, index=True, help_text="creation timestamp")
    update_time = BigIntegerField(null=False, help_text="last update timestamp")
    status = IntegerField(null=False, default=1, help_text="1=valid, 0=invalid")

    class Meta:
        db_table = "evaluation_datasets"


class EvaluationCase(DataBaseModel):
    """Individual test case in an evaluation dataset"""
    id = CharField(max_length=32, primary_key=True)
    dataset_id = CharField(max_length=32, null=False, index=True, help_text="FK to evaluation_datasets")
    question = TextField(null=False, help_text="test question")
    reference_answer = TextField(null=True, help_text="optional ground truth answer")
    relevant_doc_ids = JSONField(null=True, help_text="expected relevant document IDs")
    relevant_chunk_ids = JSONField(null=True, help_text="expected relevant chunk IDs")
    metadata = JSONField(null=True, help_text="additional context/tags")
    create_time = BigIntegerField(null=False, help_text="creation timestamp")

    class Meta:
        db_table = "evaluation_cases"


class EvaluationRun(DataBaseModel):
    """A single evaluation run"""
    id = CharField(max_length=32, primary_key=True)
    dataset_id = CharField(max_length=32, null=False, index=True, help_text="FK to evaluation_datasets")
    dialog_id = CharField(max_length=32, null=False, index=True, help_text="dialog configuration being evaluated")
    name = CharField(max_length=255, null=False, help_text="run name")
    config_snapshot = JSONField(null=False, help_text="dialog config at time of evaluation")
    metrics_summary = JSONField(null=True, help_text="aggregated metrics")
    status = CharField(max_length=32, null=False, default="PENDING", help_text="PENDING/RUNNING/COMPLETED/FAILED")
    created_by = CharField(max_length=32, null=False, index=True, help_text="user who started the run")
    create_time = BigIntegerField(null=False, index=True, help_text="creation timestamp")
    complete_time = BigIntegerField(null=True, help_text="completion timestamp")

    class Meta:
        db_table = "evaluation_runs"


class EvaluationResult(DataBaseModel):
    """Result for a single test case in an evaluation run"""
    id = CharField(max_length=32, primary_key=True)
    run_id = CharField(max_length=32, null=False, index=True, help_text="FK to evaluation_runs")
    case_id = CharField(max_length=32, null=False, index=True, help_text="FK to evaluation_cases")
    generated_answer = TextField(null=False, help_text="generated answer")
    retrieved_chunks = JSONField(null=False, help_text="chunks that were retrieved")
    metrics = JSONField(null=False, help_text="all computed metrics")
    execution_time = FloatField(null=False, help_text="response time in seconds")
    token_usage = JSONField(null=True, help_text="prompt/completion tokens")
    create_time = BigIntegerField(null=False, help_text="creation timestamp")

    class Meta:
        db_table = "evaluation_results"


class Memory(DataBaseModel):
    """
    存储用户的记忆配置，包括记忆类型、存储方式、模型配置、遗忘策略等。

    位标志组合
    Bit 0 (1):  Raw Memory      - 原始对话记录
    Bit 1 (2):  Semantic Memory  - 语义知识
    Bit 2 (4):  Episodic Memory  - 情景记忆（事件）
    Bit 3 (8):  Procedural Memory - 程序记忆（技能）

    组合示例
    值	二进制	启用的类型
    1	0001	Raw
    2	0010	Semantic
    3	0011	Raw + Semantic
    4	0100	Episodic
    5	0101	Raw + Episodic
    7	0111	Raw + Semantic + Episodic
    8	1000	Procedural
    15	1111	全部启用
    """
    # 基本标识字段
    id = CharField(max_length=32, primary_key=True)                                            # 主键，记忆的唯一标识符
    name = CharField(max_length=128, null=False, index=False, help_text="Memory name")         # 名称，记忆的显示名称
    avatar = TextField(null=True, help_text="avatar base64 string")                            # 头像，Base64 编码的图片
    tenant_id = CharField(max_length=32, null=False, index=True)                               # 租户 ID，实现多租户隔离
    description = TextField(null=True, help_text="description")                                # 描述，记忆的详细说明
    # 记忆类型字段（位标志）
    memory_type = IntegerField(null=False, default=1, index=True, help_text="Bit flags (LSB->MSB): 1=raw, 2=semantic, 4=episodic, 8=procedural. E.g., 5 enables raw + episodic.") # 位标志设计：使用整数位标志组合多种记忆类型;类型值：1（bit 0）：原始记忆（Raw）;2（bit 1）：语义记忆（Semantic）;3 → 原始 + 语义（1 + 2）;4（bit 2）：情景记忆（Episodic）;5 → 原始 + 情景（1 + 4）;7 → 原始 + 语义 + 情景（1 + 2 + 4）;8（bit 3）：程序记忆（Procedural）;
    storage_type = CharField(max_length=32, default='table', null=False, index=True, help_text="table|graph")   # 存储类型：table（表格）/ graph（图数据库）
    memory_size = IntegerField(default=5242880, null=False, index=False)                                        # 记忆大小，默认 5MB
    forgetting_policy = CharField(max_length=32, null=False, default="FIFO", index=False, help_text="LRU|FIFO") # 遗忘策略：FIFO（先进先出）/ LRU（最近最少使用）
    # 模型配置字段
    embd_id = CharField(max_length=128, null=False, index=False, help_text="embedding model ID")                # Embedding 模型 ID，用于向量化
    tenant_embd_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)                          # 租户 Embedding ID，在 tenant_llm 表中的 ID
    llm_id = CharField(max_length=128, null=False, index=False, help_text="chat model ID")                      # 聊天模型 ID，用于生成回答
    tenant_llm_id = IntegerField(null=True, help_text="id in tenant_llm", index=True)                           # 租户 LLM ID
    # 提示词配置字段
    system_prompt = TextField(null=True, help_text="system prompt", index=False)                                # 系统提示词，定义 AI 角色
    user_prompt = TextField(null=True, help_text="user prompt", index=False)                                    # 用户提示词，引导用户输入
    temperature = FloatField(default=0.5, index=False)                                                          # 温度参数，控制随机性
    # 权限字段
    permissions = CharField(max_length=16, null=False, index=True, help_text="me|team", default="me")           # 权限级别：me（私有）/ team（团队共享）

    class Meta:
        db_table = "memory"

class SystemSettings(DataBaseModel):
    name = CharField(max_length=128, primary_key=True)
    source = CharField(max_length=32, null=False, index=False)
    data_type = CharField(max_length=32, null=False, index=False)
    value = TextField(null=False, help_text="Configuration value (JSON, string, etc.)")
    class Meta:
        db_table = "system_settings"

class TenantModelProvider(DataBaseModel):
    """
    建立租户与 LLM 提供商之间的关联，支持每个租户配置不同的 LLM 提供商。
    """
    id = CharField(max_length=32, primary_key=True) # 主键，记录的唯一标识符
    provider_name = CharField(max_length=128, null=False, index=False, help_text="LLM provider name") # 提供商名称，如 OpenAI、Azure、SiliconFlow 等
    tenant_id = CharField(max_length=32, null=False, index=True) # 租户 ID，关联的租户标识

    class Meta:
        db_table = "tenant_model_provider"
        indexes = (
            (("tenant_id", "provider_name"), True),
        )

class TenantModelInstance(DataBaseModel):
    """
    存储 LLM 模型实例的连接配置，包括 API 密钥、实例名称等。

    数据存储示例
    [
        {
            "id": "tmi_001",
            "instance_name": "gpt-4-production",
            "provider_id": "tmp_001",
            "api_key": "sk-proj-abc123...",
            "status": "active",
            "extra": "{\"base_url\": \"https://api.openai.com/v1\", \"region\": \"us-east\"}"
        },
        {
            "id": "tmi_002",
            "instance_name": "azure-eastus",
            "provider_id": "tmp_002",
            "api_key": "azure-api-key...",
            "status": "active",
            "extra": "{\"base_url\": \"https://eastus.api.cognitive.microsoft.com\", \"deployment_name\": \"gpt-4\"}"
        },
        {
            "id": "tmi_003",
            "instance_name": "siliconflow-prod",
            "provider_id": "tmp_003",
            "api_key": "sf-api-key...",
            "status": "inactive",
            "extra": "{\"base_url\": \"https://api.siliconflow.cn\", \"region\": \"intl\"}"
        }
    ]
    """
    # 字段详细解释
    id = CharField(max_length=32, primary_key=True)                                                     # 主键，实例的唯一标识符
    instance_name = CharField(max_length=128, null=False, index=False, help_text="Model instance name") # 实例名称，如 gpt-4-instance、azure-eastus
    provider_id = CharField(max_length=32, null=False, index=False)                                     # 提供商 ID，关联 TenantModelProvider
    api_key = CharField(max_length=512, null=False, index=False, help_text="API key")                   # API 密钥，用于认证
    status = CharField(max_length=32, default="active", index=False)                                    # 状态：active / inactive / error
    extra = CharField(max_length=512, default="{}", index=False)                                        # 额外配置，JSON 格式存储

    class Meta:
        db_table = "tenant_model_instance"


class TenantModel(DataBaseModel):
    """
    存储租户下具体的模型配置，包括模型名称、类型和额外参数。

    数据存储示例
    [
        {
            "id": "tm_001",
            "model_name": "gpt-4",
            "provider_id": "tmp_001",
            "instance_id": "tmi_001",
            "model_type": "chat",
            "status": "active",
            "extra": "{\"max_tokens\": 8192, \"is_tools\": true}"
        },
        {
            "id": "tm_002",
            "model_name": "text-embedding-3-small",
            "provider_id": "tmp_001",
            "instance_id": "tmi_001",
            "model_type": "embedding",
            "status": "active",
            "extra": "{\"dimensions\": 1536}"
        },
        {
            "id": "tm_003",
            "model_name": "rerank-v2",
            "provider_id": "tmp_001",
            "instance_id": "tmi_001",
            "model_type": "rerank",
            "status": "active",
            "extra": "{\"top_n\": 10}"
        }
    ]
    """
    id = CharField(max_length=32, primary_key=True)                                                     # 主键，模型配置的唯一标识符
    model_name = CharField(max_length=128, null=True, index=False, help_text="Model name")              # 模型名称，如 gpt-4、qwen-7b
    provider_id = CharField(max_length=32, null=False, index=False)                                     # 提供商 ID，关联 TenantModelProvider
    instance_id = CharField(max_length=32, null=False, index=True)                                      # 实例 ID，关联 TenantModelInstance
    model_type = CharField(max_length=32, null=False, index=False, help_text="Model type")              # 模型类型：chat / embedding / rerank / image2text
    status = CharField(max_length=32, default="active", index=False)                                    # 状态：active / inactive
    extra = CharField(max_length=1024, default="{}", index=False)                                       # 额外配置，JSON 格式

    class Meta:
        db_table = "tenant_model"


class TenantModelGroup(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    group_type = CharField(max_length=32, null=False, index=False, help_text="Group type")
    model_name = CharField(max_length=128, null=True, index=False, help_text="Model name")
    strategy = CharField(max_length=32, default="weighted", index=False, help_text="Routing strategy")

    class Meta:
        db_table = "tenant_model_group"

class TenantModelGroupMapping(DataBaseModel):
    group_id = CharField(max_length=32, null=False, index=True, help_text="Group ID")
    provider_id = CharField(max_length=32, null=False, index=False)
    instance_id = CharField(max_length=32, null=False, index=False)
    model_id = CharField(max_length=32, null=False, index=True)
    weight = IntegerField(default=100, index=False, help_text="Routing weight")
    status = CharField(max_length=32, default="active", index=False)

    class Meta:
        db_table = "tenant_model_group_mapping"
        primary_key = CompositeKey("group_id", "provider_id", "instance_id", "model_id")


def alter_db_add_column(migrator, table_name, column_name, column_type):
    try:
        migrate(migrator.add_column(table_name, column_name, column_type))
    except OperationalError as ex:
        error_codes = [1060]
        error_messages = ['Duplicate column name']

        should_skip_error = (
                (hasattr(ex, 'args') and ex.args and ex.args[0] in error_codes) or
                (str(ex) in error_messages)
        )

        if not should_skip_error:
            logging.critical(f"Failed to add {settings.DATABASE_TYPE.upper()}.{table_name} column {column_name}, operation error: {ex}")

    except Exception as ex:
        logging.critical(f"Failed to add {settings.DATABASE_TYPE.upper()}.{table_name} column {column_name}, error: {ex}")
        pass

def alter_db_column_type(migrator, table_name, column_name, new_column_type):
    try:
        migrate(migrator.alter_column_type(table_name, column_name, new_column_type))
    except Exception as ex:
        logging.critical(f"Failed to alter {settings.DATABASE_TYPE.upper()}.{table_name} column {column_name} type, error: {ex}")
        pass

def alter_db_rename_column(migrator, table_name, old_column_name, new_column_name):
    try:
        migrate(migrator.rename_column(table_name, old_column_name, new_column_name))
    except Exception:
        # rename fail will lead to a weired error.
        # logging.critical(f"Failed to rename {settings.DATABASE_TYPE.upper()}.{table_name} column {old_column_name} to {new_column_name}, error: {ex}")
        pass

def migrate_add_unique_email(migrator):
    """Deduplicates user emails and add UNIQUE constraint to email column (idempotent)"""
    # step 0: check existing index state on user.email and prepare for unique constraint
    try:
        if settings.DATABASE_TYPE.upper() == "POSTGRES":
            cursor = DB.execute_sql("""
                SELECT COUNT(*)
                FROM pg_indexes
                WHERE tablename = 'user'
                  AND indexname = 'user_email'
            """)
            result = cursor.fetchone()
            if result and result[0] > 0:
                logging.info("UNIQUE index on user.email already exists, skipping migration")
                return
        else:
            # Fetch the first index on email: tells us both the name and whether it's unique.
            # non_unique=0 means unique, non_unique=1 means non-unique.
            cursor = DB.execute_sql("""
                SELECT index_name, non_unique
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name = 'user'
                  AND column_name = 'email'
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                index_name, non_unique = row
                if non_unique == 0:
                    logging.info("UNIQUE index on user.email already exists, skipping migration")
                    return
                # Non-unique index exists (e.g. from old peewee index=True); drop it so
                # the upcoming ADD UNIQUE INDEX does not hit MySQL error 1061 "Duplicate key name".
                DB.execute_sql(f"ALTER TABLE `user` DROP INDEX `{index_name}`")
                logging.info(f"Dropped non-unique index '{index_name}' on user.email before adding unique index")
    except Exception as ex:
        logging.warning(f"Failed to check/prepare email index on user table: {ex}, continuing with migration")

    # step 1: rename duplicate rows so the UNIQUE constraint can be applied
    try:
        duplicates = User.select(User.email).group_by(User.email).having(fn.COUNT(User.id) > 1).tuples()
        for (dup_email,) in duplicates:
            # Keep the superuser row, or the oldest row if there is no superuser
            rows = list(
                User
                    .select(User.id)
                    .where(User.email == dup_email)
                    .order_by(User.is_superuser.desc(), User.create_time.asc())
                    .tuples()
            )
            for (uid,) in rows[1:]:
                new_email = f"{dup_email}_DUPLICATE_{uid[:8]}"
                User.update(email=new_email).where(User.id == uid).execute()
                logging.warning("Renamed duplicate user %s email to %s during migration", uid, new_email)
    except Exception as ex:
        logging.critical("Failed to deduplicate user.email before adding UNIQUE constraint: %s", ex)
        return

    # step 2: add UNIQUE index via migrator
    try:
        migrate(migrator.add_index("user", ("email",), unique=True))
    except (OperationalError, ProgrammingError) as ex:
        msg = str(ex)
        # MySQL 1061 "Duplicate key name" or PostgreSQL "already exists" -> already migrated
        if "1061" in msg or "Duplicate key name" in msg or "already exists" in msg.lower():
            pass
        else:
            logging.critical("Failed to add UNIQUE constraint on user.email: %s", ex)
    except Exception as ex:
        logging.critical("Failed to add UNIQUE constraint on user.email: %s", ex)



def update_tenant_llm_to_id_primary_key():
    """Add ID and set to primary key step by step."""
    if settings.DATABASE_TYPE.upper() == "POSTGRES":
        _update_tenant_llm_to_id_primary_key_postgres()
    else:
        _update_tenant_llm_to_id_primary_key_mysql()


def _update_tenant_llm_to_id_primary_key_mysql():
    """MySQL implementation: Add ID column and set as AUTO_INCREMENT primary key."""
    try:
        with DB.atomic():
            # 0. Check if 'id' column already exists
            cursor = DB.execute_sql("""
                            SELECT COLUMN_NAME
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE()
                            AND TABLE_NAME = 'tenant_llm'
                            AND COLUMN_NAME = 'id'
                        """)
            if cursor.rowcount > 0:
                return

            # 1. Add nullable column
            DB.execute_sql("ALTER TABLE tenant_llm ADD COLUMN temp_id INT NULL")

            # 2. Set ID using MySQL user variables
            DB.execute_sql("SET @row = 0;")
            DB.execute_sql("UPDATE tenant_llm SET temp_id = (@row := @row + 1) ORDER BY tenant_id, llm_factory, llm_name;")

            # 3. Drop old primary key
            DB.execute_sql("ALTER TABLE tenant_llm DROP PRIMARY KEY")

            # 4. Update ID column to primary key with AUTO_INCREMENT
            DB.execute_sql("""
            ALTER TABLE tenant_llm
            MODIFY COLUMN temp_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY
            """)

            # 5. Add unique key
            DB.execute_sql("""
                ALTER TABLE tenant_llm
                ADD CONSTRAINT uk_tenant_llm UNIQUE (tenant_id, llm_factory, llm_name)
            """)

            # 6. rename
            DB.execute_sql("ALTER TABLE tenant_llm RENAME COLUMN temp_id TO id")

            logging.info("Successfully updated tenant_llm to id primary key.")

    except Exception as e:
        logging.error(str(e))
        cursor = DB.execute_sql("""
                                    SELECT COLUMN_NAME
                                    FROM INFORMATION_SCHEMA.COLUMNS
                                    WHERE TABLE_SCHEMA = DATABASE()
                                    AND TABLE_NAME = 'tenant_llm'
                                    AND COLUMN_NAME = 'temp_id'
                                """)
        if cursor.rowcount > 0:
            DB.execute_sql("ALTER TABLE tenant_llm DROP COLUMN temp_id")


def _update_tenant_llm_to_id_primary_key_postgres():
    """PostgreSQL implementation: Add SERIAL primary key column to tenant_llm."""
    try:
        with DB.atomic():
            # 0. Check if 'id' column already exists
            cursor = DB.execute_sql("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_catalog = current_database()
                            AND table_name = 'tenant_llm'
                            AND column_name = 'id'
                        """)
            if cursor.rowcount > 0:
                return

            # 1. Add nullable integer column
            DB.execute_sql("ALTER TABLE tenant_llm ADD COLUMN temp_id INTEGER NULL")

            # 2. Assign sequential row numbers ordered consistently
            DB.execute_sql("""
                UPDATE tenant_llm
                SET temp_id = subq.rn
                FROM (
                    SELECT ctid,
                           ROW_NUMBER() OVER (ORDER BY tenant_id, llm_factory, llm_name) AS rn
                    FROM tenant_llm
                ) AS subq
                WHERE tenant_llm.ctid = subq.ctid
            """)

            # 3. Drop old composite primary key constraint
            cursor = DB.execute_sql("""
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_catalog = current_database()
                  AND table_name = 'tenant_llm'
                  AND constraint_type = 'PRIMARY KEY'
            """)
            row = cursor.fetchone()
            if row:
                DB.execute_sql(f'ALTER TABLE tenant_llm DROP CONSTRAINT "{row[0]}"')

            # 4. Make temp_id NOT NULL and create a sequence for it
            DB.execute_sql("ALTER TABLE tenant_llm ALTER COLUMN temp_id SET NOT NULL")
            DB.execute_sql("CREATE SEQUENCE IF NOT EXISTS tenant_llm_id_seq")
            DB.execute_sql("""
                SELECT setval('tenant_llm_id_seq', COALESCE((SELECT MAX(temp_id) FROM tenant_llm), 0))
            """)
            DB.execute_sql("ALTER TABLE tenant_llm ALTER COLUMN temp_id SET DEFAULT nextval('tenant_llm_id_seq')")
            DB.execute_sql("ALTER SEQUENCE tenant_llm_id_seq OWNED BY tenant_llm.temp_id")
            DB.execute_sql("ALTER TABLE tenant_llm ADD PRIMARY KEY (temp_id)")

            # 5. Add unique constraint
            DB.execute_sql("""
                ALTER TABLE tenant_llm
                ADD CONSTRAINT uk_tenant_llm UNIQUE (tenant_id, llm_factory, llm_name)
            """)

            # 6. Rename temp_id to id
            DB.execute_sql("ALTER TABLE tenant_llm RENAME COLUMN temp_id TO id")

            logging.info("Successfully updated tenant_llm to id primary key (PostgreSQL).")

    except Exception as e:
        logging.error(str(e))
        cursor = DB.execute_sql("""
                                    SELECT column_name
                                    FROM information_schema.columns
                                    WHERE table_catalog = current_database()
                                    AND table_name = 'tenant_llm'
                                    AND column_name = 'temp_id'
                                """)
        if cursor.rowcount > 0:
            DB.execute_sql("ALTER TABLE tenant_llm DROP COLUMN temp_id")


def migrate_db():
    logging.disable(logging.ERROR)
    migrator = DatabaseMigrator[settings.DATABASE_TYPE.upper()].value(DB)
    alter_db_add_column(migrator, "file", "source_type", CharField(max_length=128, null=False, default="", help_text="where dose this document come from", index=True))
    alter_db_add_column(migrator, "tenant", "rerank_id", CharField(max_length=128, null=False, default="BAAI/bge-reranker-v2-m3", help_text="default rerank model ID"))
    alter_db_add_column(migrator, "dialog", "rerank_id", CharField(max_length=128, null=False, default="", help_text="default rerank model ID"))
    alter_db_column_type(migrator, "dialog", "top_k", IntegerField(default=1024))
    alter_db_add_column(migrator, "tenant_llm", "api_key", CharField(max_length=2048, null=True, help_text="API KEY", index=True))
    alter_db_add_column(migrator, "api_token", "source", CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True))
    alter_db_add_column(migrator, "tenant", "tts_id", CharField(max_length=256, null=True, help_text="default tts model ID", index=True))
    alter_db_add_column(migrator, "api_4_conversation", "source", CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True))
    alter_db_add_column(migrator, "task", "retry_count", IntegerField(default=0))
    alter_db_column_type(migrator, "api_token", "dialog_id", CharField(max_length=32, null=True, index=True))
    alter_db_add_column(migrator, "tenant_llm", "max_tokens", IntegerField(default=8192, index=True))
    alter_db_add_column(migrator, "api_4_conversation", "dsl", JSONField(null=True, default={}))
    alter_db_add_column(migrator, "knowledgebase", "pagerank", IntegerField(default=0, index=False))
    alter_db_add_column(migrator, "api_token", "beta", CharField(max_length=255, null=True, index=True))
    alter_db_add_column(migrator, "task", "digest", TextField(null=True, help_text="task digest", default=""))
    alter_db_add_column(migrator, "task", "chunk_ids", LongTextField(null=True, help_text="chunk ids", default=""))
    alter_db_add_column(migrator, "conversation", "user_id", CharField(max_length=255, null=True, help_text="user_id", index=True))
    alter_db_add_column(migrator, "task", "task_type", CharField(max_length=32, null=False, default=""))
    alter_db_add_column(migrator, "task", "priority", IntegerField(default=0))
    alter_db_add_column(migrator, "user_canvas", "permission", CharField(max_length=16, null=False, help_text="me|team", default="me", index=True))
    alter_db_add_column(migrator, "user_canvas", "release", BooleanField(null=False, help_text="is released", default=False, index=True))
    alter_db_add_column(migrator, "llm", "is_tools", BooleanField(null=False, help_text="support tools", default=False))
    alter_db_add_column(migrator, "mcp_server", "variables", JSONField(null=True, help_text="MCP Server variables", default=dict))
    alter_db_rename_column(migrator, "task", "process_duation", "process_duration")
    alter_db_rename_column(migrator, "document", "process_duation", "process_duration")
    alter_db_add_column(migrator, "document", "suffix", CharField(max_length=32, null=False, default="", help_text="The real file extension suffix", index=True))
    alter_db_add_column(migrator, "api_4_conversation", "errors", TextField(null=True, help_text="errors"))
    alter_db_add_column(migrator, "dialog", "meta_data_filter", JSONField(null=True, default={}))
    alter_db_column_type(migrator, "canvas_template", "title", JSONField(null=True, default=dict, help_text="Canvas title"))
    alter_db_column_type(migrator, "canvas_template", "description", JSONField(null=True, default=dict, help_text="Canvas description"))
    alter_db_add_column(migrator, "user_canvas", "canvas_category", CharField(max_length=32, null=False, default="agent_canvas", help_text="agent_canvas|dataflow_canvas", index=True))
    alter_db_add_column(migrator, "canvas_template", "canvas_category", CharField(max_length=32, null=False, default="agent_canvas", help_text="agent_canvas|dataflow_canvas", index=True))
    alter_db_add_column(migrator, "canvas_template", "canvas_types", ListField(null=True, default=list, help_text="Canvas types"))
    alter_db_add_column(migrator, "knowledgebase", "pipeline_id", CharField(max_length=32, null=True, help_text="Pipeline ID", index=True))
    alter_db_add_column(migrator, "chat_channel", "dialog_id", CharField(max_length=32, null=True, help_text="connected dialog id", index=True))
    alter_db_add_column(migrator, "document", "pipeline_id", CharField(max_length=32, null=True, help_text="Pipeline ID", index=True))
    alter_db_add_column(migrator, "knowledgebase", "graphrag_task_id", CharField(max_length=32, null=True, help_text="Gragh RAG task ID", index=True))
    alter_db_add_column(migrator, "knowledgebase", "raptor_task_id", CharField(max_length=32, null=True, help_text="RAPTOR task ID", index=True))
    alter_db_add_column(migrator, "knowledgebase", "graphrag_task_finish_at", DateTimeField(null=True))
    alter_db_add_column(migrator, "knowledgebase", "raptor_task_finish_at", CharField(null=True))
    alter_db_add_column(migrator, "knowledgebase", "mindmap_task_id", CharField(max_length=32, null=True, help_text="Mindmap task ID", index=True))
    alter_db_add_column(migrator, "knowledgebase", "mindmap_task_finish_at", CharField(null=True))
    alter_db_column_type(migrator, "tenant_llm", "api_key", TextField(null=True, help_text="API KEY"))
    alter_db_add_column(migrator, "tenant_llm", "status", CharField(max_length=1, null=False, help_text="is it validate(0: wasted, 1: validate)", default="1", index=True))
    alter_db_add_column(migrator, "connector2kb", "auto_parse", CharField(max_length=1, null=False, default="1", index=False))
    alter_db_add_column(migrator, "llm_factories", "rank", IntegerField(default=0, index=False))
    alter_db_add_column(migrator, "api_4_conversation", "name", CharField(max_length=255, null=True, help_text="conversation name", index=False))
    alter_db_add_column(migrator, "api_4_conversation", "exp_user_id", CharField(max_length=255, null=True, help_text="exp_user_id", index=True))
    alter_db_add_column(migrator, "sync_logs", "task_type", CharField(max_length=32, null=False, default="sync", index=True))
    # Migrate system_settings.value from CharField to TextField for longer sandbox configs
    alter_db_column_type(migrator, "system_settings", "value", TextField(null=False, help_text="Configuration value (JSON, string, etc.)"))
    alter_db_add_column(migrator, "document", "content_hash", CharField(max_length=32, null=True, help_text="xxhash128 of document content for change detection", default="", index=True))
    update_tenant_llm_to_id_primary_key()
    alter_db_add_column(migrator, "tenant", "tenant_llm_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "tenant", "tenant_embd_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "tenant", "tenant_asr_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "tenant", "tenant_img2txt_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "tenant", "tenant_rerank_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "tenant", "tenant_tts_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "knowledgebase", "tenant_embd_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "dialog", "tenant_llm_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "dialog", "tenant_rerank_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "memory", "tenant_embd_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "memory", "tenant_llm_id", IntegerField(null=True, help_text="id in tenant_llm", index=True))
    alter_db_add_column(migrator, "user_canvas_version", "release", BooleanField(null=False, help_text="is released", default=False, index=True))
    alter_db_add_column(migrator, "user_canvas", "tags", CharField(max_length=512, null=False, default="", help_text="Comma-separated tags for organizing agents", index=True))
    alter_db_add_column(migrator, "api_4_conversation", "version_title", CharField(max_length=255, null=True, help_text="canvas version title when session created", index=False))
    alter_db_column_type(migrator, "document", "size", BigIntegerField(default=0, index=True))
    alter_db_column_type(migrator, "file", "size", BigIntegerField(default=0, index=True))
    alter_db_add_column(migrator, "tenant", "ocr_id", CharField(max_length=128, null=True, help_text="default ocr model ID", index=True))
    # Drop both the explicit "idx_*" name from later migrations AND the
    # Peewee-auto-derived "<table-as-classname>_<col1>_<col2>" name from the
    # original TenantModelInstance definition (commit dc4b82523). Databases
    # created before #15460 dropped the model's `indexes = ((...,), True)`
    # tuple still carry the auto-named compound unique index, which makes a
    # second instance with an empty api_key (e.g. Ollama) fail with
    # "Duplicate entry ... for key 'tenantmodelinstance_api_key_provider_id'"
    # — see #15699.
    legacy_indexes = [
        ("tenant_model_instance", "idx_api_key_provider_id"),
        ("tenant_model_instance", "tenantmodelinstance_api_key_provider_id"),
        ("tenant_model", "idx_provider_model_instance"),
    ]
    for table_name, index_name in legacy_indexes:
        try:
            migrate(migrator.drop_index(table_name, index_name))
        except (OperationalError, ProgrammingError) as ex:
            msg = str(ex)
            if "1091" in msg or "can't DROP" in msg.lower() or "does not exist" in msg.lower() or "already exists" in msg.lower():
                pass
            else:
                logging.critical(f"Failed to drop index {index_name} on {table_name}: {ex}")
        except Exception as ex:
            logging.critical(f"Failed to drop index {index_name} on {table_name}: {ex}")
    logging.disable(logging.NOTSET)
    # this is after re-enabling logging to allow logging changed user emails
    migrate_add_unique_email(migrator)
