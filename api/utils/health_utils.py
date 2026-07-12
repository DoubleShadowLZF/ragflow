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
健康检查工具模块

提供 RAGFlow 各组件的健康探测和状态查询功能，主要分为三类：
1. 组件级健康检查：数据库、Redis、文档检索引擎、对象存储的连通性探测
2. 数据库专项状态：MySQL 进程列表、Elasticsearch 集群统计、Infinity/OceanBase 性能指标
3. 服务级存活检查：MinIO 存活、RAGFlow Server/Task Executor 心跳

所有健康检查函数返回统一结构：{"status": "alive|timeout|unhealthy|...", "message": ...}
"""

from datetime import datetime
import json
import os
import requests
from timeit import default_timer as timer

from api.db.db_models import DB
from rag.utils.redis_conn import REDIS_CONN
from rag.utils.es_conn import ESConnection
from rag.utils.infinity_conn import InfinityConnection
from rag.utils.ob_conn import OBConnection
from common import settings


# =============================================================================
#  内部工具
# =============================================================================

def _ok_nok(ok: bool) -> str:
    """将布尔健康状态转为可读字符串。"""
    return "ok" if ok else "nok"


# =============================================================================
#  组件级健康检查（轻量级探测，用于 /healthz 等接口）
# =============================================================================

def check_db() -> tuple[bool, dict]:
    """检查数据库连接是否正常。

    执行 SELECT 1 作为轻量探测，兼容 MySQL 和 PostgreSQL。
    """
    st = timer()
    try:
        DB.execute_sql("SELECT 1")
        return True, {"elapsed": f"{(timer() - st) * 1000.0:.1f}"}
    except Exception as e:
        return False, {"elapsed": f"{(timer() - st) * 1000.0:.1f}", "error": str(e)}


def check_redis() -> tuple[bool, dict]:
    """检查 Redis 连接是否正常。"""
    st = timer()
    try:
        ok = bool(REDIS_CONN.health())
        return ok, {"elapsed": f"{(timer() - st) * 1000.0:.1f}"}
    except Exception as e:
        return False, {"elapsed": f"{(timer() - st) * 1000.0:.1f}", "error": str(e)}


def check_doc_engine() -> tuple[bool, dict]:
    """检查文档检索引擎（ES / Infinity / OceanBase）是否正常。"""
    st = timer()
    try:
        meta = settings.docStoreConn.health()
        return True, {"elapsed": f"{(timer() - st) * 1000.0:.1f}", **(meta or {})}
    except Exception as e:
        return False, {"elapsed": f"{(timer() - st) * 1000.0:.1f}", "error": str(e)}


def check_storage() -> tuple[bool, dict]:
    """检查对象存储（MinIO / S3 / Azure 等）是否正常。"""
    st = timer()
    try:
        settings.STORAGE_IMPL.health()
        return True, {"elapsed": f"{(timer() - st) * 1000.0:.1f}"}
    except Exception as e:
        return False, {"elapsed": f"{(timer() - st) * 1000.0:.1f}", "error": str(e)}


# =============================================================================
#  文档检索引擎专项状态查询（ES / Infinity / OceanBase）
# =============================================================================

def get_es_cluster_stats() -> dict:
    """获取 Elasticsearch 集群的统计信息（节点数、分片数、文档数等）。

    Raises:
        Exception: 当前文档引擎不是 Elasticsearch 时抛出异常
    """
    doc_engine = os.getenv('DOC_ENGINE', 'elasticsearch')
    if doc_engine != 'elasticsearch':
        raise Exception("Elasticsearch is not in use.")
    try:
        return {
            "status": "alive",
            "message": ESConnection().get_cluster_stats()
        }
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


def get_infinity_status():
    """获取 Infinity 向量数据库的健康状态。

    Raises:
        Exception: 当前文档引擎不是 infinity 时抛出异常
    """
    doc_engine = os.getenv('DOC_ENGINE', 'elasticsearch')
    if doc_engine != 'infinity':
        raise Exception("Infinity is not in use.")
    try:
        return {
            "status": "alive",
            "message": InfinityConnection().health()
        }
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


def get_oceanbase_status():
    """
    获取 OceanBase 数据库的综合状态，包含健康信息和性能指标。

    返回内容：
    - health: 数据库连接/版本等基础健康信息
    - performance: QPS、慢查询数、连接数等运行时性能指标

    Raises:
        Exception: 当前文档引擎不是 oceanbase 时抛出异常
    """
    doc_engine = os.getenv('DOC_ENGINE', 'elasticsearch')
    if doc_engine != 'oceanbase':
        raise Exception("OceanBase is not in use.")
    try:
        ob_conn = OBConnection()
        health_info = ob_conn.health()
        performance_metrics = ob_conn.get_performance_metrics()

        # 合并健康信息和性能指标
        status = "alive" if health_info.get("status") == "healthy" else "timeout"

        return {
            "status": status,
            "message": {
                "health": health_info,
                "performance": performance_metrics
            }
        }
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


def check_oceanbase_health() -> dict:
    """
    对 OceanBase 进行深度健康检查，返回多维度的详细指标。

    检查维度包括：
    - 连接状态：是否成功连接
    - 查询延迟：SQL 响应时间（毫秒）
    - 存储使用率：已用空间 / 总空间
    - QPS：每秒查询数
    - 慢查询统计：当前慢查询数量
    - 连接池：活跃连接数 / 最大连接数

    健康判定规则：
    - unhealthy: 连接断开或健康检查不通过
    - degraded: 连接正常但延迟 >= 1 秒
    - healthy: 连接正常且延迟 < 1 秒
    """
    doc_engine = os.getenv('DOC_ENGINE', 'elasticsearch')
    if doc_engine != 'oceanbase':
        return {
            "status": "not_configured",
            "details": {
                "connection": "not_configured",
                "message": "OceanBase is not configured as the document engine"
            }
        }

    try:
        ob_conn = OBConnection()
        health_info = ob_conn.health()
        performance_metrics = ob_conn.get_performance_metrics()

        # 判断连接状态
        connection_status = performance_metrics.get("connection", "unknown")

        # 连接断开或不健康 → unhealthy
        if connection_status == "disconnected" or health_info.get("status") != "healthy":
            return {
                "status": "unhealthy",
                "details": {
                    "connection": connection_status,
                    "latency_ms": performance_metrics.get("latency_ms", 0),
                    "storage_used": performance_metrics.get("storage_used", "N/A"),
                    "storage_total": performance_metrics.get("storage_total", "N/A"),
                    "query_per_second": performance_metrics.get("query_per_second", 0),
                    "slow_queries": performance_metrics.get("slow_queries", 0),
                    "active_connections": performance_metrics.get("active_connections", 0),
                    "max_connections": performance_metrics.get("max_connections", 0),
                    "uri": health_info.get("uri", "unknown"),
                    "version": health_info.get("version_comment", "unknown"),
                    "error": health_info.get("error", performance_metrics.get("error"))
                }
            }

        # 检查是否健康：连接成功且延迟 < 1 秒
        is_healthy = (
            connection_status == "connected" and
            performance_metrics.get("latency_ms", float('inf')) < 1000
        )

        return {
            "status": "healthy" if is_healthy else "degraded",
            "details": {
                "connection": performance_metrics.get("connection", "unknown"),
                "latency_ms": performance_metrics.get("latency_ms", 0),
                "storage_used": performance_metrics.get("storage_used", "N/A"),
                "storage_total": performance_metrics.get("storage_total", "N/A"),
                "query_per_second": performance_metrics.get("query_per_second", 0),
                "slow_queries": performance_metrics.get("slow_queries", 0),
                "active_connections": performance_metrics.get("active_connections", 0),
                "max_connections": performance_metrics.get("max_connections", 0),
                "uri": health_info.get("uri", "unknown"),
                "version": health_info.get("version_comment", "unknown")
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "details": {
                "connection": "disconnected",
                "error": str(e)
            }
        }


# =============================================================================
#  数据库专项状态查询（MySQL）
# =============================================================================

def get_mysql_status():
    """获取 MySQL 当前进程列表，用于诊断连接数和慢查询。

    通过 SHOW PROCESSLIST 列出所有活跃连接的信息，包括用户、主机、
    执行的命令、耗时、状态等。
    """
    try:
        cursor = DB.execute_sql("SHOW PROCESSLIST;")
        res_rows = cursor.fetchall()
        headers = ['id', 'user', 'host', 'db', 'command', 'time', 'state', 'info']
        cursor.close()
        return {
            "status": "alive",
            "message": [dict(zip(headers, r)) for r in res_rows]
        }
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


# =============================================================================
#  服务级存活检查（MinIO / Redis / RAGFlow Server / Task Executor）
# =============================================================================

def _minio_scheme_and_verify():
    """
    根据 MinIO 配置决定 HTTP 协议（http/https）和 SSL 证书验证策略。

    - secure: 控制使用 http 还是 https
    - verify: 控制是否验证 SSL 证书（自签名证书场景下设为 False）
    """
    secure = settings.MINIO.get("secure", False)
    if isinstance(secure, str):
        secure = secure.lower() in ("true", "1", "yes")
    scheme = "https" if secure else "http"
    verify = settings.MINIO.get("verify", True)
    if isinstance(verify, str):
        verify = verify.lower() not in ("false", "0", "no")
    elif isinstance(verify, bool):
        pass
    else:
        verify = bool(verify)
    return scheme, verify


def check_minio_alive():
    """
    通过 MinIO 内置的 /minio/health/live 端点检查其存活状态。

    根据配置自动选择 http/https 协议和 SSL 证书验证策略。
    """
    start_time = timer()
    try:
        scheme, verify = _minio_scheme_and_verify()
        url = f"{scheme}://{settings.MINIO['host']}/minio/health/live"
        response = requests.get(url, timeout=10, verify=verify)
        if response.status_code == 200:
            return {"status": "alive", "message": f"Confirm elapsed: {(timer() - start_time) * 1000.0:.1f} ms."}
        return {"status": "timeout", "message": f"Confirm elapsed: {(timer() - start_time) * 1000.0:.1f} ms."}
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


def get_redis_info():
    """获取 Redis 服务器的详细运行信息（内存、连接数、键数量等）。"""
    try:
        return {
            "status": "alive",
            "message": REDIS_CONN.info()
        }
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


def check_ragflow_server_alive():
    """通过调用自身的 /api/v1/system/ping 端点检查 RAGFlow API 服务是否存活。

    注意：若 HOST_IP 为 0.0.0.0，会自动替换为 127.0.0.1 以确保本地可达。
    """
    start_time = timer()
    try:
        url = f'http://{settings.HOST_IP}:{settings.HOST_PORT}/api/v1/system/ping'
        if '0.0.0.0' in url:
            url = url.replace('0.0.0.0', '127.0.0.1')
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return {"status": "alive", "message": f"Confirm elapsed: {(timer() - start_time) * 1000.0:.1f} ms."}
        else:
            return {"status": "timeout", "message": f"Confirm elapsed: {(timer() - start_time) * 1000.0:.1f} ms."}
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}",
        }


def check_task_executor_alive():
    """检查任务执行器（Task Executor）是否存活。

    通过 Redis 中的心跳记录判断：从 TASKEXE 集合读取所有注册的执行器，
    查询最近 30 分钟的心跳数据。只要有任一台执行器有心跳，即判定为 alive。
    """
    task_executor_heartbeats = {}
    try:
        # 从 Redis 集合获取所有注册的任务执行器 ID
        task_executors = REDIS_CONN.smembers("TASKEXE")
        now = datetime.now().timestamp()
        for task_executor_id in task_executors:
            # 查询最近 30 分钟的心跳记录
            heartbeats = REDIS_CONN.zrangebyscore(task_executor_id, now - 60 * 30, now)
            heartbeats = [json.loads(heartbeat) for heartbeat in heartbeats]
            task_executor_heartbeats[task_executor_id] = heartbeats
        if task_executor_heartbeats:
            # 至少有一台执行器有心跳即视为存活
            status = "alive" if any(task_executor_heartbeats.values()) else "timeout"
            return {"status": status, "message": task_executor_heartbeats}
        else:
            return {"status": "timeout", "message": "Not found any task executor."}
    except Exception as e:
        return {
            "status": "timeout",
            "message": f"error: {str(e)}"
        }


# =============================================================================
#  全量健康检查汇总（用于 /system/healthz 端点）
# =============================================================================

def run_health_checks() -> tuple[dict, bool]:
    """运行全量健康检查，汇总所有核心组件的状态。

    检查范围：数据库、Redis、文档检索引擎、对象存储四个组件。
    每个组件失败时将详细元信息存入 _meta 字段，方便问题定位。

    Returns:
        tuple[dict, bool]:
            - dict: {"db": "ok"|"nok", "redis": ..., "doc_engine": ..., "storage": ..., "status": "ok"|"nok", "_meta": ...}
            - bool: 是否所有组件都健康
    """
    result: dict[str, str | dict] = {}

    # 数据库检查
    db_ok, db_meta = check_db()
    result["db"] = _ok_nok(db_ok)
    if not db_ok:
        result.setdefault("_meta", {})["db"] = db_meta

    # Redis 检查
    try:
        redis_ok, redis_meta = check_redis()
        result["redis"] = _ok_nok(redis_ok)
        if not redis_ok:
            result.setdefault("_meta", {})["redis"] = redis_meta
    except Exception:
        result["redis"] = "nok"

    # 文档检索引擎检查
    try:
        doc_ok, doc_meta = check_doc_engine()
        result["doc_engine"] = _ok_nok(doc_ok)
        if not doc_ok:
            result.setdefault("_meta", {})["doc_engine"] = doc_meta
    except Exception:
        result["doc_engine"] = "nok"

    # 对象存储检查
    try:
        sto_ok, sto_meta = check_storage()
        result["storage"] = _ok_nok(sto_ok)
        if not sto_ok:
            result.setdefault("_meta", {})["storage"] = sto_meta
    except Exception:
        result["storage"] = "nok"

    # 只有四个核心组件全部 ok 时，整体才判定为 ok
    all_ok = (result.get("db") == "ok") and (result.get("redis") == "ok") and (result.get("doc_engine") == "ok") and (
                result.get("storage") == "ok")
    result["status"] = "ok" if all_ok else "nok"
    return result, all_ok
