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
Service for adjusting chunk recall weights based on user feedback.

When users upvote or downvote responses, this service updates the pagerank_fea
field of the referenced chunks to improve future retrieval quality.

This feature is disabled by default. Enable it by setting the environment
variable CHUNK_FEEDBACK_ENABLED=true.

Weighting modes (CHUNK_FEEDBACK_WEIGHTING):
- relevance (default): one small budget per feedback event is split across
  cited chunks using retrieval scores (similarity / vector_similarity /
  term_similarity) from the reference payload, so chunks that drove the answer
  move more than weak tail context.
- uniform: legacy behavior — each cited chunk receives the full increment or
  decrement (stronger total effect when many chunks are cited).

Budget per feedback event is a small integer (1) applied to pagerank_fea
(0–100, integer in Infinity/OB/ES mappings). Relevance mode splits that unit
across cited chunks; uniform mode applies one unit per chunk (legacy, stronger
when many chunks are cited).

Infinity uses row_id (returned by search results since PR #13901) for targeted
single-row updates. If a concurrent update changes the row_id, the Infinity
connector retries with a fresh row_id lookup.
"""
import logging
import math
import os
from typing import List, Tuple

from common.constants import PAGERANK_FLD
from common import settings
from rag.nlp.search import index_name


# Feature flag - disabled by default to prevent unintended side effects
CHUNK_FEEDBACK_ENABLED = os.getenv("CHUNK_FEEDBACK_ENABLED", "false").lower() == "true"

# relevance: fixed budget split by retrieval signals; uniform: delta per chunk
CHUNK_FEEDBACK_WEIGHTING = os.getenv("CHUNK_FEEDBACK_WEIGHTING", "relevance").strip().lower()

# Integer units — matches pagerank_fea integer columns in doc stores
UPVOTE_WEIGHT_INCREMENT = 1
DOWNVOTE_WEIGHT_DECREMENT = 1
MIN_PAGERANK_WEIGHT = 0
MAX_PAGERANK_WEIGHT = 100

_SCORE_KEYS = ("similarity", "vector_similarity", "term_similarity")


def _retrieval_signal(chunk: dict) -> float:
    """Best available retrieval score for feedback allocation; 0 if none."""
    best = 0.0
    for key in _SCORE_KEYS:
        raw = chunk.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and val > best:
            best = val
    return best


def _split_integer_budget(magnitudes: List[float], budget: int) -> List[int]:
    """Split nonnegative integer budget across positive magnitudes (largest remainder).
    将整数预算按给定的权重（magnitudes）比例分配到 N 个分块，保证每个分块至少获得整数分配，且总和等于预算。
    """
    # 边界情况处理:没有分块或预算为 0 → 全部返回 0
    n = len(magnitudes)
    if n == 0 or budget == 0:
        return [0] * n
    # 零权重处理（均匀分配）
    # 1.如果所有权重为 0 → 均匀分配
    # 2.每个分块获得 budget // n
    # 3.剩余的 budget % n 分配给前几个分块
    total = sum(magnitudes)
    if total <= 0:
        base = budget // n
        rem = budget % n
        out = [base] * n
        for i in range(rem):
            out[i] += 1
        return out
    # 计算原始比例
    # 1.每个分块的理论分配值（浮点数）
    # 2.总和 = budget
    raw = [budget * m / total for m in magnitudes]
    # 向下取整
    # 每个分块获得整数部分
    floors = [int(math.floor(r)) for r in raw]
    # remainder 是未分配的余数
    remainder = budget - sum(floors)
    # 最大余数分配
    # 1.按余数从大到小排序
    # 2.将剩余单位逐个分配给余数最大的分块
    order = sorted(range(n), key=lambda i: raw[i] - floors[i], reverse=True)
    for j in range(remainder):
        floors[order[j]] += 1
    return floors


def _allocate_deltas_uniform(
    chunk_rows: List[Tuple[str, str]],
    signed_budget: int,
) -> List[Tuple[str, str, int]]:
    """Each row gets the full signed step (legacy: one unit per cited chunk).
    为每个引用的分块分配相同的权重增量值。
    """
    # 确定增量步长
    # signed_budget > 0（点赞）：使用 UPVOTE_WEIGHT_INCREMENT（正数）
    # signed_budget < 0（点踩）：使用 -DOWNVOTE_WEIGHT_DECREMENT（负数）
    # 所有分块获得相同的增量值
    step = UPVOTE_WEIGHT_INCREMENT if signed_budget > 0 else -DOWNVOTE_WEIGHT_DECREMENT
    # 生成增量列表
    # 1.遍历所有分块
    # 2.每个分块获得相同的 step
    # 3.返回 (chunk_id, kb_id, delta) 列表
    return [(cid, kb, step) for cid, kb in chunk_rows]


def _allocate_deltas_relevance(
    chunk_rows: List[Tuple[str, str, dict]],
    signed_budget: int,
) -> List[Tuple[str, str, int]]:
    """
    Split |signed_budget| integer units across chunks using retrieval_signal weights.
    chunk_rows: (chunk_id, kb_id, original_chunk_dict)

    根据分块的相关性信号，将总预算（整数）按比例分配到所有引用的分块。
    """
    if not chunk_rows:
        return []

    # 提取相关性信号
    # 1.从每个分块中提取相关性信号（如相似度分数）
    # 2.如果信号 ≤ 0，使用默认值 1.0（避免零值）
    magnitudes = []
    for _cid, _kb, ch in chunk_rows:
        s = _retrieval_signal(ch)
        magnitudes.append(s if s > 0 else 1.0)

    # 归一化处理
    # 1.计算总信号值
    # 2.如果总值为 0，使用均匀分布（所有分块权重相同）
    total = sum(magnitudes)
    if total <= 0:
        magnitudes = [1.0] * len(chunk_rows)

    # 确定方向
    # signed_budget > 0（点赞）→ 正数
    # signed_budget < 0（点踩）→ 负数
    # 提取绝对值用于分配
    sign = 1 if signed_budget > 0 else -1
    budget_abs = abs(signed_budget)
    # 整数预算分配
    # 1.将整数预算按比例分配到各分块
    # 2.返回整数列表，总和等于 budget_abs
    parts = _split_integer_budget(magnitudes, budget_abs)
    # 生成增量列表
    # 1.每个分块获得 sign * p 的增量
    # 2.总增量 = sign × budget_abs
    out: List[Tuple[str, str, int]] = []
    for (cid, kb, _ch), p in zip(chunk_rows, parts, strict=True):
        out.append((cid, kb, sign * p))
    return out


class ChunkFeedbackService:
    """Service to update chunk weights based on user feedback."""

    @staticmethod
    def _feedback_rows_from_reference(reference: dict) -> List[Tuple[str, str, dict]]:
        """(chunk_id, kb_id, raw_chunk) for chunks that can be updated (single pass).

        raw_chunk is kept for retrieval-signal weighting and optional row_id.
        """
        if not reference:
            return []
        rows: List[Tuple[str, str, dict]] = []
        for chunk in reference.get("chunks", []):
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            kb_id = chunk.get("dataset_id") or chunk.get("kb_id")
            if chunk_id and kb_id:
                rows.append((chunk_id, kb_id, chunk))
        return rows

    @staticmethod
    def update_chunk_weight(
        tenant_id: str,
        chunk_id: str,
        kb_id: str,
        delta: int,
        row_id: int | None = None,
    ) -> bool:
        """
        Update the pagerank weight of a single chunk.

        Elasticsearch, OpenSearch, OceanBase/SeekDB, and Infinity use an
        atomic adjust on the doc store when supported. Infinity passes
        row_id (from retrieval results) for targeted single-row updates.

        Args:
            tenant_id: The tenant ID for index naming
            chunk_id: The chunk ID to update
            kb_id: The knowledgebase ID
            delta: Signed integer weight change (pagerank_fea is stored as int)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            idx_name = index_name(tenant_id)
            conn = settings.docStoreConn
            adjust = getattr(conn, "adjust_chunk_pagerank_fea", None)
            if callable(adjust):
                kwargs: dict = {}
                if row_id is not None:
                    kwargs["row_id"] = row_id
                success = adjust(
                    chunk_id,
                    idx_name,
                    kb_id,
                    float(delta),
                    MIN_PAGERANK_WEIGHT,
                    MAX_PAGERANK_WEIGHT,
                    **kwargs,
                )
                if success:
                    logging.info(
                        "Adjusted chunk %s pagerank by %s (atomic)",
                        chunk_id,
                        delta,
                    )
                else:
                    logging.warning("Failed atomic pagerank adjust for chunk %s", chunk_id)
                return success

            chunk = conn.get(chunk_id, idx_name, [kb_id])
            if not chunk:
                logging.warning("Chunk %s not found in index %s", chunk_id, idx_name)
                return False

            current_weight = float(chunk.get(PAGERANK_FLD, 0) or 0)
            new_weight = current_weight + float(delta)
            new_weight = max(float(MIN_PAGERANK_WEIGHT), min(float(MAX_PAGERANK_WEIGHT), new_weight))

            condition = {"id": chunk_id}
            doc_engine = settings.DOC_ENGINE.lower()
            if new_weight <= 0.0 and doc_engine in ("elasticsearch", "opensearch"):
                new_value = {"remove": PAGERANK_FLD}
            else:
                new_value = {PAGERANK_FLD: new_weight}

            success = conn.update(condition, new_value, idx_name, kb_id)

            if success:
                logging.info(
                    "Updated chunk %s pagerank: %s -> %s",
                    chunk_id,
                    current_weight,
                    new_weight,
                )
            else:
                logging.warning("Failed to update chunk %s pagerank", chunk_id)

            return success

        except Exception as e:
            logging.exception("Error updating chunk %s weight: %s", chunk_id, e)
            return False

    @classmethod
    def apply_feedback(
        cls,
        tenant_id: str,
        reference: dict,
        is_positive: bool
    ) -> dict:
        """
        Apply user feedback to all chunks referenced in a response.

        Args:
            tenant_id: The tenant ID
            reference: The reference dict from the conversation message
            is_positive: True for upvote (thumbup), False for downvote

        Returns:
            Dict with 'success_count', 'fail_count', and 'chunk_ids' processed

        将用户反馈（点赞/点踩）传播到引用的文档分块，通过调整分块权重来优化检索排序。
        """
        # Check if feature is enabled
        # 功能开关检查
        # 1.检查功能是否启用
        # 2.禁用时直接返回
        if not CHUNK_FEEDBACK_ENABLED:
            logging.debug("Chunk feedback feature is disabled")
            return {"success_count": 0, "fail_count": 0, "chunk_ids": [], "disabled": True}

        # 提取分块 ID
        # 1.从引用中提取所有分块 ID
        # 2.没有分块时直接返回
        rows = cls._feedback_rows_from_reference(reference)
        chunk_ids = [r[0] for r in rows]

        if not chunk_ids:
            logging.debug("No chunk IDs found in reference for feedback")
            return {"success_count": 0, "fail_count": 0, "chunk_ids": []}

        # 计算权重增量
        # 点赞：增加权重（正数）
        # 点踩：减少权重（负数）
        signed_budget = (
            UPVOTE_WEIGHT_INCREMENT if is_positive else -DOWNVOTE_WEIGHT_DECREMENT
        )
        weighting = CHUNK_FEEDBACK_WEIGHTING if CHUNK_FEEDBACK_WEIGHTING in (
            "uniform",
            "relevance",
        ) else "relevance"

        # 分配增量
        # uniform：均匀分配（每个分块获得相同增量）
        if weighting == "uniform":
            deltas = _allocate_deltas_uniform([(r[0], r[1]) for r in rows], signed_budget)
        # relevance：按相关性分配（相关度高的分块获得更多增量）
        else:
            deltas = _allocate_deltas_relevance(rows, signed_budget)

        success_count = 0
        fail_count = 0

        # 更新分块权重
        # 1.遍历所有分块
        # 2.获取 row_id（用于数据库更新）
        # 3.调用 update_chunk_weight 更新权重
        # 4.统计成功/失败数量
        row_by_chunk = {r[0]: r[2].get("row_id") for r in rows}
        for chunk_id, kb_id, delta in deltas:
            if delta == 0:
                continue
            rid = row_by_chunk.get(chunk_id)
            rid_int = None
            if rid is not None:
                try:
                    rid_int = int(rid)
                except (TypeError, ValueError):
                    pass
            if cls.update_chunk_weight(tenant_id, chunk_id, kb_id, delta, row_id=rid_int):
                success_count += 1
            else:
                fail_count += 1

        logging.info(
            "Applied %s feedback (%s) to %s/%s chunks",
            "positive" if is_positive else "negative",
            weighting,
            success_count,
            len(chunk_ids),
        )

        return {
            "success_count": success_count,
            "fail_count": fail_count,
            "chunk_ids": chunk_ids
        }
