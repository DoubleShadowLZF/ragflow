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

import logging

logger = logging.getLogger(__name__)


def resolve_reference_metadata_preferences(
    request_payload: dict | None = None,
    config_payload: dict | None = None,
) -> tuple[bool, set[str] | None]:
    """
    Resolve metadata include/fields from request and optional config.
    Request values take precedence over config values.
    Supports legacy request keys: include_metadata / metadata_fields.

    核心职责是：从请求和配置两个来源，解析并合并“引用元数据”的偏好设置。
    将来自 request_payload 和 config_payload 的元数据引用偏好（reference_metadata）合并，Request 优先级高于 Config，并兼容旧版 include_metadata 和 metadata_fields 字段。

    设计意图：
    支持新字段 reference_metadata.include/fields
    同时兼容旧字段 include_metadata 和 metadata_fields
    旧字段优先级最高（直接在 resolved 上覆盖）
    """
    request_payload = request_payload or {}
    config_payload = config_payload or {}

    config_ref = config_payload.get("reference_metadata", {})
    request_ref = request_payload.get("reference_metadata", {})

    resolved: dict = {}
    if isinstance(config_ref, dict):
        resolved.update(config_ref)
    if isinstance(request_ref, dict):
        resolved.update(request_ref)

    if "include_metadata" in request_payload:
        resolved["include"] = bool(request_payload.get("include_metadata"))
    if "metadata_fields" in request_payload:
        resolved["fields"] = request_payload.get("metadata_fields")

    include_metadata = bool(resolved.get("include", False))
    fields = resolved.get("fields")
    if fields is None:
        return include_metadata, None
    if not isinstance(fields, list):
        logger.warning(
            "reference_metadata.fields is not a list; include_metadata=%s fields=%r type=%s resolved=%r. "
            "enrich_chunks_with_document_metadata will skip enrichment.",
            include_metadata,
            fields,
            type(fields).__name__,
            resolved,
        )
        return include_metadata, set()
    return include_metadata, {f for f in fields if isinstance(f, str)}


def enrich_chunks_with_document_metadata(
    chunks: list[dict],
    metadata_fields: set[str] | None = None,
    *,
    kb_field: str = "kb_id",
    doc_field: str = "doc_id",
    output_field: str = "document_metadata",
) -> None:
    """
    Mutates chunk payloads in-place by attaching `document_metadata`.
    Field names can be customized for different chunk schemas.

    从 DocMetadataService 获取文档元数据，并将其附加到每个 chunk 的 document_metadata 字段中。
    """
    # 如果 metadata_fields 是空集合，直接返回（无需获取元数据）
    if metadata_fields is not None and not metadata_fields:
        return

    # 收集文档 ID
    # 1.按知识库 ID 分组收集文档 ID
    # 2.支持 kb_id 为字符串或列表
    # 3.用于批量查询元数据
    doc_ids_by_kb: dict[str, set[str]] = {}
    for chunk in chunks:
        kb_ids = chunk.get(kb_field)
        doc_id = chunk.get(doc_field)
        if not kb_ids or not doc_id:
            continue
        if isinstance(kb_ids, (list, tuple)):
            for kid in kb_ids:
                if kid:
                    doc_ids_by_kb.setdefault(kid, set()).add(doc_id)
        else:
            doc_ids_by_kb.setdefault(kb_ids, set()).add(doc_id)

    if not doc_ids_by_kb:
        return

    # Resolve service lazily so callers/tests that swap service modules at runtime
    # (e.g. via monkeypatch) don't get stuck with a stale class reference.
    # 获取元数据服务
    # 懒加载 DocMetadataService
    # 支持运行时替换（便于测试）
    from api.db.services.doc_metadata_service import DocMetadataService
    metadata_getter = getattr(DocMetadataService, "get_metadata_for_documents", None)
    if not callable(metadata_getter):
        logging.warning(
            "DocMetadataService.get_metadata_for_documents is unavailable; "
            "skipping metadata enrichment."
        )
        return

    # 批量获取元数据
    # 1.按知识库分组批量获取元数据
    # 2.合并结果到 meta_by_doc
    meta_by_doc: dict[str, dict] = {}
    for kb_id, doc_ids in doc_ids_by_kb.items():
        meta_map = metadata_getter(list(doc_ids), kb_id)
        if meta_map:
            meta_by_doc.update(meta_map)
            logging.debug("Fetched metadata for %d docs in kb_id=%s", len(meta_map), kb_id)

    # 注入元数据到 chunk
    # 1.遍历所有 chunk
    # 2.根据 doc_id 获取对应的元数据
    # 3.按需过滤元数据字段
    # 4.注入到 chunk 的 document_metadata 字段
    for chunk in chunks:
        doc_id = chunk.get(doc_field)
        if not doc_id:
            continue
        meta = meta_by_doc.get(doc_id)
        if not meta:
            continue
        if metadata_fields is not None:
            meta = {k: v for k, v in meta.items() if k in metadata_fields}
        if meta:
            chunk[output_field] = meta
            logging.debug("Enriched chunk for doc_id=%s with %d metadata fields: %s", doc_id, len(meta), list(meta.keys()))
