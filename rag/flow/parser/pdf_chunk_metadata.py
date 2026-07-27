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

"""PDF 切块元数据处理模块。

负责 PDF 文档解析后的位置信息提取、规范化、多栏重排以及预览图生成。
核心功能包括：
- 从多种位置表示格式中统一提取 PDF 坐标
- 多栏文档的阅读顺序重排
- 为文本切块生成带高亮的 PDF 预览缩略图
- 切块位置的合并与索引字段构建
"""

import io
import logging
import sys
from copy import deepcopy
from functools import partial

import numpy as np
import pdfplumber
from PIL import Image

from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from common import settings
from common.misc_utils import get_uuid
from deepdoc.parser.pdf_parser import LOCK_KEY_pdfplumber, RAGFlowPdfParser
from rag.utils.base64_image import image2id

# === 预览图生成相关常量 ===

# 预览图中各段落之间的垂直间距（像素）
PDF_PREVIEW_GAP = 6

# 预览图首尾的上下文扩展高度（用于显示段落周围的内容）
PDF_PREVIEW_CONTEXT = 120

# 预览图渲染的缩放倍数（分辨率 = 72 * zoom DPI）
PDF_PREVIEW_ZOOM = 3

# PDF 位置信息的内部存储键名
PDF_POSITIONS_KEY = "_pdf_positions"

# 多栏检测时的缩放倍数
PDF_MULTI_COLUMN_ZOOM = 3


def _extract_raw_positions(item):
    """从段落项中提取原始的 PDF 位置信息。

    支持多种位置数据格式（按优先级尝试）：
    1. _pdf_positions 内部键——已规范化过的位置列表
    2. positions 字段——解析器直接输出的位置列表
    3. position_tag 字符串——TxtParser 风格的标签格式，如 "@@1\t0.1\t0.9\t0.2\t0.3##"
    4. position_int 字段——整数坐标格式
    5. page_number + x0/x1/top/bottom 字段——单点位置格式

    Args:
        item: 段落数据字典

    Returns:
        list: 位置坐标列表，统一格式为 [[page_number, left, right, top, bottom], ...]
    """
    positions = item.get(PDF_POSITIONS_KEY)
    if isinstance(positions, list):
        return deepcopy(positions)

    positions = item.get("positions")
    if isinstance(positions, list):
        return deepcopy(positions)

    position_tag = item.get("position_tag")
    if isinstance(position_tag, str) and position_tag:
        return [[pos[0][-1], *pos[1:]] for pos in RAGFlowPdfParser.extract_positions(position_tag)]

    position_int = item.get("position_int")
    if isinstance(position_int, list):
        return [
            list(pos)
            for pos in position_int
            if isinstance(pos, (list, tuple)) and len(pos) >= 5
        ]

    # 从独立的坐标字段构建位置列表
    if item.get("page_number") is not None and all(
        item.get(key) is not None for key in ["x0", "x1", "top", "bottom"]
    ):
        return [[item["page_number"], item["x0"], item["x1"], item["top"], item["bottom"]]]

    return []


def extract_pdf_positions(item):
    """提取并规范化 PDF 段落的位置信息。

    这是模块对外的主要位置提取接口。它调用 _extract_raw_positions
    获取原始位置数据后，统一处理页码偏移（将 0-based 的页码修正为 1-based），
    并将所有坐标值转换为规范的 Python 数值类型。

    输出格式：[[page_number, left, right, top, bottom], ...]
    其中 page_number 为 1-based 整数页码。

    Args:
        item: 段落数据字典

    Returns:
        list: 规范化后的位置坐标列表
    """
    if not isinstance(item, dict):
        return []

    positions = _extract_raw_positions(item)

    # 获取参考页码，用于修正页码偏移
    ref_page_number = item.get("page_number")
    ref_page_number = int(ref_page_number) if isinstance(ref_page_number, (int, float)) else None
    if ref_page_number is not None and ref_page_number <= 0:
        ref_page_number += 1  # 将 0-based 页码转为 1-based

    normalized_positions = []
    for pos in positions:
        if not isinstance(pos, (list, tuple)) or len(pos) < 5:
            continue

        # 处理 page_number 可能是列表的情况（如 MinerU 等解析器的输出）
        page_number = pos[0][-1] if isinstance(pos[0], list) else pos[0]
        try:
            page_number = int(page_number)
            # 页码修正：与参考页码对齐，或将 0-based 转为 1-based
            if ref_page_number is not None and page_number == ref_page_number - 1:
                page_number = ref_page_number
            elif page_number <= 0:
                page_number += 1

            normalized_positions.append(
                [page_number, float(pos[1]), float(pos[2]), float(pos[3]), float(pos[4])]
            )
        except (TypeError, ValueError):
            continue

    return normalized_positions


def normalize_pdf_item_metadata(item):
    """规范化单个段落项的 PDF 位置元数据。

    提取位置信息并以统一的内部键名 _pdf_positions 存储，
    同时清理无效的位置数据。

    Args:
        item: 段落数据字典

    Returns:
        dict: 规范化后的段落数据（原地修改）
    """
    if not isinstance(item, dict):
        return item

    positions = extract_pdf_positions(item)
    if positions:
        item[PDF_POSITIONS_KEY] = positions
    else:
        item.pop(PDF_POSITIONS_KEY, None)
    return item


def normalize_pdf_items_metadata(items):
    """批量规范化 PDF 段落列表的位置元数据。

    对列表中的每个段落调用 normalize_pdf_item_metadata。

    Args:
        items: 段落数据字典列表

    Returns:
        list: 规范化后的段落列表（原地修改）
    """
    if not isinstance(items, list):
        return items
    for item in items:
        normalize_pdf_item_metadata(item)
    return items


def reorder_multi_column_bboxes(pdf_parser, bboxes, zoom=PDF_MULTI_COLUMN_ZOOM):
    """对多栏 PDF 文档的段落进行阅读顺序重排。

    检测文档是否为多栏布局（通过比较列宽和页面宽度），
    如果是多栏布局，则调用 sort_X_by_page 按 XY 阅读顺序重新排列段落。

    判断逻辑：
    - 提取所有 text 类型段落的列宽中位数
    - 如果列宽 < 页面宽度的一半，判定为多栏布局
    - 按页面分组，在每页内按从上到下、从左到右的顺序重排

    Args:
        pdf_parser: PDF 解析器实例（需提供 page_images 和 sort_X_by_page 方法）
        bboxes: 段落列表，每项需包含 x0、x1、page_number 字段
        zoom: 用于页面宽度计算的缩放倍数

    Returns:
        list: 重排后的段落列表（单栏时返回原列表）
    """
    # 仅对 text 类型的段落进行多栏检测
    text_boxes = [
        box
        for box in bboxes
        if box.get("layout_type") == "text"
        and all(box.get(key) is not None for key in ["x0", "x1", "page_number"])
    ]
    if not text_boxes or not pdf_parser.page_images:
        return bboxes

    # 计算文本列宽的中位数作为列宽估计值
    column_width = np.median([box["x1"] - box["x0"] for box in text_boxes])
    # 计算页面实际宽度（考虑缩放）
    page_width = pdf_parser.page_images[0].size[0] / zoom

    # 列宽 >= 页面一半 → 单栏布局，无需重排
    if column_width >= page_width / 2:
        return bboxes

    # 多栏布局，按 XY 阅读顺序重排
    return pdf_parser.sort_X_by_page(bboxes, column_width / 2)


def merge_pdf_positions(sources):
    """合并多个来源的 PDF 位置信息。

    将多个段落的位置坐标合并为一个列表，去重后按
    (页码, top, left) 排序。用于生成跨多个切块的合并预览。

    Args:
        sources: 位置来源列表，每项可以是 dict（段落数据）或 list（坐标列表）

    Returns:
        list: 去重并排序后的合并位置列表
    """
    merged = []
    seen = set()
    for source in sources or []:
        if isinstance(source, dict):
            positions = extract_pdf_positions(source)
        elif isinstance(source, list):
            positions = source
        else:
            positions = []

        for pos in positions:
            if not isinstance(pos, (list, tuple)) or len(pos) < 5:
                continue
            key = tuple(pos[:5])
            if key in seen:
                continue
            seen.add(key)
            merged.append(list(pos[:5]))

    # 按页码 → 顶部坐标 → 左侧坐标排序
    merged.sort(key=lambda item: (item[0], item[3], item[1]))
    return merged


def build_pdf_position_fields(positions):
    """从位置坐标列表构建搜索引擎索引所需的三个字段。

    生成以下字段：
    - position_int: 整数坐标元组列表 [(page_no, left, right, top, bottom), ...]
    - page_num_int: 页码整数列表
    - top_int: top 坐标整数列表

    这些字段会被存入 Elasticsearch/Infinity 索引中，用于支持
    基于位置的检索和排序。

    Args:
        positions: 位置坐标列表

    Returns:
        dict: 包含 position_int、page_num_int、top_int 三个字段的字典
    """
    position_int = []
    page_num_int = []
    top_int = []
    for pos in positions or []:
        if not isinstance(pos, (list, tuple)) or len(pos) < 5:
            continue
        try:
            page_no = int(pos[0])
            left = int(pos[1])
            right = int(pos[2])
            top = int(pos[3])
            bottom = int(pos[4])
        except (TypeError, ValueError):
            continue

        position_int.append((page_no, left, right, top, bottom))
        page_num_int.append(page_no)
        top_int.append(top)

    return {
        "position_int": deepcopy(position_int),
        "page_num_int": deepcopy(page_num_int),
        "top_int": deepcopy(top_int),
    }


def finalize_pdf_chunk(chunk):
    """完成 PDF 切块的最终处理。

    将 _pdf_positions 内部键展开为 position_int/page_num_int/top_int
    索引字段，然后移除临时的 _pdf_positions 键。
    此函数在切块即将写入搜索引擎索引之前调用。

    Args:
        chunk: PDF 切块数据字典

    Returns:
        dict: 处理后的切块数据
    """
    if not isinstance(chunk, dict):
        return chunk

    positions = extract_pdf_positions(chunk)
    if positions:
        chunk.update(build_pdf_position_fields(positions))
    # 移除临时存储的内部键
    chunk.pop(PDF_POSITIONS_KEY, None)
    return chunk


def _fetch_source_blob(from_upstream, canvas):
    """从上游数据或存储服务中获取 PDF 文件的二进制内容。

    优先通过 doc_id 从文档存储中获取，其次通过文件服务获取。

    Args:
        from_upstream: 上游传入的 ParserFromUpstream 数据
        canvas: 画布对象（提供 _doc_id 上下文）

    Returns:
        bytes | None: PDF 文件的二进制内容，获取失败返回 None
    """
    if canvas._doc_id:
        bucket, name = File2DocumentService.get_storage_address(doc_id=canvas._doc_id)
        return settings.STORAGE_IMPL.get(bucket, name)
    if from_upstream.file:
        return FileService.get_blob(from_upstream.file["created_by"], from_upstream.file["id"])
    return None


def _load_pdf_page_images(blob, zoom=PDF_PREVIEW_ZOOM):
    """加载 PDF 所有页面的渲染图像。

    使用 pdfplumber 将 PDF 每页渲染为 PIL Image，
    用于后续的预览图裁剪和区域高亮。

    Args:
        blob: PDF 文件的二进制内容
        zoom: 渲染缩放倍数，决定输出图像的分辨率

    Returns:
        list[PIL.Image]: 每页的渲染图像列表
    """
    with sys.modules[LOCK_KEY_pdfplumber]:
        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            return [
                page.to_image(resolution=72 * zoom, antialias=True).annotated
                for page in pdf.pages
            ]


def _crop_pdf_preview(page_images, positions, zoom=PDF_PREVIEW_ZOOM):
    """根据段落位置裁剪并生成 PDF 预览图。

    在段落区域的上下方各添加一段上下文，并将首尾的上下文区域
    做半透明遮罩处理，使高亮的正文区域在视觉上更为突出。

    预览图结构：
    ┌─────────────────────┐
    │  上下文（半透明）     │  ← 段落上方扩展区域
    ├─────────────────────┤
    │  段落正文（正常）     │  ← 高亮的主体区域
    ├─────────────────────┤
    │  上下文（半透明）     │  ← 段落下方扩展区域
    └─────────────────────┘

    Args:
        page_images: _load_pdf_page_images 返回的页面图像列表
        positions: 需要高亮的段落位置坐标列表
        zoom: 缩放倍数

    Returns:
        PIL.Image | None: 生成的预览图，无法生成时返回 None
    """
    if not page_images or not positions:
        return None

    # 规范化位置坐标：排序并过滤无效坐标
    normalized_positions = []
    for pos in sorted(positions, key=lambda item: (item[0], item[3], item[1])):
        if len(pos) < 5:
            continue

        page_idx = int(pos[0]) - 1  # 转为 0-based 页面索引
        if not (0 <= page_idx < len(page_images)):
            continue

        left, right, top, bottom = map(float, pos[1:5])
        if right <= left or bottom <= top:
            continue
        normalized_positions.append((page_idx, left, right, top, bottom))

    if not normalized_positions:
        return None

    # 确定裁剪区域的统一宽度（所有段落取最宽值）
    max_width = max(right - left for _, left, right, _, _ in normalized_positions)
    first_page, first_left, _, first_top, _ = normalized_positions[0]
    last_page, last_left, _, _, last_bottom = normalized_positions[-1]

    def page_height(idx):
        return page_images[idx].size[1] / zoom

    # 构建裁剪区域列表：首部上下文 + 正文段落 + 尾部上下文
    crop_positions = [
        # 首部上下文：从第一个段落向上扩展
        (
            [first_page],
            first_left,
            first_left + max_width,
            max(0, first_top - PDF_PREVIEW_CONTEXT),
            max(first_top - PDF_PREVIEW_GAP, 0),
        )
    ]
    # 正文段落区域
    crop_positions.extend(
        [
            ([page_idx], left, right, top, bottom)
            for page_idx, left, right, top, bottom in normalized_positions
        ]
    )
    # 尾部上下文：从最后一个段落向下扩展
    crop_positions.append(
        (
            [last_page],
            last_left,
            last_left + max_width,
            min(page_height(last_page), last_bottom + PDF_PREVIEW_GAP),
            min(page_height(last_page), last_bottom + PDF_PREVIEW_CONTEXT),
        )
    )

    # 逐区域裁剪页面图像
    imgs = []
    for idx, (pages, left, right, top, bottom) in enumerate(crop_positions):
        page_idx = pages[0]
        # 首尾区域使用统一宽度，正文区域保留原有宽度
        effective_right = (
            left + max_width if idx in {0, len(crop_positions) - 1} else max(left + 10, right)
        )
        imgs.append(
            page_images[page_idx].crop(
                (
                    left * zoom,
                    top * zoom,
                    effective_right * zoom,
                    min(bottom * zoom, page_images[page_idx].size[1]),
                )
            )
        )

    # 将裁剪的区域拼接为一张预览图
    canvas_height = int(sum(img.size[1] for img in imgs) + PDF_PREVIEW_GAP * len(imgs))
    canvas_width = int(max(img.size[0] for img in imgs))
    preview = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))

    height = 0
    for idx, img in enumerate(imgs):
        if idx in {0, len(imgs) - 1}:
            # 首尾上下文区域加半透明遮罩，突出中间的正文高亮区域
            img = img.convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay.putalpha(128)
            img = Image.alpha_composite(img, overlay).convert("RGB")

        preview.paste(img, (0, height))
        height += img.size[1] + PDF_PREVIEW_GAP

    return preview


async def restore_pdf_text_previews(chunks, from_upstream, canvas):
    """为 PDF 文本切块恢复/生成预览缩略图。

    对切块列表中所有 doc_type_kwd 为 "text" 的切块，
    根据其位置信息在原 PDF 上裁剪对应区域并上传为图片，
    将图片 ID 写回切块的 img_id 字段。

    使用缓存机制：相同位置区域的切块共享同一张预览图。

    Args:
        chunks: PDF 切块列表
        from_upstream: 上游传入的文件信息
        canvas: 画布对象（提供租户 ID 等上下文）
    """
    if not chunks or not str(from_upstream.name).lower().endswith(".pdf"):
        return

    # 仅处理 text 类型且有位置信息的切块
    text_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("doc_type_kwd", "text") == "text" and extract_pdf_positions(chunk)
    ]
    if not text_chunks:
        return

    blob = _fetch_source_blob(from_upstream, canvas)
    if not blob:
        return

    # 加载 PDF 页面图像
    try:
        page_images = _load_pdf_page_images(blob)
    except Exception as e:
        logging.warning(f"Failed to load PDF page images for chunk preview restore: {e}")
        return

    # 位置→图片ID 缓存，避免重复生成相同位置的预览图
    preview_cache = {}
    storage_put = partial(settings.STORAGE_IMPL.put, tenant_id=canvas._tenant_id)

    for chunk in text_chunks:
        preview_positions = extract_pdf_positions(chunk)
        positions_key = tuple(tuple(pos[:5]) for pos in preview_positions)
        if not positions_key:
            continue

        # 命中缓存：复用已生成的预览图 ID
        if positions_key in preview_cache:
            chunk["img_id"] = preview_cache[positions_key]
            continue

        preview = _crop_pdf_preview(page_images, preview_positions)
        if not preview:
            continue

        # 将预览图写入 chunk 并上传至对象存储
        chunk["image"] = preview
        await image2id(chunk, storage_put, get_uuid())
        if chunk.get("img_id"):
            preview_cache[positions_key] = chunk["img_id"]
