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
OpenDataLoader 解析器模块。

本模块提供基于 OpenDataLoader 服务的 PDF 文档解析功能。
OpenDataLoader 是一个外部文档解析服务，将 PDF 转换为结构化的 JSON 文档树，
支持表格、图片、公式、标题、段落等多种内容类型的识别。

主要功能：
- 通过 HTTP API 提交 PDF 文件到 OpenDataLoader 服务
- 解析返回的 JSON 文档树，提取结构化内容
- 支持 Markdown 文本回退（当 JSON 解析不可用时）
- 自动坐标转换（PDF 坐标系 -> 图片坐标系）
- 图片裁剪与位置标签生成
- 断线重连（最多 3 次重试）
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pdfplumber
import requests
from PIL import Image

from common.constants import MAXIMUM_PAGE_NUMBER

# 尝试导入 RAGFlowPdfParser 基类；若失败则定义一个占位类以保持模块独立性
try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except Exception:
    class RAGFlowPdfParser:
        pass

from deepdoc.parser.utils import extract_pdf_outlines


class OpenDataLoaderContentType(str, Enum):
    """
    OpenDataLoader 支持的内容类型枚举。

    用于分类 JSON 文档树中的每个元素。
    """
    IMAGE = "image"       # 图片
    TABLE = "table"       # 表格
    TEXT = "text"         # 文本
    EQUATION = "equation" # 公式


@dataclass
class _BBox:
    """
    边界框数据类，存储元素在 PDF 中的位置信息。

    坐标使用 PDF 坐标系（原点在左下角）。
    """
    page_no: int      # 页码（从 1 开始）
    x0: float         # 左边界
    y0: float         # 下边界
    x1: float         # 右边界
    y1: float         # 上边界


# 文本类内容类型集合
_TEXT_TYPES = {"heading", "title", "paragraph", "text", "list", "list_item", "caption"}
# 表格类内容类型集合
_TABLE_TYPES = {"table"}
# 图片类内容类型集合
_IMAGE_TYPES = {"image", "picture", "figure"}
# 公式类内容类型集合
_FORMULA_TYPES = {"formula", "equation"}


def _as_float(v) -> Optional[float]:
    """
    安全地将值转换为浮点数。

    Args:
        v: 待转换的值

    Returns:
        float 或 None（转换失败时）
    """
    try:
        return float(v)
    except Exception:
        return None


def _bbox_from_element(el: dict) -> Optional[_BBox]:
    """
    从 JSON 元素中提取边界框信息。

    支持多种常见的边界框字段名（bounding box / bounding_box / bbox）
    和多种页码字段名（page number / page_number / page）。

    OpenDataLoader 返回的坐标格式为 [left, bottom, right, top]，
    使用 PDF 点数（points）单位。

    Args:
        el: JSON 文档树中的元素字典

    Returns:
        _BBox 或 None（无法提取时）
    """
    bb = el.get("bounding box") or el.get("bounding_box") or el.get("bbox")
    pn = el.get("page number")
    if pn is None:
        pn = el.get("page_number")
    if pn is None:
        pn = el.get("page")
    if bb is None or pn is None:
        return None
    if not isinstance(bb, (list, tuple)) or len(bb) < 4:
        return None
    coords = [_as_float(x) for x in bb[:4]]
    if any(c is None for c in coords):
        return None
    try:
        page_no = int(pn)
    except Exception:
        return None

    # OpenDataLoader 输出 [left, bottom, right, top]（PDF 点数）
    left, bottom, right, top = coords
    x0, x1 = min(left, right), max(left, right)
    y0, y1 = min(bottom, top), max(bottom, top)
    return _BBox(page_no=page_no, x0=x0, y0=y0, x1=x1, y1=y1)


def _iter_elements(node: Any) -> Iterable[dict]:
    """
    递归遍历 JSON 文档树，生成所有内容元素。

    当节点同时包含 type 和 content/text/cells 字段时，
    认为它是一个内容元素。

    Args:
        node: JSON 节点（dict、list 或原子值）

    Yields:
        dict: 内容元素字典
    """
    if isinstance(node, dict):
        if "type" in node and ("content" in node or "text" in node or "cells" in node):
            yield node
        for v in node.values():
            yield from _iter_elements(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_elements(item)


def _element_text(el: dict) -> str:
    """
    从 JSON 元素中提取文本内容。

    优先级：content > text > cells（表格单元格连接）

    Args:
        el: 元素字典

    Returns:
        str: 提取的文本内容
    """
    content = el.get("content")
    if isinstance(content, str):
        return content
    text = el.get("text")
    if isinstance(text, str):
        return text
    # 表格可能通过 cells 暴露单元格数据，按行连接
    cells = el.get("cells")
    if isinstance(cells, list):
        rows: dict[int, list[str]] = {}
        for c in cells:
            if not isinstance(c, dict):
                continue
            row = c.get("row") or c.get("row_index") or 0
            rows.setdefault(int(row), []).append(str(c.get("content") or c.get("text") or ""))
        return "\n".join(" | ".join(v) for _, v in sorted(rows.items()))
    return ""


def _element_html(el: dict) -> str:
    """
    从 JSON 元素中提取 HTML 内容（通常用于表格）。

    Args:
        el: 元素字典

    Returns:
        str: HTML 字符串，无内容时返回空字符串
    """
    for key in ("html", "html_content"):
        v = el.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


class OpenDataLoaderParser(RAGFlowPdfParser):
    """
    OpenDataLoader 解析器。

    通过 OpenDataLoader 外部服务解析 PDF 文档。服务将 PDF 转换为
    结构化的 JSON 文档树（json_doc），本解析器负责：
    - 调用服务 API 提交 PDF
    - 解析 JSON 文档树，提取文本、表格、图片等元素
    - 坐标转换（PDF 坐标系 -> 图片坐标系）
    - 生成位置标签和图片裁剪

    配置方式：
    - OPENDATALOADER_APISERVER: 服务地址（必需）
    - OPENDATALOADER_API_KEY: API 密钥（可选）
    - OPENDATALOADER_TIMEOUT: 请求超时时间（默认 600 秒）
    """

    def __init__(self):
        """初始化 OpenDataLoader 解析器，从环境变量读取配置。"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.page_images: list[Image.Image] = []
        self.page_from = 0
        self.page_to = 10_000
        self.outlines = []
        self.api_url = os.environ.get("OPENDATALOADER_APISERVER", "").rstrip("/")
        self.api_key = os.environ.get("OPENDATALOADER_API_KEY", "").strip()
        try:
            self.timeout = int(os.environ.get("OPENDATALOADER_TIMEOUT", "600") or "600")
        except ValueError:
            self.logger.warning("[OpenDataLoader] Invalid OPENDATALOADER_TIMEOUT, falling back to 600s")
            self.timeout = 600

    def check_installation(self) -> bool:
        """
        检查 OpenDataLoader 服务是否可达。

        通过 GET /health 端点验证服务状态。

        Returns:
            bool: 服务可达返回 True，否则返回 False
        """
        if not self.api_url:
            self.logger.warning(
                "[OpenDataLoader] OPENDATALOADER_APISERVER is not set. "
                "Start the opendataloader service and set the env var."
            )
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = requests.get(f"{self.api_url}/health", timeout=5, headers=headers)
            if resp.status_code == 200:
                return True
            self.logger.warning(
                f"[OpenDataLoader] Health check returned {resp.status_code}: {resp.text[:200]}"
            )
            return False
        except Exception as exc:
            self.logger.warning(f"[OpenDataLoader] Health check failed: {exc}")
            return False

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        """
        从 PDF 渲染页面图片（本地渲染，供裁剪功能使用）。

        图片渲染在 RAGFlow 主机上完成；只有 PDF 转换在 OpenDataLoader 容器中运行。

        Args:
            fnm: PDF 文件路径或二进制数据
            zoomin: 缩放因子
            page_from: 起始页码
            page_to: 结束页码
            callback: 进度回调
        """
        self.page_from = page_from
        self.page_to = page_to
        bytes_io = None
        try:
            if not isinstance(fnm, (str, PathLike)):
                bytes_io = fnm if isinstance(fnm, BytesIO) else BytesIO(fnm)
            opener = pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(bytes_io)
            with opener as pdf:
                pages = pdf.pages[page_from:page_to]
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for p in pages]
        except Exception as e:
            self.page_images = []
            self.logger.exception(e)
        finally:
            if bytes_io:
                bytes_io.close()

    def _make_line_tag(self, bbox: _BBox) -> str:
        """
        根据边界框生成 RAGFlow 位置标签。

        将 OpenDataLoader 的 PDF 坐标系（原点左下角）转换为
        图片坐标系（原点左上角）。

        Args:
            bbox: 边界框对象

        Returns:
            str: 格式为 @@页码\t左\t右\t上\t下## 的位置标签
        """
        if bbox is None:
            return ""
        # 安全检查：仅当页面已被渲染时才生成裁剪标签
        if not self.page_images or bbox.page_no <= 0 or len(self.page_images) < bbox.page_no:
            return ""
        x0, x1 = bbox.x0, bbox.x1
        # OpenDataLoader 的 bbox 使用 PDF 坐标空间（原点左下角）。
        # 转换为图片坐标空间（原点左上角）：用页面高度减去 y 坐标
        _, page_height = self.page_images[bbox.page_no - 1].size
        top = page_height - bbox.y1
        bott = page_height - bbox.y0
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
            bbox.page_no, x0, x1, top, bott
        )

    @staticmethod
    def extract_positions(txt: str) -> list[tuple[list[int], float, float, float, float]]:
        """
        从文本中提取位置标签信息。

        Args:
            txt: 包含 @@页码\t坐标## 标签的文本

        Returns:
            list: [(页码列表, left, right, top, bottom), ...]
        """
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text: str, ZM: int = 1, need_position: bool = False):
        """
        根据位置标签从页面裁剪图片区域。

        支持跨页裁剪，将多页裁剪结果垂直拼接为一张图片。
        首尾区域添加半透明遮罩以标记非主要内容。

        Args:
            text: 包含位置标签的文本
            ZM: 缩放因子（未使用，保留接口兼容性）
            need_position: 是否同时返回位置信息

        Returns:
            PIL.Image 或 (PIL.Image, positions) 或 None
        """
        if not self.page_images:
            return (None, None) if need_position else None
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            return (None, None) if need_position else None
        # 丢弃超出渲染页面范围的位置
        max_page = len(self.page_images) - 1
        poss = [p for p in poss if all(0 <= pn <= max_page for pn in p[0])]
        if not poss:
            return (None, None) if need_position else None
        GAP = 6
        pos = poss[0]
        # 在首尾插入上下文填充区域
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1], pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1], pos[4] + 120)))
        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if bottom <= top:
                bottom = top + 4
            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))
            remain_bottom = bottom - img0.size[1]
            # 跨页裁剪
            for pn in pns[1:]:
                if remain_bottom <= 0:
                    break
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(remain_bottom, page.size[1]))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                remain_bottom -= page.size[1]
        if not imgs:
            return (None, None) if need_position else None
        # 垂直拼接所有裁剪图片
        height = sum(i.size[1] + GAP for i in imgs)
        width = max(i.size[0] for i in imgs)
        pic = Image.new("RGB", (width, int(height)), (245, 245, 245))
        h = 0
        for ii, img in enumerate(imgs):
            # 首尾图片添加半透明遮罩
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(h)))
            h += img.size[1] + GAP
        return (pic, positions) if need_position else pic

    def _cropout_region(self, bbox: _BBox, zoomin: int = 1):
        """
        根据边界框从页面图片中裁剪区域。

        用于表格和图片类型元素的缩略图生成。

        Args:
            bbox: 边界框（PDF 坐标）
            zoomin: 缩放因子

        Returns:
            tuple: (裁剪后的 PIL Image, [位置信息]) 或 (None, "")
        """
        if not self.page_images:
            return None, ""
        idx = (bbox.page_no - 1) - self.page_from
        if idx < 0 or idx >= len(self.page_images):
            return None, ""
        page_img = self.page_images[idx]
        W, H = page_img.size
        # PDF 坐标 -> 图片坐标转换
        x0 = max(0.0, min(float(bbox.x0), W - 1))
        y0 = max(0.0, min(float(H - bbox.y1), H - 1))
        x1 = max(x0 + 1.0, min(float(bbox.x1), W))
        y1 = max(y0 + 1.0, min(float(H - bbox.y0), H))
        try:
            crop = page_img.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
        except Exception:
            return None, ""
        pos = (bbox.page_no - 1 if bbox.page_no > 0 else 0, x0, x1, y0, y1)
        return crop, [pos]

    def _classify(self, el_type: str) -> str:
        """
        将元素类型字符串分类为标准内容类型。

        保留原始结构化类型（heading、title、paragraph 等），
        以便下游解析器应用标题/标题启发式规则。

        Args:
            el_type: 原始元素类型字符串

        Returns:
            str: 标准化的内容类型
        """
        t = (el_type or "").lower()
        if t in _TABLE_TYPES:
            return OpenDataLoaderContentType.TABLE.value
        if t in _IMAGE_TYPES:
            return OpenDataLoaderContentType.IMAGE.value
        if t in _FORMULA_TYPES:
            return OpenDataLoaderContentType.EQUATION.value
        # 保留原始结构化类型，以便下游应用标题检测等启发式规则
        return t if t else OpenDataLoaderContentType.TEXT.value

    def _transfer_from_json(self, root: Any, parse_method: str):
        """
        从 JSON 文档树转换内容为 sections 和 tables。

        遍历所有内容元素，根据类型分别处理：
        - 表格：提取 HTML 或文本，生成缩略图
        - 图片：提取标题，生成缩略图
        - 文本/标题/段落等：提取文本内容和位置标签

        Args:
            root: JSON 文档树的根节点
            parse_method: 解析方法 ("raw" / "manual" / "pipeline" / "paper")

        Returns:
            tuple: (sections 列表, tables 列表)
        """
        sections: list[tuple[str, ...]] = []
        tables: list = []
        for el in _iter_elements(root):
            el_type = self._classify(el.get("type", ""))
            bbox = _bbox_from_element(el)
            tag = self._make_line_tag(bbox) if bbox else ""

            if el_type == OpenDataLoaderContentType.TABLE.value:
                html = _element_html(el) or _element_text(el)
                img = None
                positions = ""
                if bbox:
                    img, positions = self._cropout_region(bbox)
                tables.append(((img, html), positions if positions else ""))
                continue

            if el_type == OpenDataLoaderContentType.IMAGE.value:
                img = None
                positions = ""
                if bbox:
                    img, positions = self._cropout_region(bbox)
                caption = _element_text(el)
                tables.append(((img, [caption] if caption else [""]), positions if positions else ""))
                continue

            text = _element_text(el).strip()
            if not text:
                continue
            if parse_method in {"manual", "pipeline"}:
                sections.append((text, el_type, tag))
            elif parse_method == "paper":
                sections.append((text + tag, el_type))
            else:
                sections.append((text, tag))
        return sections, tables

    @staticmethod
    def _sections_from_markdown(md: str, parse_method: str) -> list[tuple[str, ...]]:
        """
        从 Markdown 文本创建 sections（当服务未返回 JSON 文档树时的回退方案）。

        Args:
            md: Markdown 文本
            parse_method: 解析方法

        Returns:
            list[tuple]: sections 列表
        """
        txt = (md or "").strip()
        if not txt:
            return []
        if parse_method in {"manual", "pipeline"}:
            return [(txt, OpenDataLoaderContentType.TEXT.value, "")]
        if parse_method == "paper":
            return [(txt, OpenDataLoaderContentType.TEXT.value)]
        return [(txt, "")]

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        parse_method: str = "raw",
        hybrid: Optional[str] = None,
        image_output: Optional[str] = None,
        sanitize: Optional[bool] = None,
    ):
        """
        解析 PDF 文档（主入口方法）。

        完整流程：
        1. 提取 PDF 目录大纲
        2. 本地渲染页面缩略图
        3. 上传 PDF 到 OpenDataLoader 服务（最多重试 3 次）
        4. 优先解析 JSON 文档树，回退到 Markdown 文本
        5. 返回标准化的 sections 和 tables

        Args:
            filepath: PDF 文件路径
            binary: PDF 二进制数据
            callback: 进度回调 (progress, message)
            parse_method: 解析方法，可选 "raw"、"manual"、"pipeline"、"paper"
            hybrid: 混合解析模式参数
            image_output: 图片输出选项
            sanitize: 是否清理输出

        Returns:
            tuple: (sections 列表, tables 列表)
        """
        self.outlines = extract_pdf_outlines(binary if binary is not None else filepath)

        if not self.api_url:
            raise RuntimeError(
                "[OpenDataLoader] OPENDATALOADER_APISERVER is not configured. "
                "Please start the opendataloader service and set the env var."
            )

        # 本地渲染页面图片 — 供 _make_line_tag() 和 crop() 使用
        # 图片渲染在 RAGFlow 主机上完成；仅 PDF 转换在容器中运行
        try:
            if binary is not None:
                src = BytesIO(binary) if isinstance(binary, (bytes, bytearray)) else binary
                self.__images__(src, zoomin=1)
            else:
                self.__images__(str(filepath), zoomin=1)
        except Exception as e:
            self.logger.warning(f"[OpenDataLoader] render pages failed: {e}")

        # 读取 PDF 字节数据用于 multipart 上传
        if binary is not None:
            pdf_bytes = binary if isinstance(binary, (bytes, bytearray)) else binary.getvalue()
        else:
            with open(filepath, "rb") as fh:
                pdf_bytes = fh.read()

        filename = Path(str(filepath)).name or "input.pdf"

        if callback:
            callback(0.1, f"[OpenDataLoader] Sending '{filename}' to service")

        # 构建表单数据
        form_data: dict[str, str] = {}
        if hybrid:
            form_data["hybrid"] = hybrid
        if image_output:
            form_data["image_output"] = image_output
        if sanitize is not None:
            form_data["sanitize"] = "true" if sanitize else "false"

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last_exc: Exception | None = None
        # 最多重试 3 次
        for attempt in range(1, 4):
            try:
                self.logger.info(f"[OpenDataLoader] POST {self.api_url}/file_parse for '{filename}' (attempt {attempt})")
                resp = requests.post(
                    url=f"{self.api_url}/file_parse",
                    files={"file": (filename, pdf_bytes, "application/pdf")},
                    data=form_data,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                self.logger.warning(f"[OpenDataLoader] attempt {attempt} failed: {exc}")
        else:
            raise RuntimeError(f"[OpenDataLoader] service call failed after 3 attempts: {last_exc}") from last_exc

        if callback:
            callback(0.7, "[OpenDataLoader] Processing response")

        # 服务响应结构：
        # {
        #   "json_doc": {...} | null,   # 结构化解析树（优先使用）
        #   "md_text":  "..." | null    # Markdown 回退（json_doc 不可用时）
        # }
        json_doc = result.get("json_doc")
        md_text = result.get("md_text")

        sections: list[tuple[str, ...]] = []
        tables: list = []
        # 优先使用 JSON 文档树
        if json_doc is not None:
            sections, tables = self._transfer_from_json(json_doc, parse_method=parse_method)
        # 回退到 Markdown 文本
        if not sections and md_text:
            sections = self._sections_from_markdown(md_text, parse_method=parse_method)

        if callback:
            callback(1.0, f"[OpenDataLoader] Done. Sections: {len(sections)}, Tables: {len(tables)}")

        return sections, tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = OpenDataLoaderParser()
    print("OpenDataLoader service reachable:", parser.check_installation())
