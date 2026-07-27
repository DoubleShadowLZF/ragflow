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

"""文档解析器模块。

Parser 是 RAGFlow 文档处理流水线的核心组件，负责将各种格式的原始文档
解析为统一的结构化输出（JSON/Markdown/Text/HTML）。

支持的文档类型及解析策略：
┌─────────────────┬──────────────────────────────────────────────┐
│ 文档类型         │ 解析策略                                      │
├─────────────────┼──────────────────────────────────────────────┤
│ PDF             │ DeepDOC / PlainText / MinerU / Docling /     │
│                 │ OpenDataLoader / TCADP / PaddleOCR / VLM     │
│ 电子表格         │ DeepDOC / TCADP                              │
│ 演示文稿         │ DeepDOC / TCADP                              │
│ DOC/DOCX        │ Tika(DOC) / python-docx(DOCX)                │
│ Markdown        │ 自定义 Markdown 解析器                        │
│ 文本/代码        │ 内置 TxtParser                               │
│ HTML            │ 内置 HtmlParser                               │
│ 图片             │ OCR 识别 / VLM 视觉语言模型描述               │
│ 音频             │ 语音转文字模型 (Speech2Text)                   │
│ 视频             │ 视觉语言模型帧分析                             │
│ 邮件             │ EML/MSG 格式解析                              │
│ EPUB            │ 内置 EpubParser                               │
└─────────────────┴──────────────────────────────────────────────┘
"""

import asyncio
import io
import json
import os
import random
import re
from functools import partial

from litellm import logging
import numpy as np
from PIL import Image

from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.llm_service import LLMBundle
from api.db.joint_services.tenant_model_service import (
    ensure_mineru_from_env,
    ensure_opendataloader_from_env,
    ensure_paddleocr_from_env,
    get_first_provider_model_name,
    get_model_config_from_provider_instance,
    get_tenant_default_model_type_by_type,
)
from common import settings
from common.constants import LLMType
from common.misc_utils import get_uuid, thread_pool_exec
from deepdoc.parser import ExcelParser, HtmlParser, TxtParser
from deepdoc.parser.docling_parser import DoclingParser
from deepdoc.parser.pdf_parser import PlainParser, RAGFlowPdfParser, VisionParser
from deepdoc.parser.tcadp_parser import TCADPParser
from rag.app.naive import Docx
from rag.flow.base import ProcessBase, ProcessParamBase
from rag.flow.parser.pdf_chunk_metadata import (
    extract_pdf_positions,
    normalize_pdf_items_metadata,
    reorder_multi_column_bboxes,
)
from rag.flow.parser.schema import ParserFromUpstream
from rag.flow.parser.utils import (
    enhance_media_sections_with_vision,
    extract_word_outlines,
    extract_docx_header_footer_texts,
    remove_header_footer_docx_sections,
    remove_header_footer_html_blob,
    remove_toc,
    remove_toc_pdf,
    remove_toc_word,
)
from rag.llm.cv_model import Base as VLM
from rag.utils.base64_image import image2id


class ParserParam(ProcessParamBase):
    """Parser 组件的参数配置类。

    定义所有支持文档类型的解析参数，包括：
    - 每种文档类型允许的输出格式
    - 默认解析方法和解析选项
    - 参数合法性校验逻辑
    """

    def __init__(self):
        super().__init__()

        # === 每种文档类型允许的输出格式 ===
        self.allowed_output_format = {
            "pdf": [
                "json",
                "markdown",
            ],
            "spreadsheet": [
                "json",
                "markdown",
                "html",
            ],
            "doc": [
                "json",
                "markdown",
            ],
            "docx": [
                "json",
                "markdown",
            ],
            "slides": [
                "json",
            ],
            "image": [
                "json",
            ],
            "email": [
                "text",
                "json",
            ],
            "markdown": [
                "text",
                "json",
            ],
            "text&code": [
                "text",
                "json",
            ],
            "html": [
                "text",
                "json",
            ],
            "audio": [
                "json",
            ],
            "video": [],
            "epub": [
                "text",
                "json",
            ],
        }

        # === 每种文档类型的默认解析配置 ===
        self.setups = {
            "pdf": {
                "parse_method": "deepdoc",  # 解析方式：deepdoc/plain_text/tcadp_parser/vlm
                "lang": "Chinese",          # 文档语言（影响 OCR 和 VLM 的识别效果）
                "flatten_media_to_text": False,  # 是否将图片表格等媒体全部视为纯文本
                "remove_toc": False,             # 是否移除目录
                "remove_header_footer": False,   # 是否移除非正文页眉页脚
                "suffix": [
                    "pdf",
                ],
                "output_format": "json",
            },
            "spreadsheet": {
                "parse_method": "deepdoc",  # 解析方式：deepdoc/tcadp_parser
                "flatten_media_to_text": False,
                "output_format": "html",
                "suffix": [
                    "xls",
                    "xlsx",
                    "csv",
                ],
            },
            "doc": {
                "remove_toc": False,
                "remove_header_footer": False,
                "suffix": [
                    "doc",
                ],
                "output_format": "json",
            },
            "docx": {
                "flatten_media_to_text": False,
                "remove_toc": False,
                "remove_header_footer": False,
                "suffix": [
                    "docx",
                ],
                "output_format": "json",
            },
            "markdown": {
                "flatten_media_to_text": False,
                "suffix": ["md", "markdown", "mdx"],
                "remove_toc": False,
                "output_format": "json",
            },
            "text&code": {
                "suffix": [
                    "txt",
                    "py",
                    "js",
                    "java",
                    "c",
                    "cpp",
                    "h",
                    "php",
                    "go",
                    "ts",
                    "sh",
                    "cs",
                    "kt",
                    "sql",
                ],
                "output_format": "json",
            },
            "html": {
                "suffix": ["htm", "html"],
                "remove_toc": False,
                "remove_header_footer": False,
                "output_format": "json",
            },
            "slides": {
                "parse_method": "deepdoc",  # 解析方式：deepdoc/tcadp_parser
                "suffix": [
                    "pptx",
                    "ppt",
                ],
                "output_format": "json",
            },
            "image": {
                "parse_method": "ocr",      # 解析方式：ocr（字符识别）/ VLM（视觉描述）
                "llm_id": "",
                "lang": "Chinese",
                "system_prompt": "",
                "suffix": ["jpg", "jpeg", "png", "gif"],
                "output_format": "json",
            },
            "email": {
                "suffix": [
                    "eml",
                    "msg",
                ],
                # 邮件中需要提取的字段列表
                "fields": ["from", "to", "cc", "bcc", "date", "subject", "body", "attachments", "metadata"],
                "output_format": "json",
            },
            "audio": {
                "suffix": [
                    "da",
                    "wave",
                    "wav",
                    "mp3",
                    "aac",
                    "flac",
                    "ogg",
                    "aiff",
                    "au",
                    "midi",
                    "wma",
                    "realaudio",
                    "vqf",
                    "oggvorbis",
                    "ape",
                ],
                "output_format": "text",
            },
            "video": {
                "suffix": [
                    "mp4",
                    "avi",
                    "mkv",
                ],
                "output_format": "text",
                "prompt": "",
            },
            "epub": {
                "suffix": [
                    "epub",
                ],
                "output_format": "json",
            },
        }

    def check(self):
        """校验所有文档类型的解析参数是否合法。

        对每种启用的文档类型进行参数检查：
        - 解析方法是否为空
        - 输出格式是否在允许范围内
        - 模型配置是否完整
        校验不通过时抛出异常。
        """
        # PDF 参数校验
        pdf_config = self.setups.get("pdf", {})
        if pdf_config:
            pdf_parse_method = pdf_config.get("parse_method", "")
            self.check_empty(pdf_parse_method, "Parse method abnormal.")

            # 非内置方法需要验证语言参数（VLM 依赖语言提示）
            if pdf_parse_method.lower() not in ["deepdoc", "plain_text", "mineru", "docling", "opendataloader", "tcadp parser", "paddleocr"]:
                self.check_empty(pdf_config.get("lang", ""), "PDF VLM language")

            pdf_output_format = pdf_config.get("output_format", "")
            self.check_valid_value(pdf_output_format, "PDF output format abnormal.", self.allowed_output_format["pdf"])

        # 电子表格参数校验
        spreadsheet_config = self.setups.get("spreadsheet", "")
        if spreadsheet_config:
            spreadsheet_output_format = spreadsheet_config.get("output_format", "")
            self.check_valid_value(spreadsheet_output_format, "Spreadsheet output format abnormal.", self.allowed_output_format["spreadsheet"])

        # DOC 参数校验
        doc_config = self.setups.get("doc", "")
        if doc_config:
            doc_output_format = doc_config.get("output_format", "")
            self.check_valid_value(doc_output_format, "DOC output format abnormal.", self.allowed_output_format["doc"])

        # DOCX 参数校验
        docx_config = self.setups.get("docx", "")
        if docx_config:
            docx_output_format = doc_config.get("output_format", "")
            self.check_valid_value(docx_output_format, "DOCX output format abnormal.", self.allowed_output_format["docx"])

        # 演示文稿参数校验
        slides_config = self.setups.get("slides", "")
        if slides_config:
            slides_output_format = slides_config.get("output_format", "")
            self.check_valid_value(slides_output_format, "Slides output format abnormal.", self.allowed_output_format["slides"])

        # 图片参数校验
        image_config = self.setups.get("image", "")
        if image_config:
            image_parse_method = image_config.get("parse_method", "")
            if image_parse_method not in ["ocr"]:
                self.check_empty(image_config.get("lang", ""), "Image VLM language")

        # Markdown 参数校验
        text_config = self.setups.get("markdown", "")
        if text_config:
            text_output_format = text_config.get("output_format", "")
            self.check_valid_value(text_output_format, "Markdown output format abnormal.", self.allowed_output_format["markdown"])

        # 文本/代码参数校验
        code_config = self.setups.get("text&code", "")
        if code_config:
            code_output_format = code_config.get("output_format", "")
            self.check_valid_value(code_output_format, "Text&Code output format abnormal.", self.allowed_output_format["text&code"])

        # HTML 参数校验
        html_config = self.setups.get("html", "")
        if html_config:
            html_output_format = html_config.get("output_format", "")
            self.check_valid_value(html_output_format, "HTML output format abnormal.", self.allowed_output_format["html"])

        # 音频参数校验——必须有 STT 模型
        audio_config = self.setups.get("audio", "")
        if audio_config:
            audio_vlm = audio_config.get("vlm") or {}
            self.check_empty(audio_vlm.get("llm_id"), "Audio VLM")

        # 视频参数校验——必须有 VLM 模型
        video_config = self.setups.get("video", "")
        if video_config:
            video_vlm = video_config.get("vlm") or {}
            self.check_empty(video_vlm.get("llm_id"), "Video VLM")

        email_config = self.setups.get("email", "")
        if email_config:
            email_output_format = email_config.get("output_format", "")
            self.check_valid_value(email_output_format, "Email output format abnormal.", self.allowed_output_format["email"])

        # EPUB 参数校验
        epub_config = self.setups.get("epub", "")
        if epub_config:
            epub_output_format = epub_config.get("output_format", "")
            self.check_valid_value(epub_output_format, "EPUB output format abnormal.", self.allowed_output_format["epub"])

    def get_input_form(self) -> dict[str, dict]:
        """获取前端输入表单配置（当前无额外表单字段）。"""
        return {}


class Parser(ProcessBase):
    """文档解析器组件。

    流水线中的核心组件之一，根据文件扩展名自动选择合适的解析方法，
    将原始文档二进制内容转换为结构化的文本/图片/表格段落列表。

    组件名称：Parser
    输出内容：json（结构化段落列表）、markdown、text、html 等
    """

    component_name = "Parser"

    def _pdf(self, name, blob, **kwargs):
        """解析 PDF 文件为结构化段落（bboxes）或 markdown/json 输出。

        支持多种解析策略：
        - deepdoc:     内置 RAGFlowPdfParser，基于视觉模型的版面分析
        - plain_text:  仅提取文本行（无版面结构）
        - MinerU:      外部 MinerU 服务（OCR + 版面分析）
        - Docling:     外部 Docling 服务
        - OpenDataLoader: 外部 OpenDataLoader 服务
        - TCADP:       腾讯云文档解析 API
        - PaddleOCR:   外部 PaddleOCR 服务
        - VLM:         直接使用视觉语言模型按页解析

        处理流程：
        1. 选择解析器并解析 PDF → 2. 提取大纲/移除目录 → 3. 规范化元数据
        → 4. 页眉页脚过滤 → 5. 媒体类型标记 → 6. VLM 增强媒体描述
        → 7. 输出格式化

        Args:
            name: 文件名（含扩展名）
            blob: 文件二进制内容
            **kwargs: 上游传递的额外参数（如 file 元信息）
        """
        # 初始化进度回调
        self.callback(random.randint(1, 5) / 100.0, "Start to work on a PDF.")
        conf = self._param.setups["pdf"]
        self.set_output("output_format", conf["output_format"])
        flatten_media_to_text = conf.get("flatten_media_to_text")
        pdf_parser = None

        # === 步骤1：解析解析方法并确定模型名称 ===
        # 支持 "@mineru" / "@paddleocr" 后缀语法来指定特定模型
        raw_parse_method = conf.get("parse_method", "")
        parser_model_name = None
        parse_method = raw_parse_method
        parse_method = parse_method or ""
        if isinstance(raw_parse_method, str):
            lowered = raw_parse_method.lower()
            if lowered.endswith("@mineru"):
                parser_model_name = raw_parse_method
                parse_method = "MinerU"
            elif lowered.endswith("@paddleocr"):
                parser_model_name = raw_parse_method
                parse_method = "PaddleOCR"

        # === 步骤2：按解析方法分发 ===

        # DeepDOC：内置解析器，直接返回结构化页面 boxes
        if parse_method.lower() == "deepdoc":
            pdf_parser = RAGFlowPdfParser()
            bboxes = pdf_parser.parse_into_bboxes(blob, callback=self.callback)
            # 多栏检测与重排
            if conf.get("enable_multi_column"):
                bboxes = reorder_multi_column_bboxes(pdf_parser, bboxes)

        # PlainText：仅提取文本行，忽略版面结构
        elif parse_method.lower() == "plain_text":
            pdf_parser = PlainParser()
            lines, _ = pdf_parser(blob)
            bboxes = [{"text": t, "layout_type": "text"} for t, _ in lines]

        # MinerU：外部 OCR + 版面分析服务，返回行级段落
        elif parse_method.lower() == "mineru":

            def resolve_mineru_llm_name():
                """解析 MinerU 模型名称：配置 > 租户默认 > 环境变量兜底"""
                configured = parser_model_name or conf.get("mineru_llm_name")
                if configured:
                    return configured

                tenant_id = self._canvas._tenant_id
                if not tenant_id:
                    return None

                return get_first_provider_model_name(tenant_id, "MinerU", LLMType.OCR) or ensure_mineru_from_env(tenant_id)

            parser_model_name = resolve_mineru_llm_name()
            if not parser_model_name:
                raise RuntimeError("MinerU model not configured. Please add MinerU in Model Providers or set MINERU_* env.")

            # 构建 OCR 模型实例
            tenant_id = self._canvas._tenant_id
            ocr_model_config = get_model_config_from_provider_instance(tenant_id, LLMType.OCR, parser_model_name)
            ocr_model = LLMBundle(tenant_id, ocr_model_config, lang=conf.get("lang", "Chinese"))
            pdf_parser = ocr_model.mdl

            # 调用 MinerU 的 pipeline 模式解析
            lines, _ = pdf_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                parse_method="pipeline",
                lang=conf.get("lang", "Chinese"),
            )
            bboxes = []
            for line in lines or []:
                if not isinstance(line, tuple) or len(line) < 3:
                    continue

                t, layout_type, poss = line[0], line[1], line[2]
                box = {
                    "text": t,
                    "layout_type": layout_type or "text",
                }
                # 提取位置坐标（页码从 1 开始）
                positions = [[pos[0][-1] + 1, *pos[1:]] for pos in pdf_parser.extract_positions(poss)]
                if positions:
                    box["positions"] = positions
                # 裁剪对应区域的图片（用于后续 VLM 增强识别）
                image = pdf_parser.crop(poss, 1)
                if image is not None:
                    box["image"] = image
                bboxes.append(box)

        # Docling：IBM 开源的文档解析工具
        elif parse_method.lower() == "docling":
            pdf_parser = DoclingParser(docling_server_url=os.environ.get("DOCLING_SERVER_URL", ""))
            lines, _ = pdf_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                parse_method="pipeline",
                docling_server_url=os.environ.get("DOCLING_SERVER_URL", ""),
            )
            bboxes = []
            for item in lines or []:
                if not isinstance(item, tuple) or len(item) < 3:
                    continue
                text, layout_type, poss = item[0], item[1], item[2]
                box = {
                    "text": text,
                    "layout_type": layout_type or "text",
                }
                if isinstance(poss, str) and poss:
                    positions = [[pos[0][-1] + 1, *pos[1:]] for pos in pdf_parser.extract_positions(poss)]
                    if positions:
                        box["positions"] = positions
                    image = pdf_parser.crop(poss, 1)
                    if image is not None:
                        box["image"] = image
                bboxes.append(box)

        # OpenDataLoader：外部文档数据加载器
        elif parse_method.lower() == "opendataloader":

            def resolve_opendataloader_llm_name():
                """解析 OpenDataLoader 模型名称"""
                configured = parser_model_name or conf.get("opendataloader_llm_name")
                if configured:
                    return configured
                tenant_id = self._canvas._tenant_id
                if not tenant_id:
                    return None

                return get_first_provider_model_name(tenant_id, "OpenDataLoader", LLMType.OCR) or ensure_opendataloader_from_env(tenant_id)

            parser_model_name = resolve_opendataloader_llm_name()
            if not parser_model_name:
                raise RuntimeError("OpenDataLoader model not configured. Please add OpenDataLoader in Model Providers.")

            tenant_id = self._canvas._tenant_id
            ocr_model_config = get_model_config_from_provider_instance(tenant_id, LLMType.OCR, parser_model_name)
            ocr_model = LLMBundle(tenant_id, ocr_model_config)
            pdf_parser = ocr_model.mdl

            # OpenDataLoader 返回两个值：lines（文本行）和 odl_tables（表格列表）
            lines, odl_tables = pdf_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                parse_method="pipeline",
            )
            bboxes = []
            for item in lines or []:
                if not isinstance(item, tuple) or len(item) < 3:
                    continue
                text, layout_type, poss = item[0], item[1], item[2]
                box = {
                    "text": text,
                    "layout_type": layout_type or "text",
                }
                if isinstance(poss, str) and poss:
                    positions = [[pos[0][-1] + 1, *pos[1:]] for pos in pdf_parser.extract_positions(poss)]
                    if positions:
                        box["positions"] = positions
                    image = pdf_parser.crop(poss, 1)
                    if image is not None:
                        box["image"] = image
                bboxes.append(box)

            # 合并 OpenDataLoader 的第二个返回值：表格和图片数据
            for (img, html_or_caption), positions in odl_tables or []:
                # 区分表格（str 类型描述）和图片（list 类型描述）
                box = {"layout_type": "table" if not isinstance(html_or_caption, list) else "figure"}
                if isinstance(html_or_caption, str):
                    box["text"] = html_or_caption
                elif isinstance(html_or_caption, list):
                    box["text"] = html_or_caption[0] if html_or_caption else ""
                if img is not None:
                    box["image"] = img
                if positions:
                    try:
                        box["positions"] = [[p[0] + 1, p[1], p[2], p[3], p[4]] for p in positions]
                    except Exception:
                        pass
                bboxes.append(box)

        # TCADP：腾讯云文档解析（ADP = Auto Document Processing）
        elif parse_method.lower() == "tcadp parser":
            table_result_type = conf.get("table_result_type", "1")
            markdown_image_response_type = conf.get("markdown_image_response_type", "1")
            pdf_parser = TCADPParser(
                table_result_type=table_result_type,
                markdown_image_response_type=markdown_image_response_type,
            )
            sections, _ = pdf_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                file_type="PDF",
                file_start_page=1,
                file_end_page=1000,
            )
            bboxes = []
            for section, position_tag in sections:
                if position_tag:
                    # 解析 TCADP 的位置标签格式：@@页码\tx0\tx1\ttop\tbottom##
                    match = re.match(r"@@([0-9-]+)\t([0-9.]+)\t([0-9.]+)\t([0-9.]+)\t([0-9.]+)##", position_tag)
                    if match:
                        pn, x0, x1, top, bott = match.groups()
                        bboxes.append(
                            {
                                "page_number": int(pn.split("-")[0]),
                                "x0": float(x0),
                                "x1": float(x1),
                                "top": float(top),
                                "bottom": float(bott),
                                "text": section,
                                "layout_type": "text",
                            }
                        )
                    else:
                        bboxes.append({"text": section, "layout_type": "text"})
                else:
                    bboxes.append({"text": section, "layout_type": "text"})

        # PaddleOCR：百度开源的 OCR 引擎
        elif parse_method.lower() == "paddleocr":

            def resolve_paddleocr_llm_name():
                """解析 PaddleOCR 模型名称"""
                configured = parser_model_name or conf.get("paddleocr_llm_name")
                if configured:
                    return configured

                tenant_id = self._canvas._tenant_id
                if not tenant_id:
                    return None

                return get_first_provider_model_name(tenant_id, "PaddleOCR", LLMType.OCR) or ensure_paddleocr_from_env(tenant_id)

            parser_model_name = resolve_paddleocr_llm_name()
            if not parser_model_name:
                raise RuntimeError("PaddleOCR model not configured. Please add PaddleOCR in Model Providers or set PADDLEOCR_* env.")

            tenant_id = self._canvas._tenant_id
            ocr_model_config = get_model_config_from_provider_instance(tenant_id, LLMType.OCR, parser_model_name)
            ocr_model = LLMBundle(tenant_id, ocr_model_config)
            pdf_parser = ocr_model.mdl

            lines, _ = pdf_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                parse_method="pipeline",
            )
            bboxes = []
            for line in lines or []:
                if not isinstance(line, tuple) or len(line) < 3:
                    continue

                t, layout_type, poss = line[0], line[1], line[2]
                box = {
                    "text": t,
                    "layout_type": layout_type or "text",
                }
                positions = [[pos[0][-1] + 1, *pos[1:]] for pos in pdf_parser.extract_positions(poss)]
                if positions:
                    box["positions"] = positions
                image = pdf_parser.crop(poss)
                if image is not None:
                    box["image"] = image
                bboxes.append(box)

        # VLM（视觉语言模型）解析：将每页作为大图输入 VLM 进行描述
        else:
            if conf.get("parse_method"):
                # 使用指定的 IMAGE2TEXT 模型
                vision_model_config = get_model_config_from_provider_instance(self._canvas._tenant_id, LLMType.IMAGE2TEXT, conf["parse_method"])
            else:
                # 使用租户默认的 IMAGE2TEXT 模型
                vision_model_config = get_tenant_default_model_by_type(self._canvas._tenant_id, LLMType.IMAGE2TEXT)
            vision_model = LLMBundle(self._canvas._tenant_id, vision_model_config, lang=self._param.setups["pdf"].get("lang"))
            pdf_parser = VisionParser(vision_model=vision_model)
            lines, _ = pdf_parser(blob, callback=self.callback)
            bboxes = []
            for t, poss in lines:
                for pn, x0, x1, top, bott in RAGFlowPdfParser.extract_positions(poss):
                    bboxes.append(
                        {
                            "page_number": int(pn[0]) + 1,
                            "x0": float(x0),
                            "x1": float(x1),
                            "top": float(top),
                            "bottom": float(bott),
                            "text": t,
                            "layout_type": "text",
                        }
                    )

        # === 步骤3：持久化 PDF 大纲信息 ===
        self.set_output("file", {**kwargs.get("file", {}), "outlines": pdf_parser.outlines})

        # === 步骤4：目录移除 ===
        if conf.get("remove_toc"):
            if not pdf_parser.outlines:
                # 无大纲时使用通用 NLP 目录检测
                bboxes, _ = remove_toc(bboxes)
            elif pdf_parser.outlines[0][2] == 1:
                # 大纲从第1页开始（PDF 页码从 1 开始），用 PDF 专用方法
                bboxes = remove_toc_pdf(bboxes, pdf_parser.outlines)
            else:
                # 大纲从其他页开始，分段处理
                first_outline_page = pdf_parser.outlines[0][2]
                split_at = len(bboxes)
                # 找到大纲起始页在 bboxes 中的位置
                for i, item in enumerate(bboxes):
                    page_number = item.get("page_number")
                    if page_number is None:
                        positions = extract_pdf_positions(item)
                        if positions:
                            page_number = positions[0][0]
                    if page_number is not None and page_number >= first_outline_page:
                        split_at = i
                        break
                # 仅对目录之前的段落做目录移除
                toc_bboxes, _ = remove_toc(bboxes[:split_at])
                bboxes = toc_bboxes + bboxes[split_at:]

        # === 步骤5：规范化元数据（layout_type 和 doc_type_kwd） ===
        normalize_bboxes = []
        for b in bboxes:
            raw_layout = str(b.get("layout_type") or "").strip()
            has_layout = bool(raw_layout)
            # 空白规范化，默认为 text 类型
            layout = re.sub(r"\s+", " ", raw_layout) if has_layout else "text"
            b["layout_type"] = layout

            # 页眉/页脚过滤
            if conf.get("remove_header_footer") and re.search(r"(header|footer|number)", raw_layout, re.I):
                continue

            # 根据 layout_type 和 flatten_media_to_text 标记文档类型关键字
            if flatten_media_to_text:
                b["doc_type_kwd"] = "text"
            elif layout == "table":
                b["doc_type_kwd"] = "table"
            elif layout == "figure":
                b["doc_type_kwd"] = "image"
            elif not has_layout and b.get("image") is not None:
                b["doc_type_kwd"] = "image"
            else:
                b["doc_type_kwd"] = "text"
            normalize_bboxes.append(b)
        bboxes = normalize_bboxes

        # === 步骤6：VLM 增强图片/表格描述 ===
        enhance_media_sections_with_vision(
            bboxes,
            self._canvas._tenant_id,
            conf.get("vlm"),
            callback=self.callback,
        )

        # === 步骤7：按输出格式生成最终结果 ===
        if conf.get("output_format") == "json":
            normalize_pdf_items_metadata(bboxes)
            self.set_output("json", bboxes)
        if conf.get("output_format") == "markdown":
            mkdn = ""
            for b in bboxes:
                if b.get("layout_type", "") == "title":
                    mkdn += "\n## "
                if b.get("layout_type", "") == "figure":
                    mkdn += "\n![Image]({})".format(VLM.image2base64(b["image"]))
                    continue
                mkdn += b.get("text", "") + "\n"
            self.set_output("markdown", mkdn)

    def _spreadsheet(self, name, blob, **kwargs):
        """解析电子表格文件并输出 html/json/markdown。

        支持两种解析器：
        - DeepDOC (ExcelParser)：默认，适用于标准 Excel 文件
        - TCADP：腾讯云文档解析 API

        Args:
            name: 文件名
            blob: 文件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on a Spreadsheet.")
        conf = self._param.setups["spreadsheet"]
        self.set_output("output_format", conf["output_format"])
        flatten_media_to_text = conf.get("flatten_media_to_text")

        parse_method = conf.get("parse_method", "deepdoc")

        # TCADP 解析器分支
        if parse_method.lower() == "tcadp parser":
            table_result_type = conf.get("table_result_type", "1")
            markdown_image_response_type = conf.get("markdown_image_response_type", "1")
            tcadp_parser = TCADPParser(
                table_result_type=table_result_type,
                markdown_image_response_type=markdown_image_response_type,
            )
            if not tcadp_parser.check_installation():
                raise RuntimeError("TCADP parser not available. Please check Tencent Cloud API configuration.")

            # 根据文件扩展名确定文件类型
            if re.search(r"\.xlsx?$", name, re.IGNORECASE):
                file_type = "XLSX"
            else:
                file_type = "CSV"

            self.callback(0.2, f"Using TCADP parser for {file_type} file.")
            sections, tables = tcadp_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                file_type=file_type,
                file_start_page=1,
                file_end_page=1000,
            )

            output_format = conf.get("output_format", "html")

            if output_format == "html":
                # HTML 输出：拼接所有 section 和 table
                html_content = ""
                for section, position_tag in sections:
                    if section:
                        html_content += section + "\n"
                for table in tables:
                    if table:
                        html_content += table + "\n"
                self.set_output("html", html_content)

            elif output_format == "json":
                # JSON 输出：转为结构化段落列表
                result = []
                for section, position_tag in sections:
                    if section:
                        result.append({"text": section, "doc_type_kwd": "text"})
                for table in tables:
                    if table:
                        result.append(
                            {
                                "text": table,
                                "doc_type_kwd": "text" if flatten_media_to_text else "table",
                            }
                        )
                self.set_output("json", result)

            elif output_format == "markdown":
                # Markdown 输出
                md_content = ""
                for section, position_tag in sections:
                    if section:
                        md_content += section + "\n\n"
                for table in tables:
                    if table:
                        md_content += table + "\n\n"
                self.set_output("markdown", md_content)
        else:
            # DeepDOC 解析器（默认）
            spreadsheet_parser = ExcelParser()
            if conf.get("output_format") == "html":
                htmls = spreadsheet_parser.html(blob, 1000000000)
                self.set_output("html", htmls[0])
            elif conf.get("output_format") == "json":
                self.set_output("json", [{"text": txt, "doc_type_kwd": "text"} for txt in spreadsheet_parser(blob) if txt])
            elif conf.get("output_format") == "markdown":
                self.set_output("markdown", spreadsheet_parser.markdown(blob))

    def _doc(self, name, blob, **kwargs):
        """解析 DOC 文件（旧版 Word 格式）为 text/json 段落。

        使用 Apache Tika 进行 .doc 文件的文本提取。
        由于 Tika 不支持直接提取图片，所有输出均为纯文本。

        Args:
            name: 文件名
            blob: 文件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on a DOC document")
        conf = self._param.setups["doc"]
        self.set_output("output_format", conf["output_format"])

        from tika import parser as tika_parser

        # Tika 提取纯文本内容
        parsed = tika_parser.from_buffer(io.BytesIO(blob))
        sections = [line for line in parsed["content"].split("\n") if line]

        if conf.get("output_format") == "json":
            self.set_output("json", [{"text": section, "doc_type_kwd": "text"} for section in sections])
            return

        self.set_output("markdown", "\n".join(sections))

    def _docx(self, name, blob, **kwargs):
        """解析 DOCX 文件并可选移除目录和页眉页脚。

        对 .doc 文件使用 Tika 降级处理；对 .docx 文件使用 python-docx
        直接解析，支持：
        - 提取文档大纲（Heading 样式段落）
        - 移除目录内容
        - 移除页眉页脚重复文本
        - 提取内嵌图片
        - 表格转 HTML

        Args:
            name: 文件名
            blob: 文件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on a DOCX document")
        conf = self._param.setups["docx"]
        self.set_output("output_format", conf["output_format"])
        flatten_media_to_text = conf.get("flatten_media_to_text")

        # .doc 文件（旧格式）：使用 Tika 降级处理
        if re.search(r"\.doc$", name, re.IGNORECASE):
            self.set_output("file", {**kwargs.get("file", {}), "outlines": []})
            try:
                from tika import parser as tika_parser
            except Exception as e:
                msg = f"tika not available: {e}. Unsupported .doc parsing."
                self.callback(0.8, msg)
                logging.warning(f"{msg} for {name}.")
                return

            doc_parsed = tika_parser.from_buffer(io.BytesIO(blob))
            content = doc_parsed.get("content")
            if content is None:
                msg = f"tika.parser got empty content from {name}."
                self.callback(0.8, msg)
                logging.warning(msg)
                return

            sections = [line.strip() for line in content.splitlines() if line and line.strip()]
            if conf.get("remove_toc"):
                sections = remove_toc_word(sections, [])

            if conf.get("output_format") == "json":
                self.set_output(
                    "json",
                    [{"text": line, "image": None, "doc_type_kwd": "text"} for line in sections],
                )
            elif conf.get("output_format") == "markdown":
                # Tika 输出为纯文本行，用空行分隔以保留段落边界
                self.set_output("markdown", "\n\n".join(sections))

            self.callback(0.8, "Finish parsing.")
            return

        # .docx 文件：使用 python-docx 解析
        docx_parser = Docx()

        # 提取标题大纲（用于元数据和目录移除）
        outlines = extract_word_outlines(name, blob)
        self.set_output("file", {**kwargs.get("file", {}), "outlines": outlines})

        # JSON 输出：包含文本/图片块 + 表格 HTML
        if conf.get("output_format") == "json":
            main_sections = docx_parser(name, binary=blob)
            # 页眉页脚过滤
            if conf.get("remove_header_footer"):
                header_footer_texts = extract_docx_header_footer_texts(binary=blob)
                main_sections = remove_header_footer_docx_sections(main_sections, header_footer_texts)
            # 目录过滤
            if conf.get("remove_toc"):
                main_sections = remove_toc_word(main_sections, outlines)
            sections = []
            for text, image, html in main_sections:
                # 文本/图片段
                sections.append(
                    {
                        "text": text,
                        "image": image,
                        "doc_type_kwd": "text" if flatten_media_to_text or image is None else "image",
                    }
                )
                # 表格段（以 HTML 形式存储）
                if html:
                    sections.append(
                        {
                            "text": html,
                            "image": None,
                            "doc_type_kwd": "text" if flatten_media_to_text else "table",
                        }
                    )
            # VLM 增强图片表格描述
            enhance_media_sections_with_vision(
                sections,
                self._canvas._tenant_id,
                conf.get("vlm"),
                callback=self.callback,
            )

            self.set_output("json", sections)

        # Markdown 输出：先移除目录/页眉页脚再输出
        elif conf.get("output_format") == "markdown":
            markdown_text = docx_parser.to_markdown(name, binary=blob)
            if conf.get("remove_header_footer"):
                header_footer_texts = extract_docx_header_footer_texts(binary=blob)
                markdown_lines = remove_header_footer_docx_sections(markdown_text.split("\n"), header_footer_texts)
                markdown_text = "\n".join(markdown_lines)
            if conf.get("remove_toc"):
                markdown_text = "\n".join(remove_toc_word(markdown_text.split("\n"), outlines))

            self.set_output("markdown", markdown_text)

    def _slides(self, name, blob, **kwargs):
        """解析演示文稿文件（PPT/PPTX）为 json 段落。

        支持两种解析器：
        - DeepDOC (RAGFlowPptParser)：默认，解析 PPTX 文件
        - TCADP：腾讯云文档解析 API

        Args:
            name: 文件名
            blob: 文件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on a PowerPoint Document")

        conf = self._param.setups["slides"]
        self.set_output("output_format", conf["output_format"])

        parse_method = conf.get("parse_method", "deepdoc")

        # TCADP 解析器分支
        if parse_method.lower() == "tcadp parser":
            table_result_type = conf.get("table_result_type", "1")
            markdown_image_response_type = conf.get("markdown_image_response_type", "1")
            tcadp_parser = TCADPParser(
                table_result_type=table_result_type,
                markdown_image_response_type=markdown_image_response_type,
            )
            if not tcadp_parser.check_installation():
                raise RuntimeError("TCADP parser not available. Please check Tencent Cloud API configuration.")

            if re.search(r"\.pptx?$", name, re.IGNORECASE):
                file_type = "PPTX"
            else:
                file_type = "PPT"

            self.callback(0.2, f"Using TCADP parser for {file_type} file.")

            sections, tables = tcadp_parser.parse_pdf(
                filepath=name,
                binary=blob,
                callback=self.callback,
                file_type=file_type,
                file_start_page=1,
                file_end_page=1000,
            )

            output_format = conf.get("output_format", "json")
            if output_format == "json":
                result = []
                for section, position_tag in sections:
                    if section:
                        result.append({"text": section, "doc_type_kwd": "text"})
                for table in tables:
                    if table:
                        result.append({"text": table, "doc_type_kwd": "table"})
                self.set_output("json", result)
        else:
            # DeepDOC 解析器（默认），仅支持 PPTX 格式
            from deepdoc.parser.ppt_parser import RAGFlowPptParser as ppt_parser

            ppt_parser = ppt_parser()
            txts = ppt_parser(blob, 0, 100000, None)

            sections = [{"text": section, "doc_type_kwd": "text"} for section in txts if section.strip()]

            # 演示文稿仅支持 JSON 输出格式
            assert conf.get("output_format") == "json", "have to be json for ppt"
            if conf.get("output_format") == "json":
                self.set_output("json", sections)

    def _markdown(self, name, blob, **kwargs):
        """解析 Markdown 文件为 text/json 段落。

        解析流程：
        1. 使用 Naive Markdown 解析器分离文本、表格和图片
        2. 对图片段落进行 VLM 增强描述
        3. 按指定格式输出

        Args:
            name: 文件名
            blob: 文件二进制内容
        """
        from functools import reduce

        from rag.app.naive import Markdown as naive_markdown_parser
        from rag.nlp import concat_img

        self.callback(random.randint(1, 5) / 100.0, "Start to work on a markdown.")
        conf = self._param.setups["markdown"]
        self.set_output("output_format", conf["output_format"])
        flatten_media_to_text = conf.get("flatten_media_to_text")

        markdown_parser = naive_markdown_parser()
        # 解析 Markdown，分离文本段落、表格和段落关联图片
        sections, tables, section_images = markdown_parser(
            name,
            blob,
            separate_tables=False,
            delimiter=conf.get("delimiter"),
            return_section_images=True,
        )

        if conf.get("output_format") == "json":
            json_results = []

            for idx, (section_text, _) in enumerate(sections):
                json_result = {
                    "text": section_text,
                }

                # 处理段落关联的图片（多图时拼接为一张）
                images = []
                if section_images and len(section_images) > idx and section_images[idx] is not None:
                    images.append(section_images[idx])
                if images:
                    combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
                    json_result["image"] = combined_image
                json_result["doc_type_kwd"] = (
                    "text"
                    if flatten_media_to_text or json_result.get("image") is None
                    else "image"
                )
                json_results.append(json_result)

            # 添加表格段落
            for table in tables:
                table_text = table[0][1] if table and table[0] else ""
                if table_text:
                    json_results.append(
                        {
                            "text": table_text,
                            "doc_type_kwd": "text" if flatten_media_to_text else "table",
                        }
                    )

            # VLM 增强图片/表格描述
            enhance_media_sections_with_vision(
                json_results,
                self._canvas._tenant_id,
                conf.get("vlm"),
                callback=self.callback,
            )
            self.set_output("json", json_results)
        else:
            # Text 输出：拼接所有文本段落和表格
            texts = [section_text for section_text, _ in sections if section_text]
            texts.extend(table[0][1] for table in tables if table and table[0] and table[0][1])
            self.set_output("text", "\n".join(texts))

    def _code(self, name, blob, **kwargs):
        """解析文本和源代码文件为纯文本切块。

        使用 TxtParser 按分隔符将内容切分为段落，
        每段不超过 chunk_token_num 个 token。

        Args:
            name: 文件名
            blob: 文件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on a text or code file.")
        conf = self._param.setups["text&code"]
        self.set_output("output_format", conf["output_format"])

        sections = TxtParser()(
            name,
            blob,
            conf.get("chunk_token_num", 128),
            conf.get("delimiter", "\n!?;。；！？"),
        )
        if conf.get("output_format") == "json":
            self.set_output("json", [{"text": section[0], "doc_type_kwd": "text"} for section in sections if section[0]])
            return

        self.set_output("text", "\n".join([section[0] for section in sections if section[0]]))

    def _html(self, name, blob, **kwargs):
        """解析 HTML 文件为 text/json 段落。

        支持：
        - 页眉/页脚元素移除（<header>、<footer>、role=banner/contentinfo）
        - 目录内容移除
        - 按 chunk_token_num 切分段落

        Args:
            name: 文件名
            blob: HTML 文件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on an HTML document.")
        conf = self._param.setups["html"]
        self.set_output("output_format", conf["output_format"])

        # 预处理：移除 HTML 页眉/页脚元素
        if conf.get("remove_header_footer"):
            blob = remove_header_footer_html_blob(blob)

        sections = HtmlParser()(name, blob, int(conf.get("chunk_token_num", 512)))
        # 目录移除
        if conf.get("remove_toc"):
            sections, _ = remove_toc(sections)
        if conf.get("output_format") == "json":
            self.set_output("json", [{"text": section, "doc_type_kwd": "text"} for section in sections if section])
            return

        self.set_output("text", "\n".join([section for section in sections if section]))

    def _image(self, name, blob, **kwargs):
        """解析图片文件。

        支持两种解析模式：
        - OCR 模式：使用光学字符识别提取图片中的文字
        - VLM 模式：使用视觉语言模型生成图片的自然语言描述

        Args:
            name: 文件名
            blob: 图片二进制内容
        """
        from deepdoc.vision import OCR

        self.callback(random.randint(1, 5) / 100.0, "Start to work on an image.")
        conf = self._param.setups["image"]
        self.set_output("output_format", "json")

        # 加载并转换为 RGB 格式
        img = Image.open(io.BytesIO(blob)).convert("RGB")

        if conf["parse_method"] == "ocr":
            # OCR 模式：识别图片中的文字
            ocr = OCR()
            bxs = ocr(np.array(img))  # 返回识别框和文字结果
            txt = "\n".join([t[0] for _, t in bxs if t[0]])
        else:
            # VLM 模式：用视觉语言模型生成图片描述
            lang = conf["lang"]
            cv_model_config = get_model_config_from_provider_instance(self._canvas.get_tenant_id(), LLMType.IMAGE2TEXT, conf["parse_method"])
            cv_model = LLMBundle(self._canvas.get_tenant_id(), cv_model_config, lang=lang)
            img_binary = io.BytesIO()
            img.save(img_binary, format="JPEG")
            img_binary.seek(0)

            system_prompt = conf.get("system_prompt")
            if system_prompt:
                # 使用自定义提示词指导 VLM 描述
                txt = cv_model.describe_with_prompt(img_binary.read(), system_prompt)
            else:
                # 默认图片描述
                txt = cv_model.describe(img_binary.read())

        json_result = [
            {
                "text": txt,
                "image": img,
                "doc_type_kwd": "image",
            }
        ]
        self.set_output("json", json_result)

    def _audio(self, name, blob, **kwargs):
        """解析音频文件为文本（语音转文字）。

        将音频文件保存为临时文件，调用 Speech-to-Text 模型进行转录。

        Args:
            name: 音频文件名
            blob: 音频二进制内容
        """
        import os
        import tempfile

        self.callback(random.randint(1, 5) / 100.0, "Start to work on an audio.")

        conf = self._param.setups["audio"]
        vlm = conf.get("vlm")
        self.set_output("output_format", conf["output_format"])

        # 保留文件扩展名以确保 STT 模型正确识别音频格式
        _, ext = os.path.splitext(name)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmpf:
            tmpf.write(blob)
            tmpf.flush()
            tmp_path = os.path.abspath(tmpf.name)
            # 获取语音转文字模型并进行转录
            seq2txt_model_config = get_model_config_from_provider_instance(self._canvas.get_tenant_id(), LLMType.SPEECH2TEXT, vlm["llm_id"])
            seq2txt_mdl = LLMBundle(self._canvas.get_tenant_id(), seq2txt_model_config)
            txt = seq2txt_mdl.transcription(tmp_path)

            self.set_output("text", txt)

    def _video(self, name, blob, **kwargs):
        """解析视频文件为文本描述。

        使用视觉语言模型（IMAGE2TEXT）对视频进行帧分析，
        生成视频内容的文字描述。

        Args:
            name: 视频文件名
            blob: 视频二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on an video.")

        conf = self._param.setups["video"]
        vlm = conf.get("vlm")
        self.set_output("output_format", conf["output_format"])

        # 获取视觉语言模型
        cv_model_config = get_model_config_from_provider_instance(self._canvas.get_tenant_id(), LLMType.IMAGE2TEXT, vlm["llm_id"])
        cv_mdl = LLMBundle(self._canvas.get_tenant_id(), cv_model_config)
        video_prompt = str(conf.get("prompt", "") or "")

        # 异步调用 VLM 的 video chat 接口进行帧分析
        txt = asyncio.run(cv_mdl.async_chat(system="", history=[], gen_conf={}, video_bytes=blob, filename=name, video_prompt=video_prompt))

        self.set_output("text", txt)

    def _email(self, name, blob, **kwargs):
        """解析邮件文件（EML/MSG）为结构化内容。

        提取邮件的核心字段：
        - 基本信息：from, to, cc, bcc, date, subject
        - 正文内容：纯文本 + HTML 格式
        - 附件列表
        - 元数据：邮件头中的其他字段

        支持 .eml（MIME 格式）和 .msg（Outlook 格式）两种邮件格式。

        Args:
            name: 邮件文件名
            blob: 邮件二进制内容
        """
        self.callback(random.randint(1, 5) / 100.0, "Start to work on an email.")

        email_content = {}
        conf = self._param.setups["email"]
        self.set_output("output_format", conf["output_format"])
        target_fields = conf["fields"]

        _, ext = os.path.splitext(name)
        if ext == ".eml":
            # === 解析 EML 格式邮件 ===
            from email import policy
            from email.parser import BytesParser

            msg = BytesParser(policy=policy.default).parse(io.BytesIO(blob))
            email_content["metadata"] = {}

            # 提取邮件头信息
            for header, value in msg.items():
                # 目标字段（from, to, cc, bcc, date, subject）
                if header.lower() in target_fields:
                    email_content[header.lower()] = value
                # 其他字段存入 metadata
                elif header.lower() not in ["from", "to", "cc", "bcc", "date", "subject"]:
                    email_content["metadata"][header.lower()] = value

            # 提取邮件正文（支持纯文本和 HTML 两种格式）
            if "body" in target_fields:
                body_text, body_html = [], []

                def _add_content(m, content_type):
                    """递归提取 multipart 邮件各部分的内容"""
                    def _decode_payload(payload, charset, target_list):
                        """尝试多种编码解码邮件正文"""
                        try:
                            target_list.append(payload.decode(charset))
                        except (UnicodeDecodeError, LookupError):
                            # 编码失败时尝试常见中英文编码
                            for enc in ["utf-8", "gb2312", "gbk", "gb18030", "latin1"]:
                                try:
                                    target_list.append(payload.decode(enc))
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                target_list.append(payload.decode("utf-8", errors="ignore"))

                    if content_type == "text/plain":
                        payload = msg.get_payload(decode=True)
                        charset = msg.get_content_charset() or "utf-8"
                        _decode_payload(payload, charset, body_text)
                    elif content_type == "text/html":
                        payload = msg.get_payload(decode=True)
                        charset = msg.get_content_charset() or "utf-8"
                        _decode_payload(payload, charset, body_html)
                    elif "multipart" in content_type:
                        if m.is_multipart():
                            for part in m.iter_parts():
                                _add_content(part, part.get_content_type())

                _add_content(msg, msg.get_content_type())

                email_content["text"] = "\n".join(body_text)
                email_content["text_html"] = "\n".join(body_html)

            # 提取附件
            if "attachments" in target_fields:
                attachments = []
                for part in msg.iter_attachments():
                    content_disposition = part.get("Content-Disposition")
                    if content_disposition:
                        dispositions = content_disposition.strip().split(";")
                        if dispositions[0].lower() == "attachment":
                            filename = part.get_filename()
                            payload = part.get_payload(decode=True).decode(part.get_content_charset())
                            attachments.append(
                                {
                                    "filename": filename,
                                    "payload": payload,
                                }
                            )
                email_content["attachments"] = attachments
        else:
            # === 解析 MSG 格式邮件（Outlook） ===
            import extract_msg

            msg = extract_msg.Message(blob)

            # 提取基本邮件头字段
            basic_content = {
                "from": msg.sender,
                "to": msg.to,
                "cc": msg.cc,
                "bcc": msg.bcc,
                "date": msg.date,
                "subject": msg.subject,
            }
            email_content.update({k: v for k, v in basic_content.items() if k in target_fields})

            # 元数据
            email_content["metadata"] = {
                "message_id": msg.messageId,
                "in_reply_to": msg.inReplyTo,
            }

            # 提取正文（优先纯文本，无纯文本时使用 HTML）
            if "body" in target_fields:
                email_content["text"] = msg.body[0] if isinstance(msg.body, list) and msg.body else msg.body
                if not email_content["text"] and msg.htmlBody:
                    email_content["text"] = msg.htmlBody[0] if isinstance(msg.htmlBody, list) and msg.htmlBody else msg.htmlBody

            # 提取附件
            if "attachments" in target_fields:
                attachments = []
                for t in msg.attachments:
                    attachments.append(
                        {
                            "filename": t.name,
                            "payload": t.data.decode("utf-8"),
                        }
                    )
                email_content["attachments"] = attachments

        # === 按输出格式生成结果 ===
        if conf["output_format"] == "json":
            email_content["doc_type_kwd"] = "text"
            self.set_output("json", [email_content])
        else:
            # Text 格式：拼接所有字段为可读文本
            content_txt = ""
            for k, v in email_content.items():
                if isinstance(v, str):
                    content_txt += f"{k}:{v}" + "\n"
                elif isinstance(v, dict):
                    content_txt += f"{k}:{json.dumps(v)}" + "\n"
                elif isinstance(v, list):
                    for fb in v:
                        if isinstance(fb, dict):
                            content_txt += f"{fb['filename']}:{fb['payload']}" + "\n"
                        else:
                            content_txt += fb
            self.set_output("text", content_txt)

    def _epub(self, name, blob, **kwargs):
        """解析 EPUB 电子书文件为 text/json 段落。

        Args:
            name: 文件名
            blob: EPUB 文件二进制内容
        """
        from deepdoc.parser import EpubParser

        self.callback(random.randint(1, 5) / 100.0, "Start to work on an EPUB.")
        conf = self._param.setups["epub"]
        self.set_output("output_format", conf["output_format"])

        epub_parser = EpubParser()
        sections = epub_parser(name, binary=blob)

        if conf.get("output_format") == "json":
            json_results = [{"text": s, "doc_type_kwd": "text"} for s in sections if s]
            self.set_output("json", json_results)
        else:
            self.set_output("text", "\n".join(s for s in sections if s))

    async def _invoke(self, **kwargs):
        """Parser 组件的异步入口方法。

        流水线引擎调用此方法触发文档解析。处理流程：
        1. 校验上游输入数据（ParserFromUpstream schema）
        2. 根据 doc_id 或 file 信息从存储服务获取文件二进制内容
        3. 根据文件扩展名匹配对应的解析方法
        4. 在线程池中执行解析（避免阻塞事件循环）
        5. 异步上传解析结果中的图片到对象存储

        Args:
            **kwargs: 上游组件传递的参数字典，需符合 ParserFromUpstream schema
        """
        # 文件类型 → 解析方法的映射表
        function_map = {
            "pdf": self._pdf,
            "markdown": self._markdown,
            "text&code": self._code,
            "html": self._html,
            "spreadsheet": self._spreadsheet,
            "slides": self._slides,
            "doc": self._doc,
            "docx": self._docx,
            "image": self._image,
            "audio": self._audio,
            "video": self._video,
            "email": self._email,
            "epub": self._epub,
        }

        # === 步骤1：校验上游输入 ===
        try:
            from_upstream = ParserFromUpstream.model_validate(kwargs)
        except Exception as e:
            self.set_output("_ERROR", f"Input error: {str(e)}")
            return

        # === 步骤2：获取文件二进制内容 ===
        name = from_upstream.name
        if self._canvas._doc_id:
            # 从文档存储服务获取（doc_id 模式）
            b, n = File2DocumentService.get_storage_address(doc_id=self._canvas._doc_id)
            blob = settings.STORAGE_IMPL.get(b, n)
        else:
            # 从文件服务获取（file 模式）
            blob = FileService.get_blob(from_upstream.file["created_by"], from_upstream.file["id"])

        # === 步骤3：按文件扩展名分发到对应解析方法 ===
        done = False
        for p_type, conf in self._param.setups.items():
            # 检查文件扩展名是否匹配当前文档类型
            if from_upstream.name.split(".")[-1].lower() not in conf.get("suffix", []):
                continue
            call_kwargs = dict(kwargs)
            call_kwargs.pop("name", None)
            call_kwargs.pop("blob", None)

            # 在线程池中执行解析（各解析方法均为同步函数）
            await thread_pool_exec(function_map[p_type], name, blob, **call_kwargs)
            done = True
            break

        if not done:
            raise Exception("No suitable for file extension: `.%s`" % from_upstream.name.split(".")[-1].lower())

        # === 步骤4：异步上传解析结果中的图片到对象存储 ===
        outs = self.output()
        tasks = []
        for d in outs.get("json", []):
            # image2id 将 dict 中的 image（PIL.Image）上传到对象存储并替换为 img_id
            tasks.append(asyncio.create_task(image2id(d, partial(settings.STORAGE_IMPL.put, tenant_id=self._canvas._tenant_id), get_uuid())))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("Error while parsing: %s" % e)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
