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
PaddleOCR 解析器模块。

本模块提供基于 PaddleOCR API 的 PDF 文档解析功能。支持以下算法：
- PaddleOCR-VL：视觉语言模型，端到端文档解析
- PaddleOCR-VL-1.5：VL 模型升级版
- PP-OCRv5：传统 OCR 文本检测与识别
- PP-StructureV3：文档结构分析（版面分析 + 表格识别）

主要功能：
- 通过 HTTP API 提交 PDF 文件进行解析
- 支持丰富的版面分析参数配置（布局检测、图表识别、印章识别等）
- Markdown 格式输出，自动去除图片标签
- 位置标签生成与图片裁剪
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Optional, Union, Tuple, List

import numpy as np
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


# 类型别名：支持的算法类型
AlgorithmType = Literal["PaddleOCR-VL", "PP-OCRv5", "PP-StructureV3", "PaddleOCR-VL-1.5"]
# Section 为可变长度的元组（根据 parse_method 不同而变化）
SectionTuple = tuple[str, ...]
TableTuple = tuple[str, ...]
ParseResult = tuple[list[SectionTuple], list[TableTuple]]
# 当前支持的所有 PaddleOCR 算法
SUPPORTED_PADDLEOCR_ALGORITHMS: tuple[AlgorithmType, ...] = (
    "PaddleOCR-VL",
    "PP-OCRv5",
    "PP-StructureV3",
    "PaddleOCR-VL-1.5",
)


# Markdown 图片标签的正则表达式，用于去除 <img> 和包裹的 <div> 标签
_MARKDOWN_IMAGE_PATTERN = re.compile(
    r"""
        <div[^>]*>\s*
        <img[^>]*/>\s*
        </div>
        |
        <img[^>]*/>
        """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _remove_images_from_markdown(markdown: str) -> str:
    """
    从 Markdown 文本中移除图片标签。

    PaddleOCR API 返回的 Markdown 可能包含 <img> 标签，
    这些标签在纯文本检索场景中没有意义，需要移除。

    Args:
        markdown: 包含图片标签的 Markdown 文本

    Returns:
        str: 移除图片标签后的 Markdown 文本
    """
    return _MARKDOWN_IMAGE_PATTERN.sub("", markdown)


def _normalize_bbox(bbox: list[Any] | tuple[Any, ...]) -> tuple[float, float, float, float]:
    """
    归一化边界框坐标，确保坐标顺序合法。

    处理以下情况：
    - 坐标不足 4 个时返回零值
    - left > right 时交换
    - top > bottom 时交换

    Args:
        bbox: 边界框坐标 [left, top, right, bottom] 或元组

    Returns:
        tuple: (left, top, right, bottom) 归一化后的浮点坐标
    """
    if len(bbox) < 4:
        return 0.0, 0.0, 0.0, 0.0

    left, top, right, bottom = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top
    return left, top, right, bottom


@dataclass
class PaddleOCRVLConfig:
    """
    PaddleOCR-VL 算法专用配置。

    所有字段均为 Optional，仅非 None 值才会被发送到 API。
    支持的配置项涵盖文档方向分类、文档矫正、版面检测、图表识别、
    印章识别、VLM 推理参数等。
    """

    use_doc_orientation_classify: Optional[bool] = False
    use_doc_unwarping: Optional[bool] = False
    use_layout_detection: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_ocr_for_image_block: Optional[bool] = None
    layout_threshold: Optional[Union[float, dict]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None
    layout_merge_bboxes_mode: Optional[Union[str, dict]] = None
    layout_shape_mode: Optional[str] = None
    prompt_label: Optional[str] = None
    format_block_content: Optional[bool] = True
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    max_new_tokens: Optional[int] = None
    merge_layout_blocks: Optional[bool] = False
    markdown_ignore_labels: Optional[List[str]] = None
    vlm_extra_args: Optional[dict] = None
    restructure_pages: Optional[bool] = False
    merge_tables: Optional[bool] = None
    relevel_titles: Optional[bool] = None


@dataclass
class PaddleOCRConfig:
    """
    PaddleOCR 解析器主配置。

    包含 API 连接参数、算法选择、通用输出选项和算法专用配置。
    支持从字典或关键字参数构建。
    """

    api_url: str = ""
    access_token: Optional[str] = None
    algorithm: AlgorithmType = "PaddleOCR-VL"
    request_timeout: int = 600
    prettify_markdown: bool = True
    show_formula_number: bool = True
    visualize: bool = False
    additional_params: dict[str, Any] = field(default_factory=dict)
    algorithm_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Optional[dict[str, Any]]) -> "PaddleOCRConfig":
        """
        从字典创建配置对象。

        自动分离通用参数和算法专用参数，并验证算法的有效性。

        Args:
            config: 配置字典，可包含 api_url、algorithm、algorithm_config 等字段

        Returns:
            PaddleOCRConfig: 配置实例

        Raises:
            ValueError: 算法类型不受支持时抛出
        """
        if not config:
            return cls()

        cfg = config.copy()
        algorithm = cfg.get("algorithm", "PaddleOCR-VL")

        # 验证算法是否受支持
        if algorithm not in SUPPORTED_PADDLEOCR_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # 提取算法专用配置，使用 VL 配置的默认值作为基准
        algorithm_config: dict[str, Any] = {}
        if algorithm in SUPPORTED_PADDLEOCR_ALGORITHMS:
            algorithm_config = asdict(PaddleOCRVLConfig())
        algorithm_config_user = cfg.get("algorithm_config")
        if isinstance(algorithm_config_user, dict):
            algorithm_config.update({k: v for k, v in algorithm_config_user.items() if v is not None})

        # 移除已处理的键
        cfg.pop("algorithm_config", None)

        # 准备初始化参数：只保留 dataclass 定义的字段
        field_names = {field.name for field in fields(cls)}
        init_kwargs: dict[str, Any] = {}

        for field_name in field_names:
            if field_name in cfg:
                init_kwargs[field_name] = cfg[field_name]

        init_kwargs["algorithm_config"] = algorithm_config

        return cls(**init_kwargs)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "PaddleOCRConfig":
        """
        从关键字参数创建配置对象，是 from_dict 的便捷包装。

        Args:
            **kwargs: 配置关键字参数

        Returns:
            PaddleOCRConfig: 配置实例
        """
        return cls.from_dict(kwargs)


class PaddleOCRParser(RAGFlowPdfParser):
    """
    PaddleOCR API 解析器。

    通过 PaddleOCR 远程 API 服务对 PDF 文档进行版面分析、OCR 识别和
    结构化信息提取。继承自 RAGFlowPdfParser 以获得图片裁剪等基础能力。

    使用方式：
    1. 设置环境变量 PADDLEOCR_API_URL 指向 PaddleOCR API 服务
    2. 可选设置 PADDLEOCR_ACCESS_TOKEN 进行鉴权
    3. 调用 parse_pdf() 进行文档解析
    """

    # PDF 渲染缩放因子
    _ZOOMIN = 2

    # 通用参数名到 API 参数名的映射（camelCase 转换）
    _COMMON_FIELD_MAPPING: ClassVar[dict[str, str]] = {
        "prettify_markdown": "prettifyMarkdown",
        "show_formula_number": "showFormulaNumber",
        "visualize": "visualize",
    }

    # VL 算法专用参数名到 API 参数名的映射
    _VL_FIELD_MAPPING: ClassVar[dict[str, str]] = {
        "use_doc_orientation_classify": "useDocOrientationClassify",
        "use_doc_unwarping": "useDocUnwarping",
        "use_layout_detection": "useLayoutDetection",
        "use_chart_recognition": "useChartRecognition",
        "use_seal_recognition": "useSealRecognition",
        "use_ocr_for_image_block": "useOcrForImageBlock",
        "layout_threshold": "layoutThreshold",
        "layout_nms": "layoutNms",
        "layout_unclip_ratio": "layoutUnclipRatio",
        "layout_merge_bboxes_mode": "layoutMergeBboxesMode",
        "layout_shape_mode": "layoutShapeMode",
        "prompt_label": "promptLabel",
        "format_block_content": "formatBlockContent",
        "repetition_penalty": "repetitionPenalty",
        "temperature": "temperature",
        "top_p": "topP",
        "min_pixels": "minPixels",
        "max_pixels": "maxPixels",
        "max_new_tokens": "maxNewTokens",
        "merge_layout_blocks": "mergeLayoutBlocks",
        "markdown_ignore_labels": "markdownIgnoreLabels",
        "vlm_extra_args": "vlmExtraArgs",
        "restructure_pages": "restructurePages",
        "merge_tables": "mergeTables",
        "relevel_titles": "relevelTitles",
    }

    # 各算法对应的参数映射表（当前所有算法使用相同的 VL 映射）
    _ALGORITHM_FIELD_MAPPINGS: ClassVar[dict[str, dict[str, str]]] = {
        "PaddleOCR-VL": _VL_FIELD_MAPPING,
        "PP-OCRv5": _VL_FIELD_MAPPING,
        "PP-StructureV3": _VL_FIELD_MAPPING,
        "PaddleOCR-VL-1.5": _VL_FIELD_MAPPING,
    }

    def __init__(
        self,
        api_url: Optional[str] = None,
        access_token: Optional[str] = None,
        algorithm: AlgorithmType = "PaddleOCR-VL",
        *,
        request_timeout: int = 600,
    ):
        """
        初始化 PaddleOCR 解析器。

        参数优先级：构造函数参数 > 环境变量

        Args:
            api_url: PaddleOCR API 服务地址
            access_token: API 访问令牌
            algorithm: 使用的算法类型
            request_timeout: 请求超时时间（秒）
        """
        self.outlines = []
        self.api_url = api_url.rstrip("/") if api_url else os.getenv("PADDLEOCR_API_URL", "")
        self.access_token = access_token or os.getenv("PADDLEOCR_ACCESS_TOKEN")
        self.algorithm = algorithm
        self.request_timeout = request_timeout
        self.logger = logging.getLogger(self.__class__.__name__)

        # PDF 文件类型固定为 0
        self.file_type = 0

        # 初始化页面图片列表（用于裁剪功能）
        self.page_images: list[Image.Image] = []
        self.page_from = 0

    # ==================== 公共方法 ====================

    def check_installation(self) -> tuple[bool, str]:
        """
        检查解析器是否正确安装和配置。

        Returns:
            tuple: (是否可用, 状态描述)
        """
        if not self.api_url:
            return False, "[PaddleOCR] API URL not configured"

        # TODO [@Bobholamovic]: 检查 URL 可用性和 token 有效性

        return True, ""

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable[[float, str], None]] = None,
        *,
        parse_method: str = "raw",
        api_url: Optional[str] = None,
        access_token: Optional[str] = None,
        algorithm: Optional[AlgorithmType] = None,
        request_timeout: Optional[int] = None,
        prettify_markdown: Optional[bool] = None,
        show_formula_number: Optional[bool] = None,
        visualize: Optional[bool] = None,
        additional_params: Optional[dict[str, Any]] = None,
        algorithm_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ParseResult:
        """
        使用 PaddleOCR API 解析 PDF 文档（主入口方法）。

        完整流程：
        1. 提取 PDF 目录大纲
        2. 构建配置并验证
        3. 准备文件数据并生成页面缩略图
        4. 发送 API 请求
        5. 处理响应，转换为 sections 和 tables

        Args:
            filepath: PDF 文件路径
            binary: PDF 二进制数据
            callback: 进度回调 (progress, message)
            parse_method: 解析方法，可选 "raw"、"manual"、"pipeline"、"paper"
            api_url: 覆盖默认的 API URL
            access_token: 覆盖默认的访问令牌
            algorithm: 覆盖默认的算法
            request_timeout: 覆盖默认的超时时间
            prettify_markdown: 是否美化 Markdown 输出
            show_formula_number: 是否显示公式编号
            visualize: 是否生成可视化结果
            additional_params: 额外的 API 参数
            algorithm_config: 算法专用配置
            **kwargs: 其他参数（预留）

        Returns:
            ParseResult: (sections 列表, tables 列表)
        """
        self.outlines = extract_pdf_outlines(binary if binary is not None else filepath)

        # 构建配置字典：参数覆盖优先级为 函数参数 > 实例属性
        config_dict = {
            "api_url": api_url if api_url is not None else self.api_url,
            "access_token": access_token if access_token is not None else self.access_token,
            "algorithm": algorithm if algorithm is not None else self.algorithm,
            "request_timeout": request_timeout if request_timeout is not None else self.request_timeout,
        }
        if prettify_markdown is not None:
            config_dict["prettify_markdown"] = prettify_markdown
        if show_formula_number is not None:
            config_dict["show_formula_number"] = show_formula_number
        if visualize is not None:
            config_dict["visualize"] = visualize
        if additional_params is not None:
            config_dict["additional_params"] = additional_params
        if algorithm_config is not None:
            config_dict["algorithm_config"] = algorithm_config

        cfg = PaddleOCRConfig.from_dict(config_dict)

        if not cfg.api_url:
            raise RuntimeError("[PaddleOCR] API URL missing")

        # 准备文件数据并生成页面缩略图（用于后续裁剪）
        data_bytes = self._prepare_file_data(filepath, binary)

        # 生成页面图片供裁剪功能使用
        input_source = filepath if binary is None else binary
        try:
            self.__images__(input_source, callback=callback)
        except Exception as e:
            self.logger.warning(f"[PaddleOCR] Failed to generate page images for cropping: {e}")

        # 构建并发送 API 请求
        result = self._send_request(data_bytes, cfg, callback)

        # 处理 API 响应
        sections = self._transfer_to_sections(result, algorithm=cfg.algorithm, parse_method=parse_method)
        if callback:
            callback(0.9, f"[PaddleOCR] done, sections: {len(sections)}")

        tables = self._transfer_to_tables(result)
        if callback:
            callback(1.0, f"[PaddleOCR] done, tables: {len(tables)}")

        return sections, tables

    # ==================== 私有方法 ====================

    def _prepare_file_data(self, filepath: str | PathLike[str], binary: BytesIO | bytes | None) -> bytes:
        """
        准备 API 请求所需的文件数据。

        优先使用内存中的二进制数据，其次从文件路径读取。

        Args:
            filepath: 文件路径
            binary: 二进制数据（可为 bytes、bytearray 或 BytesIO）

        Returns:
            bytes: 文件的原始字节数据

        Raises:
            FileNotFoundError: 文件不存在且没有提供二进制数据时抛出
        """
        source_path = Path(filepath)

        if binary is not None:
            if isinstance(binary, (bytes, bytearray)):
                return binary
            return binary.getbuffer().tobytes()

        if not source_path.exists():
            raise FileNotFoundError(f"[PaddleOCR] file not found: {source_path}")

        return source_path.read_bytes()

    def _build_payload(self, data: bytes, file_type: int, config: PaddleOCRConfig) -> dict[str, Any]:
        """
        构建 API 请求的 JSON 负载。

        将文件 Base64 编码后与配置参数组装为 API 期望的 JSON 格式。
        参数名通过字段映射表转换为 API 的 camelCase 命名。

        Args:
            data: 文件字节数据
            file_type: 文件类型标识
            config: 解析器配置

        Returns:
            dict: API 请求负载
        """
        payload: dict[str, Any] = {
            "file": base64.b64encode(data).decode("ascii"),
            "fileType": file_type,
        }

        # 添加通用参数（通过映射表转换参数名）
        for param_key, param_value in [
            ("prettify_markdown", config.prettify_markdown),
            ("show_formula_number", config.show_formula_number),
            ("visualize", config.visualize),
        ]:
            if param_value is not None:
                api_param = self._COMMON_FIELD_MAPPING[param_key]
                payload[api_param] = param_value

        # 添加算法专用参数（通过映射表转换参数名）
        algorithm_mapping = self._ALGORITHM_FIELD_MAPPINGS.get(config.algorithm, {})
        for param_key, param_value in config.algorithm_config.items():
            if param_value is not None and param_key in algorithm_mapping:
                api_param = algorithm_mapping[param_key]
                payload[api_param] = param_value

        # 添加额外的自定义参数
        if config.additional_params:
            payload.update(config.additional_params)

        return payload

    def _send_request(self, data: bytes, config: PaddleOCRConfig, callback: Optional[Callable[[float, str], None]]) -> dict[str, Any]:
        """
        发送请求到 PaddleOCR API 并验证响应。

        包含请求构建、HTTP 调用、JSON 解析和响应格式验证的完整流程。

        Args:
            data: 文件字节数据
            config: 解析器配置
            callback: 进度回调函数

        Returns:
            dict: API 返回的 result 字段内容

        Raises:
            RuntimeError: 请求失败或响应格式无效时抛出
        """
        # 构建请求负载
        payload = self._build_payload(data, self.file_type, config)

        # 准备请求头
        headers = {"Content-Type": "application/json", "Client-Platform": "ragflow"}
        if config.access_token:
            headers["Authorization"] = f"token {config.access_token}"

        self.logger.info("[PaddleOCR] invoking API")
        if callback:
            callback(0.1, "[PaddleOCR] submitting request")

        # 发送 HTTP POST 请求
        try:
            resp = requests.post(config.api_url, json=payload, headers=headers, timeout=self.request_timeout)
            resp.raise_for_status()
        except Exception as exc:
            if callback:
                callback(-1, f"[PaddleOCR] request failed: {exc}")
            raise RuntimeError(f"[PaddleOCR] request failed: {exc}")

        # 解析 JSON 响应
        try:
            response_data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"[PaddleOCR] response is not JSON: {exc}") from exc

        if callback:
            callback(0.8, "[PaddleOCR] response received")

        # 验证响应格式：errorCode 必须为 0，result 必须是字典
        if response_data.get("errorCode") != 0 or not isinstance(response_data.get("result"), dict):
            if callback:
                callback(-1, "[PaddleOCR] invalid response format")
            raise RuntimeError("[PaddleOCR] invalid response format")

        return response_data["result"]

    def _transfer_to_sections(self, result: dict[str, Any], algorithm: AlgorithmType, parse_method: str) -> list[SectionTuple]:
        """
        将 API 响应转换为 RAGFlow 标准 sections 格式。

        遍历每个页面的 layoutParsingResults，提取每个版面块的文本内容，
        生成位置标签并根据 parse_method 组合不同长度的元组。

        Args:
            result: API 返回的 result 字段
            algorithm: 使用的算法类型
            parse_method: 解析方法 ("raw" / "manual" / "pipeline" / "paper")

        Returns:
            list[SectionTuple]: sections 列表
        """
        sections: list[SectionTuple] = []

        if algorithm in SUPPORTED_PADDLEOCR_ALGORITHMS:
            layout_parsing_results = result.get("layoutParsingResults", [])

            for page_idx, layout_result in enumerate(layout_parsing_results):
                pruned_result = layout_result.get("prunedResult", {})
                parsing_res_list = pruned_result.get("parsing_res_list", [])

                for block in parsing_res_list:
                    block_content = block.get("block_content", "").strip()
                    if not block_content:
                        continue

                    # 去除 Markdown 中的图片标签
                    block_content = _remove_images_from_markdown(block_content)

                    # 获取版面块标签和边界框
                    label = block.get("block_label", "")
                    block_bbox = block.get("block_bbox", [0, 0, 0, 0])
                    left, top, right, bottom = _normalize_bbox(block_bbox)

                    # 生成位置标签：@@页码\t左\t右\t上\t下##
                    tag = f"@@{page_idx + 1}\t{left // self._ZOOMIN}\t{right // self._ZOOMIN}\t{top // self._ZOOMIN}\t{bottom // self._ZOOMIN}##"

                    # 根据 parse_method 构建不同格式的 section 元组
                    if parse_method in {"manual", "pipeline"}:
                        sections.append((block_content, label, tag))
                    elif parse_method == "paper":
                        sections.append((block_content + tag, label))
                    else:
                        sections.append((block_content, tag))

        return sections

    def _transfer_to_tables(self, result: dict[str, Any]) -> list[TableTuple]:
        """
        将 API 响应转换为 tables 格式（当前版本暂未实现表格提取）。

        Args:
            result: API 返回的 result 字段

        Returns:
            list: 空列表（预留扩展）
        """
        return []

    def __images__(self, fnm, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        """
        从 PDF 生成页面缩略图，用于后续的图片裁剪功能。

        使用 pdfplumber 渲染每一页为 PIL Image 对象。

        Args:
            fnm: PDF 文件路径或二进制数据
            page_from: 起始页码
            page_to: 结束页码
            callback: 进度回调
        """
        self.page_from = page_from
        self.page_to = page_to
        try:
            with pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(BytesIO(fnm)) as pdf:
                self.pdf = pdf
                self.page_images = [p.to_image(resolution=72, antialias=True).original for i, p in enumerate(self.pdf.pages[page_from:page_to])]
        except Exception as e:
            self.page_images = None
            self.logger.exception(e)

    @staticmethod
    def extract_positions(txt: str):
        """
        从文本标签中提取位置信息。

        解析 @@页码\t左\t右\t上\t下## 格式的位置标签。

        Args:
            txt: 包含位置标签的文本

        Returns:
            list: [(页码列表, left, right, top, bottom), ...]
        """
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text: str, need_position: bool = False):
        """
        根据文本中的位置标签从 PDF 页面中裁剪对应区域图片。

        支持跨页裁剪，将多页的裁剪结果垂直拼接为一张图片。

        Args:
            text: 包含位置标签的文本
            need_position: 是否同时返回裁剪位置信息

        Returns:
            PIL.Image 或 (PIL.Image, positions) 或 None
        """
        imgs = []
        poss = self.extract_positions(text)

        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            self.logger.warning("[PaddleOCR] crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        # 过滤超出页面范围的位置
        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                self.logger.warning("[PaddleOCR] Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                self.logger.warning(f"[PaddleOCR] All page indices {pns} out of range for {page_count} pages; skipping.")
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            self.logger.warning("[PaddleOCR] No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6  # 裁剪区域之间的间距（像素）
        pos = poss[0]
        first_page_idx = pos[0][0]
        # 在首尾各插入一个上下文填充区域
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            self.logger.warning(f"[PaddleOCR] Last page index {last_page_idx} out of range for {page_count} pages; skipping crop.")
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1]
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width

            if bottom <= top:
                bottom = top + 2

            # 累加跨页高度
            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += self.page_images[pn - 1].size[1]
                else:
                    self.logger.warning(f"[PaddleOCR] Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation.")

            if not (0 <= pns[0] < page_count):
                self.logger.warning(f"[PaddleOCR] Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment.")
                continue

            # 裁剪第一页的部分
            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            x0 = max(0, min(x0, img0.size[0]))
            x1 = max(0, min(x1, img0.size[0]))
            y0 = max(0, min(y0, img0.size[1]))
            y1 = max(0, min(y1, img0.size[1]))
            if x1 <= x0 or y1 <= y0:
                continue
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))

            # 裁剪后续页面
            bottom -= img0.size[1]
            for pn in pns[1:]:
                if not (0 <= pn < page_count):
                    self.logger.warning(f"[PaddleOCR] Page index {pn} out of range for {page_count} pages during crop; skipping this page.")
                    continue
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(bottom, page.size[1]))
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                x0 = max(0, min(x0, page.size[0]))
                x1 = max(0, min(x1, page.size[0]))
                y0 = max(0, min(y0, page.size[1]))
                y1 = max(0, min(y1, page.size[1]))
                if x1 <= x0 or y1 <= y0:
                    bottom -= page.size[1]
                    continue
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                bottom -= page.size[1]

        if not imgs:
            if need_position:
                return None, None
            return

        # 将所有裁剪图片垂直拼接
        total_height = 0
        max_width = 0
        img_sizes = []
        for img in imgs:
            w, h = img.size
            img_sizes.append((w, h))
            max_width = max(max_width, w)
            total_height += h + GAP

        pic = Image.new("RGB", (max_width, int(total_height)), (245, 245, 245))
        current_height = 0
        imgs_count = len(imgs)
        for ii, (img, (w, h)) in enumerate(zip(imgs, img_sizes)):
            # 首尾图片添加半透明遮罩（标记为非主要内容区域）
            if ii == 0 or ii + 1 == imgs_count:
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 128))
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(current_height)))
            current_height += h + GAP

        if need_position:
            return pic, positions
        return pic


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = PaddleOCRParser(api_url=os.getenv("PADDLEOCR_API_URL", ""), algorithm=os.getenv("PADDLEOCR_ALGORITHM", "PaddleOCR-VL"))
    ok, reason = parser.check_installation()
    print("PaddleOCR available:", ok, reason)
