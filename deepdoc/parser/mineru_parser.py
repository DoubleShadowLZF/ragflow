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
MinerU 解析器模块。

本模块提供基于 MinerU API 的 PDF 文档解析功能。MinerU 是一款开源的
文档解析工具，支持多种后端引擎和 OCR 语言。

主要功能：
- 支持多种后端：pipeline（传统多模型流水线）、vlm-transformers、
  vlm-vllm-engine、vlm-mlx-engine、vlm-http-client 等
- 多语言 OCR 支持（中、英、日、韩、俄、阿拉伯、印地语等）
- 结构化内容提取：文本、表格、图片、公式、代码块、列表等
- 自动 VLM 图片描述生成（可选，需要视觉模型）
- 内容清洗：HTML 标签去除、空白规范化
- 安全的 ZIP 解压（路径穿越防护、符号链接检测）
"""

import json
import html
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pdfplumber
import requests
from PIL import Image
from enum import StrEnum

from deepdoc.parser.pdf_parser import RAGFlowPdfParser
from deepdoc.parser.utils import extract_pdf_outlines

from common.constants import MAXIMUM_PAGE_NUMBER

# pdfplumber 全局锁，避免多线程并发打开 PDF 时的竞态问题
LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


class MinerUContentType(StrEnum):
    """
    MinerU 支持的内容类型枚举。

    用于分类 _content_list.json 中的每个输出块。
    """

    IMAGE = "image"           # 图片
    TABLE = "table"           # 表格
    TEXT = "text"             # 文本
    EQUATION = "equation"     # 公式
    CODE = "code"             # 代码块
    LIST = "list"             # 列表
    HEADER = "header"         # 页眉
    FOOTER = "footer"         # 页脚
    PAGE_NUMBER = "page_number"  # 页码
    DISCARDED = "discarded"   # 已丢弃（水印等无意义内容）


# 语言名称到 MinerU 语言代码的映射表
LANGUAGE_TO_MINERU_MAP = {
    'English': 'en',
    'Chinese': 'ch',
    'Traditional Chinese': 'chinese_cht',
    'Russian': 'east_slavic',
    'Ukrainian': 'east_slavic',
    'Indonesian': 'latin',
    'Spanish': 'latin',
    'Vietnamese': 'latin',
    'Japanese': 'japan',
    'Korean': 'korean',
    'Portuguese BR': 'latin',
    'German': 'latin',
    'French': 'latin',
    'Italian': 'latin',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'ka',
    'Thai': 'th',
    'Greek': 'el',
    'Hindi': 'devanagari',
    'Bulgarian': 'cyrillic',
    'Turkish': 'latin',
}


class MinerUBackend(StrEnum):
    """
    MinerU 处理后端选项。

    不同后端适合不同的硬件环境和性能需求。
    """

    PIPELINE = "pipeline"                  # 传统多模型流水线（默认）
    VLM_TRANSFORMERS = "vlm-transformers"  # 使用 HuggingFace Transformers 的视觉语言模型
    VLM_MLX_ENGINE = "vlm-mlx-engine"      # Apple Silicon 加速（需要 macOS 13.5+）
    VLM_VLLM_ENGINE = "vlm-vllm-engine"    # 本地 vLLM 引擎（需要本地 GPU）
    VLM_VLLM_ASYNC_ENGINE = "vlm-vllm-async-engine"  # 异步 vLLM 引擎（MinerU API 新增）
    VLM_LMDEPLOY_ENGINE = "vlm-lmdeploy-engine"      # LMDeploy 引擎
    VLM_HTTP_CLIENT = "vlm-http-client"    # HTTP 客户端连接远程 vLLM 服务器（CPU 也可用）


class MinerULanguage(StrEnum):
    """
    MinerU 支持的 OCR 语言（仅 pipeline 后端）。

    不同语言对应不同的 OCR 模型。
    """

    CH = "ch"                # 中文
    CH_SERVER = "ch_server"  # 中文（服务器版）
    CH_LITE = "ch_lite"      # 中文（轻量版）
    EN = "en"                # 英文
    KOREAN = "korean"        # 韩文
    JAPAN = "japan"          # 日文
    CHINESE_CHT = "chinese_cht"  # 繁体中文
    TA = "ta"                # 泰米尔语
    TE = "te"                # 泰卢固语
    KA = "ka"                # 卡纳达语
    TH = "th"                # 泰语
    EL = "el"                # 希腊语
    LATIN = "latin"          # 拉丁字母语言
    ARABIC = "arabic"        # 阿拉伯语
    EAST_SLAVIC = "east_slavic"    # 东斯拉夫语（俄语、乌克兰语等）
    CYRILLIC = "cyrillic"          # 西里尔字母语言（保加利亚语等）
    DEVANAGARI = "devanagari"      # 天城文书（印地语等）


class MinerUParseMethod(StrEnum):
    """
    MinerU PDF 解析方法（仅 pipeline 后端）。

    不同的解析方法适用于不同类型的 PDF（文字型 vs 扫描型）。
    """

    AUTO = "auto"  # 自动根据文件类型确定解析方法
    TXT = "txt"    # 使用文本提取方法（适用于文字型 PDF）
    OCR = "ocr"    # 使用 OCR 方法（适用于扫描型/图片型 PDF）


@dataclass
class MinerUParseOptions:
    """
    MinerU PDF 解析选项。

    封装所有解析参数，传递给 MinerU API。
    """

    backend: MinerUBackend = MinerUBackend.PIPELINE
    lang: Optional[MinerULanguage] = None  # OCR 语言（仅 pipeline 后端）
    method: MinerUParseMethod = MinerUParseMethod.AUTO
    server_url: Optional[str] = None       # VLM 服务地址（vlm-http-client 后端必需）
    delete_output: bool = True             # 解析完成后是否删除临时输出
    parse_method: str = "raw"              # RAGFlow 内部解析方法标记
    formula_enable: bool = True            # 是否启用公式识别
    table_enable: bool = True              # 是否启用表格识别


class MinerUParser(RAGFlowPdfParser):
    """
    MinerU 解析器。

    通过 MinerU API 服务对 PDF 文档进行深度解析，支持多种后端引擎。
    继承自 RAGFlowPdfParser 以获得图片裁剪等基础能力。

    主要工作流程：
    1. 将 PDF 文件发送到 MinerU API
    2. 下载解析结果 ZIP 包
    3. 安全解压并读取 _content_list.json
    4. 将结构化内容转换为 RAGFlow 标准 sections 格式
    5. 可选：使用视觉模型为图片块生成语义描述

    配置方式：
    - 设置环境变量 MINERU_APISERVER 指向 MinerU API 服务
    - 可选设置 MINERU_SERVER_URL 用于 vlm-http-client 后端
    """

    def __init__(self, mineru_path: str = "mineru", mineru_api: str = "", mineru_server_url: str = ""):
        """
        初始化 MinerU 解析器。

        Args:
            mineru_path: MinerU 命令行路径（保留参数，当前未使用）
            mineru_api: MinerU API 服务地址
            mineru_server_url: VLM 服务器地址（用于 vlm-http-client 后端）
        """
        self.mineru_api = mineru_api.rstrip("/")
        self.mineru_server_url = mineru_server_url.rstrip("/")
        self.outlines = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _is_zipinfo_symlink(member: zipfile.ZipInfo) -> bool:
        """
        检查 ZIP 条目是否为符号链接。

        通过检查 external_attr 的文件类型位来判断。

        Args:
            member: ZIP 文件条目信息

        Returns:
            bool: 如果是符号链接返回 True
        """
        return (member.external_attr >> 16) & 0o170000 == 0o120000

    def _extract_zip_no_root(self, zip_path, extract_to, root_dir):
        """
        安全解压 ZIP 文件，自动去除根目录前缀。

        包含多层安全防护：
        - 加密条目检测
        - 符号链接检测
        - 绝对路径注入防护
        - 路径穿越攻击防护

        Args:
            zip_path: ZIP 文件路径
            extract_to: 解压目标目录
            root_dir: ZIP 内的根目录名称（将被剥离），为 None 时自动检测
        """
        self.logger.info(f"[MinerU] Extract zip: zip_path={zip_path}, extract_to={extract_to}, root_hint={root_dir}")
        base_dir = Path(extract_to).resolve()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            # 自动检测根目录：如果第一个条目以 / 结尾，则为根目录
            if not root_dir:
                if members and members[0].filename.endswith("/"):
                    root_dir = members[0].filename
                else:
                    root_dir = None
            if root_dir:
                root_dir = root_dir.replace("\\", "/")
                if not root_dir.endswith("/"):
                    root_dir += "/"

            for member in members:
                # 安全检查：不支持加密条目
                if member.flag_bits & 0x1:
                    raise RuntimeError(f"[MinerU] Encrypted zip entry not supported: {member.filename}")
                # 安全检查：不支持符号链接
                if self._is_zipinfo_symlink(member):
                    raise RuntimeError(f"[MinerU] Symlink zip entry not supported: {member.filename}")

                name = member.filename.replace("\\", "/")
                # 跳过根目录自身
                if root_dir and name == root_dir:
                    self.logger.info("[MinerU] Ignore root folder...")
                    continue
                # 剥离根目录前缀
                if root_dir and name.startswith(root_dir):
                    name = name[len(root_dir) :]
                if not name:
                    continue
                # 安全检查：防止绝对路径注入
                if name.startswith("/") or name.startswith("//") or re.match(r"^[A-Za-z]:", name):
                    raise RuntimeError(f"[MinerU] Unsafe zip path (absolute): {member.filename}")

                # 安全检查：防止路径穿越（../）
                parts = [p for p in name.split("/") if p not in ("", ".")]
                if any(p == ".." for p in parts):
                    raise RuntimeError(f"[MinerU] Unsafe zip path (traversal): {member.filename}")

                rel_path = os.path.join(*parts) if parts else ""
                dest_path = (Path(extract_to) / rel_path).resolve(strict=False)
                # 安全检查：确保不在解压目录之外
                if dest_path != base_dir and base_dir not in dest_path.parents:
                    raise RuntimeError(f"[MinerU] Unsafe zip path (escape): {member.filename}")

                if member.is_dir():
                    os.makedirs(dest_path, exist_ok=True)
                    continue

                os.makedirs(dest_path.parent, exist_ok=True)
                with zip_ref.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    @staticmethod
    def _is_http_endpoint_valid(url, timeout=5):
        """
        检查 HTTP 端点是否可达。

        通过 HEAD 请求验证 URL 的可访问性。

        Args:
            url: 待检查的 URL
            timeout: 超时时间（秒）

        Returns:
            bool: 端点可达返回 True
        """
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code in [200, 301, 302, 307, 308]
        except Exception:
            return False

    @staticmethod
    def _sanitize_section_text(section: str) -> str:
        """
        清洗 MinerU 输出文本，使其更适合分块处理。

        处理步骤：
        1. HTML 实体解码（如 &amp; -> &）
        2. 保留结构标签（br/p/div/li/tr/h1-6/table/caption -> 换行符）
        3. 去除其余 HTML 标签
        4. 压缩多余空白，保留行边界

        Args:
            section: MinerU 输出的原始文本块

        Returns:
            str: 清洗后的纯文本
        """
        if not section:
            return ""
        section = html.unescape(section)
        # 保留粗粒度结构，将块级标签转为换行
        section = re.sub(r"(?is)<\s*br\s*/?\s*>", "\n", section)
        section = re.sub(r"(?is)</\s*(p|div|li|tr|h[1-6]|table|caption)\s*>", "\n", section)
        # 去除所有剩余的 HTML 标签
        section = re.sub(r"(?is)<[^>]+>", "", section)
        # 压缩空白，保留行边界
        section = re.sub(r"[ \t]+\n", "\n", section)
        section = re.sub(r"\n{3,}", "\n\n", section)
        section = re.sub(r"[ \t]{2,}", " ", section)
        return section.strip()

    def check_installation(self, backend: str = "pipeline", server_url: Optional[str] = None) -> tuple[bool, str]:
        """
        检查 MinerU 服务是否正确安装和配置。

        验证内容：
        1. 后端类型合法性
        2. MINERU_APISERVER 是否配置
        3. API openapi.json 端点是否可达
        4. vlm-http-client 后端模式下验证 VLM 服务器是否可达

        Args:
            backend: 后端类型
            server_url: VLM 服务器地址

        Returns:
            tuple: (是否可用, 状态描述)
        """
        reason = ""

        # 验证后端类型是否合法
        valid_backends = ["pipeline", "vlm-http-client", "vlm-transformers", "vlm-vllm-engine", "vlm-mlx-engine", "vlm-vllm-async-engine", "vlm-lmdeploy-engine"]
        if backend not in valid_backends:
            reason = f"[MinerU] Invalid backend '{backend}'. Valid backends are: {valid_backends}"
            self.logger.warning(reason)
            return False, reason

        if not self.mineru_api:
            reason = "[MinerU] MINERU_APISERVER not configured."
            self.logger.warning(reason)
            return False, reason

        # 检查 API openapi.json 端点
        api_openapi = f"{self.mineru_api}/openapi.json"
        try:
            api_ok = self._is_http_endpoint_valid(api_openapi)
            self.logger.info(f"[MinerU] API openapi.json reachable={api_ok} url={api_openapi}")
            if not api_ok:
                reason = f"[MinerU] MinerU API not accessible: {api_openapi}"
                return False, reason
        except Exception as exc:
            reason = f"[MinerU] MinerU API check failed: {exc}"
            self.logger.warning(reason)
            return False, reason

        # vlm-http-client 后端需要额外的 VLM 服务器地址
        if backend == "vlm-http-client":
            resolved_server = server_url or self.mineru_server_url
            if not resolved_server:
                reason = "[MinerU] MINERU_SERVER_URL required for vlm-http-client backend."
                self.logger.warning(reason)
                return False, reason
            try:
                server_ok = self._is_http_endpoint_valid(resolved_server)
                self.logger.info(f"[MinerU] vlm-http-client server check reachable={server_ok} url={resolved_server}")
            except Exception as exc:
                self.logger.warning(f"[MinerU] vlm-http-client server probe failed: {resolved_server}: {exc}")

        return True, reason

    def _run_mineru(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> Path:
        """
        运行 MinerU 解析（调度方法，当前委托给 API 模式）。

        Args:
            input_path: 输入 PDF 文件路径
            output_dir: 输出目录
            options: 解析选项
            callback: 进度回调

        Returns:
            Path: 解析结果输出目录
        """
        return self._run_mineru_api(input_path, output_dir, options, callback)

    def _run_mineru_api(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> Path:
        """
        通过 MinerU API 运行文档解析。

        完整流程：
        1. 准备输出目录和请求参数
        2. 以 multipart/form-data 格式上传 PDF 文件
        3. 接收 ZIP 格式的解析结果
        4. 解压到输出目录

        Args:
            input_path: PDF 文件路径
            output_dir: 输出目录
            options: 解析选项
            callback: 进度回调

        Returns:
            Path: 解压后的输出目录路径

        Raises:
            RuntimeError: PDF 不存在或 API 调用失败时抛出
        """
        pdf_file_path = str(input_path)

        if not os.path.exists(pdf_file_path):
            raise RuntimeError(f"[MinerU] PDF file not exists: {pdf_file_path}")

        # 生成唯一的输出路径（带解析方法标识）
        pdf_file_name = Path(pdf_file_path).stem.strip()
        output_path = tempfile.mkdtemp(prefix=f"{pdf_file_name}_{options.method}_", dir=str(output_dir))
        output_zip_path = os.path.join(str(output_dir), f"{Path(output_path).name}.zip")

        # 构建 API 请求参数
        data = {
            "output_dir": "./output",
            "lang_list": options.lang,
            "backend": options.backend,
            "parse_method": options.method,
            "formula_enable": options.formula_enable,
            "table_enable": options.table_enable,
            "server_url": None,
            "return_md": True,
            "return_middle_json": True,
            "return_model_output": True,
            "return_content_list": True,
            "return_images": True,
            "response_format_zip": True,
            "start_page_id": 0,
            "end_page_id": 99999,
        }

        # 设置 VLM 服务器地址（优先级：options > 实例属性）
        if options.server_url:
            data["server_url"] = options.server_url
        elif self.mineru_server_url:
            data["server_url"] = self.mineru_server_url

        self.logger.info(f"[MinerU] request {data=}")
        self.logger.info(f"[MinerU] request {options=}")

        headers = {"Accept": "application/json"}
        try:
            self.logger.info(f"[MinerU] invoke api: {self.mineru_api}/file_parse backend={options.backend} server_url={data.get('server_url')}")
            if callback:
                callback(0.20, f"[MinerU] invoke api: {self.mineru_api}/file_parse")
            with open(pdf_file_path, "rb") as pdf_file:
                files = {"files": (pdf_file_name + ".pdf", pdf_file, "application/pdf")}
                with requests.post(
                    url=f"{self.mineru_api}/file_parse",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=1800,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    # 验证返回的内容类型为 ZIP
                    if not content_type.startswith("application/zip"):
                        raise RuntimeError(f"[MinerU] not zip returned from api: {content_type}")
                    self.logger.info(f"[MinerU] zip file returned, saving to {output_zip_path}...")
                    if callback:
                        callback(0.30, f"[MinerU] zip file returned, saving to {output_zip_path}...")
                    # 流式保存 ZIP 文件
                    with open(output_zip_path, "wb") as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    self.logger.info(f"[MinerU] Unzip to {output_path}...")
                    # 安全解压
                    self._extract_zip_no_root(output_zip_path, output_path, pdf_file_name + "/")
                    if callback:
                        callback(0.40, f"[MinerU] Unzip to {output_path}...")
            self.logger.info("[MinerU] Api completed successfully.")
            return Path(output_path)
        except requests.RequestException as e:
            raise RuntimeError(f"[MinerU] api failed with exception {e}")

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        """
        从 PDF 渲染页面图片，供裁剪功能使用。

        使用 pdfplumber 渲染每一页为 PIL Image 对象。

        Args:
            fnm: PDF 文件路径或二进制数据
            zoomin: 缩放因子（默认 1 = 72 DPI）
            page_from: 起始页码
            page_to: 结束页码
            callback: 进度回调
        """
        self.page_from = page_from
        self.page_to = page_to
        try:
            with pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(BytesIO(fnm)) as pdf:
                self.pdf = pdf
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for _, p in
                                    enumerate(self.pdf.pages[page_from:page_to])]
        except Exception as e:
            self.page_images = None
            self.total_page = 0
            self.logger.exception(e)

    def _line_tag(self, bx):
        """
        根据 MinerU 输出的边界框生成 RAGFlow 位置标签。

        MinerU 使用归一化坐标（0-1000），需要转换为像素坐标。

        Args:
            bx: MinerU 输出块，包含 page_idx 和 bbox 字段

        Returns:
            str: 格式为 @@页码\t左\t右\t上\t下## 的位置标签
        """
        pn = [bx["page_idx"] + 1]
        positions = bx.get("bbox", (0, 0, 0, 0))
        x0, top, x1, bott = positions
        # 修正翻转坐标（MinerU 可能为翻转图片输出颠倒的 bbox）
        if x0 > x1:
            x0, x1 = x1, x0
        if top > bott:
            top, bott = bott, top

        # 将归一化坐标（0-1000）转换为像素坐标
        if hasattr(self, "page_images") and self.page_images and len(self.page_images) > bx["page_idx"]:
            page_width, page_height = self.page_images[bx["page_idx"]].size
            x0 = (x0 / 1000.0) * page_width
            x1 = (x1 / 1000.0) * page_width
            top = (top / 1000.0) * page_height
            bott = (bott / 1000.0) * page_height

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), x0, x1, top, bott)

    def crop(self, text, ZM=1, need_position=False):
        """
        根据位置标签从页面裁剪图片区域。

        支持跨页裁剪，将多页裁剪结果垂直拼接为一张图片。

        Args:
            text: 包含位置标签的文本
            ZM: 缩放因子（保留接口兼容性，当前未使用）
            need_position: 是否同时返回位置信息

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
            self.logger.warning("[MinerU] crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        # 过滤超出页面范围的位置
        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                self.logger.warning("[MinerU] Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                self.logger.warning(f"[MinerU] All page indices {pns} out of range for {page_count} pages; skipping.")
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            self.logger.warning("[MinerU] No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        # 在首尾插入上下文填充区域
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            self.logger.warning(
                f"[MinerU] Last page index {last_page_idx} out of range for {page_count} pages; skipping crop.")
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
                    self.logger.warning(
                        f"[MinerU] Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation.")

            if not (0 <= pns[0] < page_count):
                self.logger.warning(
                    f"[MinerU] Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment.")
                continue

            # 裁剪第一页
            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
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
                    self.logger.warning(
                        f"[MinerU] Page index {pn} out of range for {page_count} pages during crop; skipping this page.")
                    continue
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(bottom, page.size[1]))
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
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

        # 垂直拼接所有裁剪图片
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            # 首尾图片添加半透明遮罩
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    @staticmethod
    def extract_positions(txt: str):
        """
        从文本标签中提取位置信息。

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

    def _read_output(self, output_dir: Path, file_stem: str, method: str = "auto", backend: str = "pipeline") -> list[
        dict[str, Any]]:
        """
        从 MinerU 输出目录读取解析结果。

        支持多种目录结构：
        - 直接输出：output_dir/<stem>_content_list.json
        - 嵌套输出：output_dir/<stem>/<stem>_content_list.json
        - vlm 子目录：output_dir/vlm/<stem>_content_list.json
        - 解析方法子目录：output_dir/**/<method>/<stem>_content_list.json

        Args:
            output_dir: MinerU 输出根目录
            file_stem: PDF 文件名（不含扩展名）
            method: 解析方法（用于在子目录中定位）
            backend: 后端类型（用于推断可能的子目录名）

        Returns:
            list[dict]: 内容列表，每个元素已补全图片路径

        Raises:
            FileNotFoundError: 找不到输出文件时抛出
        """
        json_file = None
        subdir = None
        attempted = []

        # 镜像 MinerU 的文件名清洗逻辑以对齐 ZIP 命名
        def _sanitize_filename(name: str) -> str:
            sanitized = re.sub(r"[/\\\.]{2,}|[/\\]", "", name)
            sanitized = re.sub(r"[^\w.-]", "_", sanitized, flags=re.UNICODE)
            if sanitized.startswith("."):
                sanitized = "_" + sanitized[1:]
            return sanitized or "unnamed"

        safe_stem = _sanitize_filename(file_stem)
        content_names = tuple(dict.fromkeys((f"{file_stem}_content_list.json", f"{safe_stem}_content_list.json")))
        allowed_names = set(content_names)
        self.logger.info(f"[MinerU] Expected output files: {', '.join(sorted(allowed_names))}")
        self.logger.info(f"[MinerU] Searching output in: {output_dir}")

        # 策略 1：直接输出目录
        jf = output_dir / f"{file_stem}_content_list.json"
        self.logger.info(f"[MinerU] Trying original path: {jf}")
        attempted.append(jf)
        if jf.exists():
            subdir = output_dir
            json_file = jf
        else:
            alt = output_dir / f"{safe_stem}_content_list.json"
            self.logger.info(f"[MinerU] Trying sanitized filename: {alt}")
            attempted.append(alt)
            if alt.exists():
                subdir = output_dir
                json_file = alt
            else:
                # 策略 2：嵌套目录（MinerU 可能将结果放在子目录中）
                nested_alt = output_dir / safe_stem / f"{safe_stem}_content_list.json"
                self.logger.info(f"[MinerU] Trying sanitized nested path: {nested_alt}")
                attempted.append(nested_alt)
                if nested_alt.exists():
                    subdir = nested_alt.parent
                    json_file = nested_alt
                else:
                    # 策略 3：vlm 子目录（vlm-http-client 后端使用）
                    vlm_path = output_dir / "vlm" / f"{file_stem}_content_list.json"
                    self.logger.info(f"[MinerU] Trying vlm subdirectory: {vlm_path}")
                    attempted.append(vlm_path)
                    if vlm_path.exists():
                        subdir = vlm_path.parent
                        json_file = vlm_path
                    else:
                        vlm_safe = output_dir / "vlm" / f"{safe_stem}_content_list.json"
                        self.logger.info(f"[MinerU] Trying vlm subdirectory with sanitized name: {vlm_safe}")
                        attempted.append(vlm_safe)
                        if vlm_safe.exists():
                            subdir = vlm_safe.parent
                            json_file = vlm_safe

        # 策略 4：解析方法子目录（如 auto/ocr/txt 或 vlm）
        if not json_file:
            parse_subdir = None
            if backend.startswith("pipeline"):
                parse_subdir = method
            elif backend.startswith("hybrid"):
                parse_subdir = f"hybrid_{method}"
            elif backend.startswith("vlm"):
                parse_subdir = "vlm"

            if parse_subdir:
                for content_name in content_names:
                    for candidate in output_dir.glob(f"**/{parse_subdir}/{content_name}"):
                        self.logger.info(f"[MinerU] Trying parse-method path: {candidate}")
                        attempted.append(candidate)
                        subdir = candidate.parent
                        json_file = candidate
                        break
                    if json_file:
                        break

        # 策略 5：广泛搜索回退（以文件名和 stem 匹配过滤）
        if not json_file:
            stem_dirs = tuple(dict.fromkeys((file_stem, safe_stem)))
            patterns = []
            if parse_subdir:
                for stem_dir in stem_dirs:
                    patterns.extend(
                        [
                            f"**/{stem_dir}/{parse_subdir}/content_list.json",
                            f"**/{stem_dir}/{parse_subdir}/*_content_list.json",
                        ]
                    )
                patterns.extend(
                    [
                        f"**/{parse_subdir}/content_list.json",
                        f"**/{parse_subdir}/*_content_list.json",
                    ]
                )
            for stem_dir in stem_dirs:
                patterns.extend(
                    [
                        f"**/{stem_dir}/content_list.json",
                        f"**/{stem_dir}/*_content_list.json",
                    ]
                )
            patterns.extend(["**/content_list.json", "**/*_content_list.json"])

            for pattern in patterns:
                for candidate in sorted(output_dir.glob(pattern)):
                    self.logger.info(f"[MinerU] Trying fallback path: {candidate}")
                    if candidate.name.endswith("_content_list.json"):
                        rel_parts = candidate.relative_to(output_dir).parts
                        in_stem_dir = any(stem_dir in rel_parts for stem_dir in stem_dirs)
                        stem_match = candidate.stem.startswith(file_stem) or candidate.stem.startswith(safe_stem)
                        # 跳过不相关的文件
                        if not (stem_match or in_stem_dir):
                            self.logger.info(f"[MinerU] Skip unrelated fallback candidate: {candidate}")
                            continue
                    attempted.append(candidate)
                    subdir = candidate.parent
                    json_file = candidate
                    break
                if json_file:
                    break

        if not json_file:
            raise FileNotFoundError(f"[MinerU] Missing output file, tried: {', '.join(str(p) for p in attempted)}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 将相对路径补全为绝对路径
        for item in data:
            for key in ("img_path", "table_img_path", "equation_img_path"):
                if key in item and item[key]:
                    item[key] = str((subdir / item[key]).resolve())
        return data

    def _transfer_to_sections(self, outputs: list[dict[str, Any]], parse_method: str = None):
        """
        将 MinerU 输出转换为 RAGFlow 标准 sections 格式。

        根据内容类型做不同的文本提取策略：
        - text: 直接使用 text 字段
        - table: 拼接 table_body + table_caption + table_footnote
        - image: 拼接 image_caption + image_footnote + vlm_description
        - equation: 使用 text 字段
        - code: 拼接 code_body + code_caption
        - list: 用换行连接 list_items
        - header/footer/page_number/discarded: 跳过

        Args:
            outputs: MinerU 的内容列表
            parse_method: 解析方法 ("raw" / "manual" / "pipeline" / "paper")

        Returns:
            list[tuple]: sections 列表
        """
        sections = []
        for output in outputs:
            match output.get("type"):
                case MinerUContentType.TEXT:
                    section = output.get("text", "")
                case MinerUContentType.TABLE:
                    section = output.get("table_body", "") + "\n".join(output.get("table_caption", [])) + "\n".join(
                        output.get("table_footnote", []))
                    if not section.strip():
                        section = "FAILED TO PARSE TABLE"
                case MinerUContentType.IMAGE:
                    section = "".join(output.get("image_caption", [])) + "\n" + "".join(
                        output.get("image_footnote", []))
                    # 如果视觉模型为此图片生成了语义描述（见 _enhance_images_with_vlm），
                    # 将其嵌入到 chunk 中，使其可被检索
                    vlm_description = (output.get("vlm_description") or "").strip()
                    if vlm_description:
                        section = (section.strip("\n") + "\n" + vlm_description).strip("\n") if section.strip() else vlm_description
                case MinerUContentType.EQUATION:
                    section = output.get("text", "")
                case MinerUContentType.CODE:
                    section = output.get("code_body", "") + "\n".join(output.get("code_caption", []))
                case MinerUContentType.LIST:
                    section = "\n".join(output.get("list_items", []))
                case (
                    MinerUContentType.HEADER
                    | MinerUContentType.FOOTER
                    | MinerUContentType.PAGE_NUMBER
                    | MinerUContentType.DISCARDED
                ):
                    # 页眉、页脚、页码、废弃内容不进入最终输出
                    continue
                case _:
                    self.logger.debug("[MinerU] Skip unsupported section type=%s", output.get("type"))
                    continue

            section = self._sanitize_section_text(section)
            if not section:
                self.logger.debug("[MinerU] Skip section after sanitization: type=%s", output.get("type"))
                continue

            if section and parse_method in {"manual", "pipeline"}:
                sections.append((section, output["type"], self._line_tag(output)))
            elif section and parse_method == "paper":
                sections.append((section + self._line_tag(output), output["type"]))
            else:
                sections.append((section, self._line_tag(output)))
        return sections

    def _transfer_to_tables(self, outputs: list[dict[str, Any]]):
        """
        将 MinerU 输出转换为 tables 格式（当前版本暂未实现）。

        Args:
            outputs: MinerU 的内容列表

        Returns:
            list: 空列表（预留扩展）
        """
        return []

    def _enhance_images_with_vlm(self, outputs: list[dict[str, Any]], vision_model, callback: Optional[Callable] = None):
        """
        使用视觉语言模型为图片块生成语义描述。

        通过租户的 IMAGE2TEXT 模型为每个 IMAGE 类型的块生成描述文本，
        这些描述会被 _transfer_to_sections 嵌入到 chunk 中，
        使得图片内容可被全文检索（解决 issue #14869）。

        支持并发处理（最多 10 个工作线程）。

        Args:
            outputs: MinerU 的内容列表（原地修改，添加 vlm_description 字段）
            vision_model: 视觉语言模型实例
            callback: 进度回调
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from rag.app.picture import vision_llm_chunk
        from rag.prompts.generator import vision_llm_figure_describe_prompt

        # 筛选可处理的图片块
        image_jobs = [
            (idx, item)
            for idx, item in enumerate(outputs)
            if item.get("type") == MinerUContentType.IMAGE
            and item.get("img_path")
            and os.path.exists(item["img_path"])
        ]
        if not image_jobs:
            return

        if callback:
            callback(0.78, f"[MinerU] Generating VLM descriptions for {len(image_jobs)} images...")

        prompt = vision_llm_figure_describe_prompt()

        def worker(idx, item):
            """单个图片的 VLM 描述生成工作函数。"""
            try:
                with Image.open(item["img_path"]) as img:
                    img.load()
                    desc = vision_llm_chunk(binary=img, vision_model=vision_model, prompt=prompt)
                return idx, (desc or "").strip()
            except Exception as e:
                logging.warning(f"[MinerU] VLM description failed for image #{idx}: {e}")
                return idx, ""

        # 并发处理所有图片
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, idx, item) for idx, item in image_jobs]
            for fut in as_completed(futures):
                idx, desc = fut.result()
                if desc:
                    outputs[idx]["vlm_description"] = desc

    def parse_pdf(
            self,
            filepath: str | PathLike[str],
            binary: BytesIO | bytes,
            callback: Optional[Callable] = None,
            *,
            output_dir: Optional[str] = None,
            backend: str = "pipeline",
            server_url: Optional[str] = None,
            delete_output: bool = True,
            parse_method: str = "raw",
            **kwargs,
    ) -> tuple:
        """
        解析 PDF 文档（MinerU 主入口方法）。

        完整流程：
        1. 提取 PDF 目录大纲
        2. 处理输入文件（二进制数据写入临时文件）
        3. 渲染页面缩略图
        4. 调用 MinerU API 进行文档解析
        5. 读取并解析输出 JSON
        6. 可选：使用 VLM 为图片生成语义描述
        7. 转换为标准 sections/tables 格式
        8. 清理临时文件

        Args:
            filepath: PDF 文件路径
            binary: PDF 二进制数据
            callback: 进度回调 (progress, message)
            output_dir: 输出目录，默认为临时目录
            backend: MinerU 后端类型
            server_url: VLM 服务器地址
            delete_output: 是否删除临时输出
            parse_method: 解析方法 ("raw" / "manual" / "pipeline" / "paper")
            **kwargs: 可包含 parser_config（mineru_lang, mineru_parse_method,
                      mineru_formula_enable, mineru_table_enable) 和 vision_model

        Returns:
            tuple: (sections 列表, tables 列表)
        """
        import shutil

        self.outlines = extract_pdf_outlines(binary if binary is not None else filepath)
        temp_pdf = None
        created_tmp_dir = False

        # 从 parser_config 或 kwargs 中提取 MinerU 特定参数
        parser_cfg = kwargs.get('parser_config', {})
        lang = parser_cfg.get('mineru_lang') or kwargs.get('lang', 'English')
        mineru_lang_code = LANGUAGE_TO_MINERU_MAP.get(lang, 'ch')  # 未匹配时默认中文
        mineru_method_raw_str = parser_cfg.get('mineru_parse_method', 'auto')
        enable_formula = parser_cfg.get('mineru_formula_enable', True)
        enable_table = parser_cfg.get('mineru_table_enable', True)

        # 去除文件名中的空格（MinerU 对空格敏感，_read_output 也可能失败）
        file_path = Path(filepath)
        pdf_file_name = file_path.stem.replace(" ", "") + ".pdf"
        pdf_file_path_valid = os.path.join(file_path.parent, pdf_file_name)

        # 处理二进制输入：写入临时文件
        if binary:
            temp_dir = Path(tempfile.mkdtemp(prefix="mineru_bin_pdf_"))
            temp_pdf = temp_dir / pdf_file_name
            with open(temp_pdf, "wb") as f:
                f.write(binary)
            pdf = temp_pdf
            self.logger.info(f"[MinerU] Received binary PDF -> {temp_pdf}")
            if callback:
                callback(0.15, f"[MinerU] Received binary PDF -> {temp_pdf}")
        else:
            if pdf_file_path_valid != filepath:
                self.logger.info(f"[MinerU] Remove all space in file name: {pdf_file_path_valid}")
                shutil.move(filepath, pdf_file_path_valid)
            pdf = Path(pdf_file_path_valid)
            if not pdf.exists():
                if callback:
                    callback(-1, f"[MinerU] PDF not found: {pdf}")
                raise FileNotFoundError(f"[MinerU] PDF not found: {pdf}")

        # 准备输出目录
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="mineru_pdf_"))
            created_tmp_dir = True

        self.logger.info(f"[MinerU] Output directory: {out_dir} backend={backend} api={self.mineru_api} server_url={server_url or self.mineru_server_url}")
        if callback:
            callback(0.15, f"[MinerU] Output directory: {out_dir}")

        # 渲染页面缩略图（供裁剪功能使用）
        self.__images__(pdf, zoomin=1)

        try:
            # 构建解析选项并运行 MinerU
            options = MinerUParseOptions(
                backend=MinerUBackend(backend),
                lang=MinerULanguage(mineru_lang_code),
                method=MinerUParseMethod(mineru_method_raw_str),
                server_url=server_url,
                delete_output=delete_output,
                parse_method=parse_method,
                formula_enable=enable_formula,
                table_enable=enable_table,
            )
            final_out_dir = self._run_mineru(pdf, out_dir, options, callback=callback)
            outputs = self._read_output(final_out_dir, pdf.stem, method=mineru_method_raw_str, backend=backend)
            self.logger.info(f"[MinerU] Parsed {len(outputs)} blocks from PDF.")
            if callback:
                callback(0.75, f"[MinerU] Parsed {len(outputs)} blocks from PDF.")

            # 可选的 VLM 图片语义描述生成
            vision_model = kwargs.get("vision_model")
            if vision_model is not None:
                try:
                    self._enhance_images_with_vlm(outputs, vision_model, callback=callback)
                except Exception as e:
                    self.logger.warning(f"[MinerU] VLM image enhancement failed: {e}. Continuing without descriptions.")

            return self._transfer_to_sections(outputs, parse_method), self._transfer_to_tables(outputs)
        finally:
            # 清理临时文件
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                    temp_pdf.parent.rmdir()
                except Exception:
                    pass
            if delete_output and created_tmp_dir and out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = MinerUParser("mineru")
    ok, reason = parser.check_installation()
    print("MinerU available:", ok)

    filepath = ""
    with open(filepath, "rb") as file:
        outputs = parser.parse_pdf(filepath=filepath, binary=file.read())
        for output in outputs:
            print(output)
