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
腾讯云文档解析（TCADP）解析器模块。

本模块提供基于腾讯云知识引擎原子能力（LKEAP）的 PDF 文档解析功能。
通过腾讯云官方 SDK 调用 ReconstructDocumentSSE 接口，支持：
- 服务端流式（SSE）和非流式响应处理
- Base64 编码文件上传
- 解析结果 ZIP 包下载与内容提取
- 自动重试与指数退避
- 表格结果类型、Markdown 图片响应类型等参数配置
"""

import base64
import json
import logging
import os
import re
import shutil
import tempfile
import time
import traceback
import types
import zipfile
from datetime import datetime
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.lkeap.v20240522 import lkeap_client, models

from common.config_utils import get_base_config
from deepdoc.parser.pdf_parser import RAGFlowPdfParser
from deepdoc.parser.utils import extract_pdf_outlines


class TencentCloudAPIClient:
    """
    腾讯云 API 客户端，封装官方 SDK 的文档解析接口调用。

    负责：
    - 使用 SecretId/SecretKey 创建认证凭据
    - 调用 ReconstructDocumentSSE 接口进行文档解析
    - 处理流式（SSE）和非流式两种响应模式
    - 下载解析结果 ZIP 文件
    """

    def __init__(self, secret_id, secret_key, region):
        """
        初始化腾讯云 API 客户端。

        Args:
            secret_id: 腾讯云 API 密钥 ID
            secret_key: 腾讯云 API 密钥 Key
            region: 腾讯云服务区域，如 ap-guangzhou
        """
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.outlines = []

        # 创建认证凭据对象
        self.cred = credential.Credential(secret_id, secret_key)

        # 配置 HTTP 请求端点（可选，无特殊需求可跳过）
        self.httpProfile = HttpProfile()
        self.httpProfile.endpoint = "lkeap.tencentcloudapi.com"

        # 配置客户端参数（可选，无特殊需求可跳过）
        self.clientProfile = ClientProfile()
        self.clientProfile.httpProfile = self.httpProfile

        # 实例化要请求的产品的 client 对象，clientProfile 可选
        self.client = lkeap_client.LkeapClient(self.cred, region, self.clientProfile)

    def reconstruct_document_sse(self, file_type, file_url=None, file_base64=None, file_start_page=1, file_end_page=1000, config=None):
        """
        调用腾讯云文档解析 API（ReconstructDocumentSSE）。

        支持通过文件 URL 或 Base64 编码内容两种方式提交文档。

        Args:
            file_type: 文件类型，如 "PDF"
            file_url: 文件的 URL 地址（与 file_base64 二选一）
            file_base64: 文件的 Base64 编码内容（与 file_url 二选一）
            file_start_page: 起始页码，默认为 1
            file_end_page: 结束页码，默认为 1000
            config: 额外配置参数，如 TableResultType、MarkdownImageResponseType

        Returns:
            dict: 解析结果，包含 DocumentRecognizeResultUrl 下载链接等字段；
                  失败时返回 None
        """
        try:
            # 实例化请求对象，每个接口对应一个请求对象
            req = models.ReconstructDocumentSSERequest()

            # 构建请求参数
            params = {
                "FileType": file_type,
                "FileStartPageNumber": file_start_page,
                "FileEndPageNumber": file_end_page,
            }

            # 根据腾讯云 API 文档，FileUrl 或 FileBase64 参数必须二选一，同时提供时仅 FileUrl 生效
            if file_url:
                params["FileUrl"] = file_url
                logging.info(f"[TCADP] Using file URL: {file_url}")
            elif file_base64:
                params["FileBase64"] = file_base64
                logging.info(f"[TCADP] Using Base64 data, length: {len(file_base64)} characters")
            else:
                raise ValueError("Must provide either FileUrl or FileBase64 parameter")

            if config:
                params["Config"] = config

            req.from_json_string(json.dumps(params))

            # 调用 API，返回的 resp 是 ReconstructDocumentSSEResponse 实例
            resp = self.client.ReconstructDocumentSSE(req)
            parser_result = {}

            # 处理流式响应（SSE）：逐个接收事件，直到进度达到 100%
            if isinstance(resp, types.GeneratorType):
                logging.info("[TCADP] Detected streaming response")
                for event in resp:
                    logging.info(f"[TCADP] Received event: {event}")
                    if event.get('data'):
                        try:
                            data_dict = json.loads(event['data'])
                            logging.info(f"[TCADP] Parsed data: {data_dict}")

                            if data_dict.get('Progress') == "100":
                                parser_result = data_dict
                                logging.info("[TCADP] Document parsing completed!")
                                logging.info(f"[TCADP] Task ID: {data_dict.get('TaskId')}")
                                logging.info(f"[TCADP] Success pages: {data_dict.get('SuccessPageNum')}")
                                logging.info(f"[TCADP] Failed pages: {data_dict.get('FailPageNum')}")

                                # 打印失败页面信息
                                failed_pages = data_dict.get("FailedPages", [])
                                if failed_pages:
                                    logging.warning("[TCADP] Failed parsing pages:")
                                    for page in failed_pages:
                                        logging.warning(f"[TCADP]   Page number: {page.get('PageNumber')}, Error: {page.get('ErrorMsg')}")

                                # 检查是否有下载链接
                                download_url = data_dict.get("DocumentRecognizeResultUrl")
                                if download_url:
                                    logging.info(f"[TCADP] Got download link: {download_url}")
                                else:
                                    logging.warning("[TCADP] No download link obtained")

                                break  # 已找到最终结果，退出循环
                            else:
                                # 打印进度信息
                                progress = data_dict.get("Progress", "0")
                                logging.info(f"[TCADP] Progress: {progress}%")
                        except json.JSONDecodeError as e:
                            logging.error(f"[TCADP] Failed to parse JSON data: {e}")
                            logging.error(f"[TCADP] Raw data: {event.get('data')}")
                            continue
                    else:
                        logging.info(f"[TCADP] Event without data: {event}")
            else:
                # 处理非流式响应
                logging.info("[TCADP] Detected non-streaming response")
                if hasattr(resp, 'data') and resp.data:
                    try:
                        data_dict = json.loads(resp.data)
                        parser_result = data_dict
                        logging.info(f"[TCADP] JSON parsing successful: {parser_result}")
                    except json.JSONDecodeError as e:
                        logging.error(f"[TCADP] JSON parsing failed: {e}")
                        return None
                else:
                    logging.error("[TCADP] No data in response")
                    return None

            return parser_result

        except TencentCloudSDKException as err:
            logging.error(f"[TCADP] Tencent Cloud SDK error: {err}")
            return None
        except Exception as e:
            logging.error(f"[TCADP] Unknown error: {e}")
            logging.error(f"[TCADP] Error stack trace: {traceback.format_exc()}")
            return None

    def download_result_file(self, download_url, output_dir):
        """
        下载解析结果 ZIP 文件。

        Args:
            download_url: 解析结果的下载 URL
            output_dir: 输出目录路径

        Returns:
            str: 下载文件的本地路径；失败时返回 None
        """
        if not download_url:
            logging.warning("[TCADP] No downloadable result file")
            return None

        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tcadp_result_{timestamp}.zip"
            file_path = os.path.join(output_dir, filename)

            # 流式下载，避免大文件占用过多内存
            with requests.get(download_url, stream=True) as response:
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)

            logging.info(f"[TCADP] Document parsing result downloaded to: {os.path.basename(file_path)}")
            return file_path

        except Exception as e:
            logging.error(f"[TCADP] Failed to download file: {e}")
            # 下载失败时清理可能不完整的文件
            try:
                if "file_path" in locals() and os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
            return None


class TCADPParser(RAGFlowPdfParser):
    """
    腾讯云文档解析器，继承自 RAGFlowPdfParser。

    使用腾讯云知识引擎原子能力（LKEAP）对 PDF 文档进行结构化解析，
    返回标准化的 sections 和 tables 格式供下游 Pipeline 使用。

    配置方式：
    - 在 service_conf.yaml 中配置 tcadp_config 段，包含 secret_id、secret_key、region 等参数
    - 也可以通过构造函数参数直接传入

    主要流程：
    1. 将 PDF 文件编码为 Base64
    2. 调用腾讯云文档解析 API（支持重试）
    3. 下载并解压解析结果 ZIP 包
    4. 将 JSON/Markdown 结果转换为 sections 和 tables 格式
    """

    def __init__(self, secret_id: str = None, secret_key: str = None, region: str = "ap-guangzhou",
                 table_result_type: str = None, markdown_image_response_type: str = None):
        """
        初始化 TCADP 解析器。

        参数优先级：构造函数参数 > service_conf.yaml 配置文件 > 默认值

        Args:
            secret_id: 腾讯云 API 密钥 ID
            secret_key: 腾讯云 API 密钥 Key
            region: 腾讯云服务区域，默认 ap-guangzhou（广州）
            table_result_type: 表格结果类型，"0"=Markdown，"1"=HTML
            markdown_image_response_type: Markdown 图片响应类型
        """
        super().__init__()

        # 初始化 logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"[TCADP] Initializing with parameters - table_result_type: {table_result_type}, markdown_image_response_type: {markdown_image_response_type}")

        # 优先从 RAGFlow 配置系统（service_conf.yaml）读取配置
        try:
            tcadp_parser = get_base_config("tcadp_config", {})
            if isinstance(tcadp_parser, dict) and tcadp_parser:
                self.secret_id = secret_id or tcadp_parser.get("secret_id")
                self.secret_key = secret_key or tcadp_parser.get("secret_key")
                self.region = region or tcadp_parser.get("region", "ap-guangzhou")
                # 从配置或参数中设置表格结果类型和 Markdown 图片响应类型
                self.table_result_type = table_result_type if table_result_type is not None else tcadp_parser.get("table_result_type", "1")
                self.markdown_image_response_type = markdown_image_response_type if markdown_image_response_type is not None else tcadp_parser.get("markdown_image_response_type", "1")

            else:
                self.logger.error("[TCADP] Please configure tcadp_config in service_conf.yaml first")
                # 配置文件为空时，使用传入参数或默认值
                self.secret_id = secret_id
                self.secret_key = secret_key
                self.region = region or "ap-guangzhou"
                self.table_result_type = table_result_type if table_result_type is not None else "1"
                self.markdown_image_response_type = markdown_image_response_type if markdown_image_response_type is not None else "1"

        except ImportError:
            self.logger.info("[TCADP] Configuration module import failed")
            # 配置模块不可用时，使用传入参数或默认值
            self.secret_id = secret_id
            self.secret_key = secret_key
            self.region = region or "ap-guangzhou"
            self.table_result_type = table_result_type if table_result_type is not None else "1"
            self.markdown_image_response_type = markdown_image_response_type if markdown_image_response_type is not None else "1"

        self.logger.info(f"[TCADP] Final values - table_result_type: {self.table_result_type}, markdown_image_response_type: {self.markdown_image_response_type}")

        if not self.secret_id or not self.secret_key:
            raise ValueError("[TCADP] Please set Tencent Cloud API keys, configure tcadp_config in service_conf.yaml")

    @staticmethod
    def _is_zipinfo_symlink(member: zipfile.ZipInfo) -> bool:
        """
        检查 ZIP 条目是否为符号链接。

        通过检查 external_attr 的文件类型位来判断，防止 ZIP 解压时跟随恶意符号链接。

        Args:
            member: ZIP 文件条目信息

        Returns:
            bool: 如果是符号链接返回 True，否则返回 False
        """
        return (member.external_attr >> 16) & 0o170000 == 0o120000

    def check_installation(self) -> bool:
        """
        检查腾讯云 API 配置是否正确。

        通过尝试创建客户端对象来验证 secret_id 和 secret_key 的有效性。

        Returns:
            bool: 配置正确返回 True，否则返回 False
        """
        try:
            # 检查必要的配置参数
            if not self.secret_id or not self.secret_key:
                self.logger.error("[TCADP] Tencent Cloud API configuration incomplete")
                return False

            # 尝试创建客户端来验证配置
            TencentCloudAPIClient(self.secret_id, self.secret_key, self.region)
            self.logger.info("[TCADP] Tencent Cloud API configuration check passed")
            return True
        except Exception as e:
            self.logger.error(f"[TCADP] Tencent Cloud API configuration check failed: {e}")
            return False

    def _file_to_base64(self, file_path: str, binary: bytes = None) -> str:
        """
        将文件转换为 Base64 编码字符串。

        Args:
            file_path: 文件路径
            binary: 文件的二进制数据（如果已读入内存）。为 None 时从 file_path 读取

        Returns:
            str: 文件的 Base64 编码字符串
        """
        if binary:
            # 如果已有二进制数据，直接编码
            return base64.b64encode(binary).decode('utf-8')
        else:
            # 从文件路径读取后编码
            with open(file_path, 'rb') as f:
                file_data = f.read()
                return base64.b64encode(file_data).decode('utf-8')

    def _extract_content_from_zip(self, zip_path: str) -> list[dict[str, Any]]:
        """
        从下载的 ZIP 文件中提取解析结果。

        支持 JSON 和 Markdown 两种格式的内容文件。
        包含路径穿越检查和符号链接检查等安全防护。

        Args:
            zip_path: ZIP 文件路径

        Returns:
            list[dict]: 解析结果列表，每个元素是一个内容块字典
        """
        results = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                members = zip_file.infolist()
                for member in members:
                    name = member.filename.replace("\\", "/")
                    # 跳过目录条目
                    if member.is_dir():
                        continue
                    # 安全检查：加密条目不支持
                    if member.flag_bits & 0x1:
                        raise RuntimeError(f"[TCADP] Encrypted zip entry not supported: {member.filename}")
                    # 安全检查：符号链接不支持
                    if self._is_zipinfo_symlink(member):
                        raise RuntimeError(f"[TCADP] Symlink zip entry not supported: {member.filename}")
                    # 安全检查：防止绝对路径注入
                    if name.startswith("/") or name.startswith("//") or re.match(r"^[A-Za-z]:", name):
                        raise RuntimeError(f"[TCADP] Unsafe zip path (absolute): {member.filename}")
                    # 安全检查：防止路径穿越攻击
                    parts = [p for p in name.split("/") if p not in ("", ".")]
                    if any(p == ".." for p in parts):
                        raise RuntimeError(f"[TCADP] Unsafe zip path (traversal): {member.filename}")

                    # 只处理 JSON 和 Markdown 文件
                    if not (name.endswith(".json") or name.endswith(".md")):
                        continue

                    with zip_file.open(member) as f:
                        if name.endswith(".json"):
                            # JSON 文件：解析后追加到结果列表
                            data = json.load(f)
                            if isinstance(data, list):
                                results.extend(data)
                            else:
                                results.append(data)
                        else:
                            # Markdown 文件：读取文本内容
                            content = f.read().decode("utf-8")
                            results.append({"type": "text", "content": content, "file": name})

        except Exception as e:
            self.logger.error(f"[TCADP] Failed to extract ZIP file content: {e}")

        return results

    def _parse_content_to_sections(self, content_data: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """
        将解析结果转换为 RAGFlow 标准 sections 格式。

        根据内容类型（text/paragraph/table/image/equation）做不同的拼接处理。

        Args:
            content_data: 从 ZIP 中提取的内容块列表

        Returns:
            list[tuple]: (文本内容, 位置标签) 的元组列表
        """
        sections = []

        for item in content_data:
            content_type = item.get("type", "text")
            content = item.get("content", "")

            if not content:
                continue

            # 根据内容类型处理文本
            if content_type == "text" or content_type == "paragraph":
                section_text = content
            elif content_type == "table":
                # 处理表格内容：将行数据转为管道符分隔的文本
                table_data = item.get("table_data", {})
                if isinstance(table_data, dict):
                    rows = table_data.get("rows", [])
                    section_text = "\n".join([" | ".join(row) for row in rows])
                else:
                    section_text = str(table_data)
            elif content_type == "image":
                # 处理图片内容：保留标题信息
                caption = item.get("caption", "")
                section_text = f"[Image] {caption}" if caption else "[Image]"
            elif content_type == "equation":
                # 处理公式内容：用 LaTeX 格式包裹
                section_text = f"$${content}$$"
            else:
                section_text = content

            if section_text.strip():
                # 生成位置标签（简化版本，使用固定坐标占位）
                position_tag = "@@1\t0.0\t1000.0\t0.0\t100.0##"
                sections.append((section_text, position_tag))

        return sections

    def _parse_content_to_tables(self, content_data: list[dict[str, Any]]) -> list:
        """
        将解析结果转换为 RAGFlow 标准 tables 格式（HTML 表格）。

        Args:
            content_data: 从 ZIP 中提取的内容块列表

        Returns:
            list: HTML 表格字符串列表
        """
        tables = []

        for item in content_data:
            if item.get("type") == "table":
                table_data = item.get("table_data", {})
                if isinstance(table_data, dict):
                    rows = table_data.get("rows", [])
                    if rows:
                        # 将行数据转换为 HTML 表格，第一行作为表头
                        table_html = "<table>\n"
                        for i, row in enumerate(rows):
                            table_html += "  <tr>\n"
                            for cell in row:
                                tag = "th" if i == 0 else "td"
                                table_html += f"    <{tag}>{cell}</{tag}>\n"
                            table_html += "  </tr>\n"
                        table_html += "</table>"
                        tables.append(table_html)

        return tables

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        file_type: str = "PDF",
        file_start_page: Optional[int] = 1,
        file_end_page: Optional[int] = 1000,
        delete_output: Optional[bool] = True,
        max_retries: Optional[int] = 1,
    ) -> tuple:
        """
        解析 PDF 文档（TCADP 主入口方法）。

        完整流程：
        1. 提取 PDF 目录大纲
        2. 将文件转为 Base64 编码
        3. 调用腾讯云文档解析 API（支持指数退避重试）
        4. 下载并解压解析结果 ZIP 包
        5. 转换为 sections 和 tables 格式
        6. 清理临时文件

        Args:
            filepath: PDF 文件路径
            binary: PDF 文件的二进制数据（与 filepath 二选一）
            callback: 进度回调函数，签名为 (progress: float, message: str)
            output_dir: 输出目录，默认为临时目录
            file_type: 文件类型，默认 "PDF"
            file_start_page: 起始页码，默认 1
            file_end_page: 结束页码，默认 1000
            delete_output: 是否在解析完成后删除临时输出目录，默认 True
            max_retries: API 调用最大重试次数，默认 1

        Returns:
            tuple: (sections: list[tuple[str, str]], tables: list[str])
        """

        self.outlines = extract_pdf_outlines(binary if binary else filepath)
        temp_file = None
        created_tmp_dir = False

        try:
            # 处理输入文件：二进制数据写入临时文件
            if binary:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp_file.write(binary)
                temp_file.close()
                file_path = temp_file.name
                self.logger.info(f"[TCADP] Received binary PDF -> {os.path.basename(file_path)}")
                if callback:
                    callback(0.1, f"[TCADP] Received binary PDF -> {os.path.basename(file_path)}")
            else:
                file_path = str(filepath)
                if not os.path.exists(file_path):
                    if callback:
                        callback(-1, f"[TCADP] PDF file does not exist: {file_path}")
                    raise FileNotFoundError(f"[TCADP] PDF file does not exist: {file_path}")

            # 将文件转换为 Base64 格式
            if callback:
                callback(0.2, "[TCADP] Converting file to Base64 format")

            file_base64 = self._file_to_base64(file_path, binary)
            if callback:
                callback(0.25, f"[TCADP] File converted to Base64, size: {len(file_base64)} characters")

            # 创建腾讯云 API 客户端
            client = TencentCloudAPIClient(self.secret_id, self.secret_key, self.region)

            # 调用文档解析 API（带重试机制）
            if callback:
                callback(0.3, "[TCADP] Starting to call Tencent Cloud document parsing API")

            result = None
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        self.logger.info(f"[TCADP] Retry attempt {attempt + 1}")
                        if callback:
                            callback(0.3 + attempt * 0.1, f"[TCADP] Retry attempt {attempt + 1}")
                        time.sleep(2 ** attempt)  # 指数退避：1s, 2s, 4s, ...

                    # 构建 API 请求配置
                    config = {
                        "TableResultType": self.table_result_type,
                        "MarkdownImageResponseType": self.markdown_image_response_type
                    }

                    self.logger.info(f"[TCADP] API request config - TableResultType: {self.table_result_type}, MarkdownImageResponseType: {self.markdown_image_response_type}")

                    result = client.reconstruct_document_sse(
                        file_type=file_type,
                        file_base64=file_base64,
                        file_start_page=file_start_page,
                        file_end_page=file_end_page,
                        config=config
                    )

                    if result:
                        self.logger.info(f"[TCADP] Attempt {attempt + 1} successful")
                        break
                    else:
                        self.logger.warning(f"[TCADP] Attempt {attempt + 1} failed, result is None")

                except Exception as e:
                    self.logger.error(f"[TCADP] Attempt {attempt + 1} exception: {e}")
                    if attempt == max_retries - 1:
                        raise

            if not result:
                error_msg = f"[TCADP] Document parsing failed, retried {max_retries} times"
                self.logger.error(error_msg)
                if callback:
                    callback(-1, error_msg)
                raise RuntimeError(error_msg)

            # 获取下载链接
            download_url = result.get("DocumentRecognizeResultUrl")
            if not download_url:
                if callback:
                    callback(-1, "[TCADP] No parsing result download link obtained")
                raise RuntimeError("[TCADP] No parsing result download link obtained")

            if callback:
                callback(0.6, f"[TCADP] Parsing result download link: {download_url}")

            # 设置输出目录
            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                out_dir = Path(tempfile.mkdtemp(prefix="adp_pdf_"))
                created_tmp_dir = True

            # 下载解析结果文件
            zip_path = client.download_result_file(download_url, str(out_dir))
            if not zip_path:
                if callback:
                    callback(-1, "[TCADP] Failed to download parsing result")
                raise RuntimeError("[TCADP] Failed to download parsing result")

            if callback:
                # 缩短文件路径显示，仅显示文件名
                zip_filename = os.path.basename(zip_path)
                callback(0.8, f"[TCADP] Parsing result downloaded: {zip_filename}")

            # 提取 ZIP 文件中的内容
            content_data = self._extract_content_from_zip(zip_path)
            self.logger.info(f"[TCADP] Extracted {len(content_data)} content blocks")

            if callback:
                callback(0.9, f"[TCADP] Extracted {len(content_data)} content blocks")

            # 转换为 sections 和 tables 格式
            sections = self._parse_content_to_sections(content_data)
            tables = self._parse_content_to_tables(content_data)

            self.logger.info(f"[TCADP] Parsing completed: {len(sections)} sections, {len(tables)} tables")

            if callback:
                callback(1.0, f"[TCADP] Parsing completed: {len(sections)} sections, {len(tables)} tables")

            return sections, tables

        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass

            if delete_output and created_tmp_dir and out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass


if __name__ == "__main__":
    # 测试 ADP 解析器
    parser = TCADPParser()
    print("ADP available:", parser.check_installation())

    # 测试解析功能
    filepath = ""
    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as file:
            sections, tables = parser.parse_pdf(filepath=filepath, binary=file.read())
            print(f"Parsing result: {len(sections)} sections, {len(tables)} tables")
            for i, (section, tag) in enumerate(sections[:3]):  # 仅打印前 3 个
                print(f"Section {i + 1}: {section[:100]}...")
