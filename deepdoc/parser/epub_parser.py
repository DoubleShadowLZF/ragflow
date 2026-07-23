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
EPUB 电子书解析器

EPUB 本质上是一个包含 XHTML 内容文件的 ZIP 压缩包。
本解析器读取 EPUB 的 spine（阅读顺序）信息，按指定顺序提取各章节的 XHTML 内容，
然后委托给 RAGFlowHtmlParser 进行 HTML 解析和分块。

核心技术点：
1. 解析 META-INF/container.xml 找到 OPF 文件路径
2. 解析 OPF 文件获取 manifest（文件清单）和 spine（阅读顺序）
3. 按 spine 顺序逐个提取 XHTML 内容并用 HTML 解析器处理
4. 如果无法解析 OPF 结构，降级为按文件名字母序提取所有 XHTML/HTML 文件
"""

import logging
import warnings
import zipfile
from io import BytesIO
from xml.etree import ElementTree

from .html_parser import RAGFlowHtmlParser

# OPF XML 命名空间常量
_OPF_NS = "http://www.idpf.org/2007/opf"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

# 可读取的 XHTML 内容类型
_XHTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "text/xml"}

logger = logging.getLogger(__name__)


class RAGFlowEpubParser:
    """EPUB 电子书解析器

    通过解析 EPUB 的 OPF 结构获取正确的阅读顺序，
    然后将各章节的 XHTML 内容委托给 RAGFlowHtmlParser 进行解析。
    """

    def __call__(self, fnm, binary=None, chunk_token_num=512):
        """解析 EPUB 文件

        Args:
            fnm: EPUB 文件路径
            binary: 文件的二进制内容（优先使用）
            chunk_token_num: 每个分块的最大 token 数

        Returns:
            文本段列表，每项为 HTML 解析器输出的一个段落

        Raises:
            ValueError: 二进制内容为空时抛出
        """
        if binary is not None:
            if not binary:
                logger.warning(
                    "RAGFlowEpubParser received an empty EPUB binary payload for %r",
                    fnm,
                )
                raise ValueError("Empty EPUB binary payload")
            zf = zipfile.ZipFile(BytesIO(binary))
        else:
            zf = zipfile.ZipFile(fnm)

        try:
            # 获取按阅读顺序排列的内容文件路径
            content_items = self._get_spine_items(zf)
            all_sections = []
            html_parser = RAGFlowHtmlParser()

            for item_path in content_items:
                try:
                    html_bytes = zf.read(item_path)
                except KeyError:
                    continue
                if not html_bytes:
                    logger.debug("Skipping empty EPUB content item: %s", item_path)
                    continue
                # 抑制不必要的警告，如 BeautifulSoup 的 XML 解析警告
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sections = html_parser(
                        item_path, binary=html_bytes, chunk_token_num=chunk_token_num
                    )
                all_sections.extend(sections)

            return all_sections
        finally:
            zf.close()

    @staticmethod
    def _get_spine_items(zf):
        """获取按阅读顺序（spine）排列的内容文件路径

        解析流程：
        1. 从 META-INF/container.xml 找到 OPF 文件的路径
        2. 解析 OPF 文件中的 <manifest> 获取文件清单（id -> href + media_type）
        3. 按 <spine> 中的 <itemref idref="..."> 顺序获取阅读顺序
        4. 过滤掉非 XHTML/HTML 类型的文件
        5. 如果任何步骤失败或结果为空，降级为字母序排列的 XHTML 文件列表

        Args:
            zf: 已打开的 ZipFile 对象

        Returns:
            按阅读顺序排列的文件路径列表
        """
        # 1. 从 container.xml 找到 OPF 文件路径
        try:
            container_xml = zf.read("META-INF/container.xml")
        except KeyError:
            return RAGFlowEpubParser._fallback_xhtml_order(zf)

        try:
            container_root = ElementTree.fromstring(container_xml)
        except ElementTree.ParseError:
            logger.warning("Failed to parse META-INF/container.xml; falling back to XHTML order.")
            return RAGFlowEpubParser._fallback_xhtml_order(zf)

        rootfile_el = container_root.find(f".//{{{_CONTAINER_NS}}}rootfile")
        if rootfile_el is None:
            return RAGFlowEpubParser._fallback_xhtml_order(zf)

        opf_path = rootfile_el.get("full-path", "")
        if not opf_path:
            return RAGFlowEpubParser._fallback_xhtml_order(zf)

        # 内容文件路径是相对于 OPF 文件所在目录的
        opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""

        # 2. 解析 OPF 文件
        try:
            opf_xml = zf.read(opf_path)
        except KeyError:
            return RAGFlowEpubParser._fallback_xhtml_order(zf)

        try:
            opf_root = ElementTree.fromstring(opf_xml)
        except ElementTree.ParseError:
            logger.warning("Failed to parse OPF file '%s'; falling back to XHTML order.", opf_path)
            return RAGFlowEpubParser._fallback_xhtml_order(zf)

        # 3. 构建 id -> (href, media_type) 的 manifest 映射
        manifest = {}
        for item in opf_root.findall(f".//{{{_OPF_NS}}}item"):
            item_id = item.get("id", "")
            href = item.get("href", "")
            media_type = item.get("media-type", "")
            if item_id and href:
                manifest[item_id] = (href, media_type)

        # 4. 按 spine 顺序提取 XHTML 文件路径
        spine_items = []
        for itemref in opf_root.findall(f".//{{{_OPF_NS}}}itemref"):
            idref = itemref.get("idref", "")
            if idref not in manifest:
                continue
            href, media_type = manifest[idref]
            # 只保留 XHTML/HTML 类型的内容
            if media_type not in _XHTML_MEDIA_TYPES:
                continue
            spine_items.append(opf_dir + href)

        # 如果 spine 解析为空，降级处理
        return (
            spine_items if spine_items else RAGFlowEpubParser._fallback_xhtml_order(zf)
        )

    @staticmethod
    def _fallback_xhtml_order(zf):
        """降级方案：返回 ZIP 中所有 XHTML/HTML 文件，按文件名字母序排列

        排除 META-INF/ 目录下的文件（元数据，非内容文件）。
        """
        return sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm"))
            and not n.startswith("META-INF/")
        )
