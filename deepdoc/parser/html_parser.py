# -*- coding: utf-8 -*-
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
HTML 文件解析器

基于 BeautifulSoup 实现 HTML 文档的结构化解析和分块。
核心处理流程：
1. 清理冗余标签（style、script、HTML 注释、内联样式）
2. 递归遍历 DOM 树，按块级标签（h1-h6/p/div/table 等）分组
3. 为每个块级元素分配唯一 block_id，相同 block_id 的内容合并为一组
4. 按 token 数量阈值将文本块合并为最终的分段（chunks）
5. 表格独立处理，超大表格按行数拆分为子表
"""

from rag.nlp import find_codec, rag_tokenizer
import uuid
import chardet
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
import html

def get_encoding(file):
    """检测 HTML 文件的字符编码"""
    with open(file,'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']

# 块级标签：这些标签的内容被视为独立的结构块
BLOCK_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "div", "article", "section", "aside",
    "ul", "ol", "li",
    "table", "pre", "code", "blockquote",
    "figure", "figcaption"
]

# 标题标签到 Markdown 标记的映射
TITLE_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class RAGFlowHtmlParser:
    """HTML 文档解析器

    将 HTML 文档按结构语义拆分为合适大小的文本块。
    保留文档结构（标题层级、段落边界），表格独立处理并分割。
    """

    def __call__(self, fnm, binary=None, chunk_token_num=512):
        """解析 HTML 文件

        Args:
            fnm: 文件路径（用于日志）
            binary: HTML 文件的二进制内容
            chunk_token_num: 每个最终文本块的最大 token 数

        Returns:
            文本块列表，每项为一段文本（可能包含 Markdown 标题标记）
        """
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r",encoding=get_encoding(fnm)) as f:
                txt = f.read()
        return self.parser_txt(txt, chunk_token_num)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num):
        """解析 HTML 文本的完整流程

        流程：
        1. 清理：移除 style/script 标签、内联样式、HTML 注释
        2. 递归读取：遍历 DOM 树，按 BLOCK_TAGS 分组提取文本
        3. 合并：将相同 block_id 的文本片段合并为段落
        4. 分块：按 token 数量阈值将段落合并为最终块
        5. 表格追加：将表格 HTML 作为独立块追加

        Args:
            txt: HTML 文本字符串
            chunk_token_num: 每个块的最大 token 数

        Returns:
            文本块字符串列表
        """
        if not isinstance(txt, str):
            raise TypeError("txt type should be string!")

        temp_sections = []
        soup = BeautifulSoup(txt, "html.parser")

        # 清理步骤1：删除 <style> 和 <script> 标签及其内容
        for style_tag in soup.find_all(["style", "script"]):
            style_tag.decompose()

        # 清理步骤2：删除 <div> 内嵌套的 <script> 标签
        for div_tag in soup.find_all("div"):
            for script_tag in div_tag.find_all("script"):
                script_tag.decompose()

        # 清理步骤3：删除所有内联 style 属性
        for tag in soup.find_all(True):
            if 'style' in tag.attrs:
                del tag.attrs['style']

        # 清理步骤4：删除 HTML 注释
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 从 <body> 开始递归读取文本结构
        cls.read_text_recursively(soup.body, temp_sections, chunk_token_num=chunk_token_num)

        # 合并相同 block_id 的文本片段为段落，分离表格
        block_txt_list, table_list = cls.merge_block_text(temp_sections)

        # 按 token 数量阈值将段落合并为最终块
        sections = cls.chunk_block(block_txt_list, chunk_token_num=chunk_token_num)

        # 将表格 HTML 作为独立块追加
        for table in table_list:
            sections.append(table.get("content", ""))
        return sections

    @classmethod
    def split_table(cls, html_table, chunk_token_num=512):
        """将超大表格按行数拆分为多个子表

        当表格的总 token 数超过 chunk_token_num 时，按行切割确保
        每个子表的 token 数不超过限制。

        Args:
            html_table: 原始表格的 HTML 字符串
            chunk_token_num: 每个子表的最大 token 数

        Returns:
            子表 HTML 字符串列表
        """
        soup = BeautifulSoup(html_table, "html.parser")
        rows = soup.find_all("tr")
        tables = []
        current_table = []
        current_count = 0
        table_str_list = []
        for row in rows:
            tks_str = rag_tokenizer.tokenize(str(row))
            token_count = len(tks_str.split(" ")) if tks_str else 0
            if current_count + token_count > chunk_token_num:
                tables.append(current_table)
                current_table = []
                current_count = 0
            current_table.append(row)
            current_count += token_count
        if current_table:
            tables.append(current_table)

        # 为每个行组构造新的 <table> 元素
        for table_rows in tables:
            new_table = soup.new_tag("table")
            for row in table_rows:
                new_table.append(row)
            table_str_list.append(str(new_table))

        return table_str_list

    @classmethod
    def read_text_recursively(cls, element, parser_result, chunk_token_num=512, parent_name=None, block_id=None):
        """递归遍历 DOM 树，按块级标签分组提取文本

        遍历规则：
        - NavigableString（文本节点）：提取纯文本，过滤掉内嵌 HTML
        - Tag 且为 <table>：生成唯一 table_id，作为表格类型单独记录
        - Tag 且为块级标签（BLOCK_TAGS）：生成新的 block_id，子节点均归属此 block
        - 其他 Tag：继承父节点的 block_id，继续递归

        Args:
            element: 当前 DOM 元素（Tag 或 NavigableString）
            parser_result: 累积的解析结果列表（原地修改）
            chunk_token_num: 表格分块的 token 阈值
            parent_name: 父标签名
            block_id: 当前所属的块 ID

        Returns:
            info 列表（内部使用，用于递归传递）
        """
        if isinstance(element, NavigableString):
            content = element.strip()

            def is_valid_html(content):
                """判断字符串是否包含有效的 HTML 标签"""
                try:
                    soup = BeautifulSoup(content, "html.parser")
                    return bool(soup.find())
                except Exception:
                    return False

            return_info = []
            if content:
                if is_valid_html(content):
                    # 如果文本内容包含 HTML 标签，递归解析
                    soup = BeautifulSoup(content, "html.parser")
                    child_info = cls.read_text_recursively(soup, parser_result, chunk_token_num, element.name, block_id)
                    parser_result.extend(child_info)
                else:
                    # 纯文本：记录为 inner_text 类型
                    info = {"content": element.strip(), "tag_name": "inner_text", "metadata": {"block_id": block_id}}
                    if parent_name:
                        info["tag_name"] = parent_name
                    return_info.append(info)
            return return_info
        elif isinstance(element, Tag):
            # 表格：作为独立类型处理，不参与普通文本合并
            if str.lower(element.name) == "table":
                table_info_list = []
                table_id = str(uuid.uuid1())
                table_list = [html.unescape(str(element))]
                for t in table_list:
                    table_info_list.append({"content": t, "tag_name": "table",
                                            "metadata": {"table_id": table_id, "index": table_list.index(t)}})
                return table_info_list
            else:
                # 块级标签：创建新的 block_id
                if str.lower(element.name) in BLOCK_TAGS:
                    block_id = str(uuid.uuid1())
                # 递归处理子节点
                for child in element.children:
                    child_info = cls.read_text_recursively(child, parser_result, chunk_token_num, element.name,
                                                           block_id)
                    parser_result.extend(child_info)
        return []

    @classmethod
    def merge_block_text(cls, parser_result):
        """将解析结果中相同 block_id 的文本片段合并为段落

        合并规则：
        - 相同 block_id 的片段用空格连接
        - 不同 block_id 的片段作为独立段落
        - 标题类型（h1-h6）自动添加 Markdown 标题前缀
        - 表格类型（table）单独收集，不参与文本合并

        Args:
            parser_result: read_text_recursively 的输出列表

        Returns:
            (block_content_list, table_info_list) 元组
        """
        block_content = []
        current_content = ""
        table_info_list = []
        last_block_id = None
        for item in parser_result:
            content = item.get("content")
            tag_name = item.get("tag_name")
            title_flag = tag_name in TITLE_TAGS
            block_id = item.get("metadata", {}).get("block_id")
            if block_id:
                # 标题添加 Markdown 前缀
                if title_flag:
                    content = f"{TITLE_TAGS[tag_name]} {content}"
                if last_block_id != block_id:
                    # 新的 block：保存上一个 block，开始新的
                    if last_block_id is not None:
                        block_content.append(current_content)
                    current_content = content
                    last_block_id = block_id
                else:
                    # 同一个 block 内：用空格连接
                    current_content += (" " if current_content else "") + content
            else:
                # 没有 block_id：表格独立收集，其他文本追加到当前块
                if tag_name == "table":
                    table_info_list.append(item)
                else:
                    current_content += (" " if current_content else "") + content
        if current_content:
            block_content.append(current_content)
        return block_content, table_info_list

    @classmethod
    def chunk_block(cls, block_txt_list, chunk_token_num=512):
        """将段落列表按 token 数量合并为最终文本块

        使用贪心算法：
        1. 如果单个段落的 token 数已超过 chunk_token_num，
           则按 token 数等分切割
        2. 否则将相邻段落不断合并，直到超过阈值
        3. 超过阈值时创建新块

        Args:
            block_txt_list: 段落文本列表
            chunk_token_num: 每个块的最大 token 数

        Returns:
            文本块列表
        """
        chunks = []
        current_block = ""
        current_token_count = 0

        for block in block_txt_list:
            tks_str = rag_tokenizer.tokenize(block)
            block_token_count = len(tks_str.split(" ")) if tks_str else 0
            if block_token_count > chunk_token_num:
                # 单个段落已超过阈值：先保存当前块，再按固定大小切割
                if current_block:
                    chunks.append(current_block)
                start = 0
                tokens = tks_str.split(" ")
                while start < len(tokens):
                    end = start + chunk_token_num
                    split_tokens = tokens[start:end]
                    chunks.append(" ".join(split_tokens))
                    start = end
                current_block = ""
                current_token_count = 0
            else:
                if current_token_count + block_token_count <= chunk_token_num:
                    # 还能追加到当前块
                    current_block += ("\n" if current_block else "") + block
                    current_token_count += block_token_count
                else:
                    # 新段落会使当前块超限：保存当前块并创建新块
                    chunks.append(current_block)
                    current_block = block
                    current_token_count = block_token_count

        if current_block:
            chunks.append(current_block)

        return chunks
