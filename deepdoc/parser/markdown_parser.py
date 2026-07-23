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
Markdown 文件解析器

提供两个核心类：
1. RAGFlowMarkdownParser：提取和分离 Markdown 文档中的表格
   - 支持标准表格（|...|格式）、无边框表格、HTML <table> 标签
   - 支持带 <html><body> 包装的表格结构
   - 可配置表格是否从正文中分离

2. MarkdownElementExtractor：按 Markdown 语法元素拆分文档
   - 识别标题(h1-h6)、代码块(```fenced```)、列表、引用块、普通段落
   - 支持按自定义分隔符切割（_extract_delimited_elements）
   - 保护区域机制：fenced code 块内的内容不参与分隔符切割
   - 返回含元数据（起止行号、元素类型）的结构化结果
"""

import logging
import re

from markdown import markdown


class RAGFlowMarkdownParser:
    """Markdown 文档表格提取器

    从 Markdown 文档中识别并提取表格（Markdown 原生表格、HTML 表格），
    可选择将表格从正文中分离出单独处理。
    """

    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = int(chunk_token_num)

    def extract_tables_and_remainder(self, markdown_text, separate_tables=True):
        """从 Markdown 文本中提取表格和剩余内容

        识别三种表格格式：
        1. 标准 Markdown 表格：有竖线边框 + 分隔行（如 |---|---|）
        2. 无边框 Markdown 表格：有竖线分隔但无边线
        3. HTML <table> 标签：支持 <html><body><table> 嵌套

        Args:
            markdown_text: Markdown 原始文本
            separate_tables: True=表格从正文中移除单独返回，False=表格渲染为 HTML 保留在正文

        Returns:
            (working_text, tables) 元组：
            - working_text: 移除表格后的正文（或含渲染 HTML 的正文）
            - tables: 提取的表格原始文本列表
        """
        tables = []
        working_text = markdown_text

        def replace_tables_with_rendered_html(pattern, table_list, render=True):
            """使用正则替换表格：收集到 table_list，选择移除或渲染"""
            new_text = ""
            last_end = 0
            for match in pattern.finditer(working_text):
                raw_table = match.group()
                table_list.append(raw_table)
                if separate_tables:
                    # 从正文中移除表格，保留段落间距
                    new_text += working_text[last_end : match.start()] + "\n\n"
                else:
                    # 将 Markdown 表格渲染为 HTML 保留在正文中
                    html_table = markdown(raw_table, extensions=["markdown.extensions.tables"]) if render else raw_table
                    new_text += working_text[last_end : match.start()] + html_table + "\n\n"
                last_end = match.end()
            new_text += working_text[last_end:]
            return new_text

        # 性能优化：仅当包含竖线时才尝试匹配 Markdown 表格
        if "|" in markdown_text:
            # 标准 Markdown 表格（有边框线）
            border_table_pattern = re.compile(
                r"""
                (?:\n|^)
                (?:\|.*?\|.*?\|.*?\n)
                (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)
                (?:\|.*?\|.*?\|.*?\n)+
            """,
                re.VERBOSE,
            )
            working_text = replace_tables_with_rendered_html(border_table_pattern, tables, render=separate_tables)

            # 无边框 Markdown 表格（仅有竖线分隔，无外围边框线）
            no_border_table_pattern = re.compile(
                r"""
                (?:\n|^)
                (?:\S.*?\|.*?\n)
                (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
                (?:\S.*?\|.*?\n)+
                """,
                re.VERBOSE,
            )
            working_text = replace_tables_with_rendered_html(no_border_table_pattern, tables, render=separate_tables)

        # 将带属性的标签（如 <table class="...">）简化为纯标签名（<table>）
        TAGS = ["table", "td", "tr", "th", "tbody", "thead", "div"]
        table_with_attributes_pattern = re.compile(rf"<(?:{'|'.join(TAGS)})[^>]*>", re.IGNORECASE)

        def replace_tag(m):
            tag_name = re.match(r"<(\w+)", m.group()).group(1)
            return "<{}>".format(tag_name)

        working_text = re.sub(table_with_attributes_pattern, replace_tag, working_text)

        # 性能优化：仅当包含 <table> 时才匹配 HTML 表格
        if "<table>" in working_text.lower():
            # HTML 表格提取：支持 <html><body><table> / <body><table> / <table> 三种嵌套
            html_table_pattern = re.compile(
                r"""
            (?:\n|^)
            \s*
            (?:
                # case1: <html><body><table>...</table></body></html>
                (?:<html[^>]*>\s*<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>\s*</html>)
                |
                # case2: <body><table>...</table></body>
                (?:<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>)
                |
                # case3: only <table>...</table>
                (?:<table[^>]*>.*?</table>)
            )
            \s*
            (?=\n|$)
            """,
                re.VERBOSE | re.DOTALL | re.IGNORECASE,
            )

            def replace_html_tables():
                nonlocal working_text
                new_text = ""
                last_end = 0
                for match in html_table_pattern.finditer(working_text):
                    raw_table = match.group()
                    tables.append(raw_table)
                    if separate_tables:
                        new_text += working_text[last_end : match.start()] + "\n\n"
                    else:
                        new_text += working_text[last_end : match.start()] + raw_table + "\n\n"
                    last_end = match.end()
                new_text += working_text[last_end:]
                working_text = new_text

            replace_html_tables()

        return working_text, tables


class MarkdownElementExtractor:
    """Markdown 文档结构化元素提取器

    按 Markdown 语法结构将文档拆分为独立元素（标题、代码块、列表、引用、段落），
    支持保护区域（代码块、表格）内的内容不参与分隔符切割。
    """

    def __init__(self, markdown_content):
        self.markdown_content = markdown_content
        self.lines = markdown_content.split("\n")

    def get_delimiters(self, delimiters):
        """解析分隔符字符串中的反引号包裹的多字符分隔符

        例如 "`### `" 会将 "### " 作为整体分隔符而非按字符分割。
        """
        toks = re.findall(r"`([^`]+)`", delimiters)
        toks = sorted(set(toks), key=lambda x: -len(x))
        return "|".join(re.escape(t) for t in toks if t)

    def _get_fence_marker(self, line):
        """检测代码块围栏标记（``` 或 ~~~）

        Returns:
            (围栏字符, 围栏长度)，非围栏行返回 None
        """
        match = re.match(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})(?:.*)$", line)
        if not match:
            return None
        fence = match.group("fence")
        return fence[0], len(fence)

    def _is_closing_fence(self, line, fence_char, fence_len):
        """判断是否为代码块的闭合围栏行"""
        pattern = r"^[ \t]{0,3}" + re.escape(fence_char) + r"{" + str(fence_len) + r",}\s*$"
        return re.match(pattern, line) is not None

    def _line_start_offsets(self, text):
        """计算每行在原文中的起始字符偏移量"""
        offsets = []
        offset = 0
        for line in self.lines:
            offsets.append(offset)
            offset += len(line) + 1  # +1 for \n
        return offsets

    def _fenced_code_ranges(self, text):
        """获取所有 fenced code block 在原文中的 (起始位置, 结束位置) 范围"""
        ranges = []
        line_offsets = self._line_start_offsets(text)
        i = 0
        while i < len(self.lines):
            marker = self._get_fence_marker(self.lines[i])
            if not marker:
                i += 1
                continue
            fence_char, fence_len = marker
            start_pos = line_offsets[i]
            end_line = len(self.lines) - 1
            for j in range(i + 1, len(self.lines)):
                if self._is_closing_fence(self.lines[j], fence_char, fence_len):
                    end_line = j
                    break
            end_pos = min(len(text), line_offsets[end_line] + len(self.lines[end_line]))
            ranges.append((start_pos, end_pos))
            i = end_line + 1
        return ranges

    def _table_cells(self, line):
        """提取 Markdown 表格行中的单元格列表"""
        stripped = line.strip()
        if "|" not in stripped:
            return []
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def _is_table_row(self, line):
        """判断是否为 Markdown 表格的数据行"""
        cells = self._table_cells(line)
        return len(cells) >= 2 and any(cell for cell in cells)

    def _is_table_separator_row(self, line):
        """判断是否为 Markdown 表格的分隔行（如 |---|---|）"""
        cells = self._table_cells(line)
        return len(cells) >= 2 and all(re.match(r"^:?-{3,}:?$", cell.replace(" ", "")) for cell in cells)

    def _markdown_table_ranges(self, text):
        """获取所有 Markdown 表格在原文中的位置范围"""
        ranges = []
        line_offsets = self._line_start_offsets(text)
        i = 0
        while i < len(self.lines) - 1:
            if not self._is_table_row(self.lines[i]) or not self._is_table_separator_row(self.lines[i + 1]):
                i += 1
                continue
            end_line = i + 1
            j = i + 2
            while j < len(self.lines) and self._is_table_row(self.lines[j]):
                end_line = j
                j += 1
            end_pos = min(len(text), line_offsets[end_line] + len(self.lines[end_line]))
            ranges.append((line_offsets[i], end_pos))
            i = end_line + 1
        return ranges

    def _html_table_ranges(self, text):
        """获取所有 HTML 表格在原文中的位置范围"""
        table_pattern = re.compile(
            r"""
            (?:
                (?:<html[^>]*>\s*<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>\s*</html>)
                |
                (?:<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>)
                |
                (?:<table[^>]*>.*?</table>)
            )
            """,
            re.VERBOSE | re.DOTALL | re.IGNORECASE,
        )
        return [(match.start(), match.end()) for match in table_pattern.finditer(text)]

    def _merge_ranges(self, ranges):
        """合并重叠或相邻的范围区间"""
        if not ranges:
            return []
        merged = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _protected_ranges(self, text):
        """获取需要保护的范围（代码块 + Markdown表格 + HTML表格）

        这些范围内的内容不参与分隔符切割，防止破坏代码块或表格结构。
        """
        return self._merge_ranges(
            self._fenced_code_ranges(text)
            + self._markdown_table_ranges(text)
            + self._html_table_ranges(text)
        )

    def _append_delimited_section(self, sections, text, start, end, include_meta):
        """将文本切片追加到 sections 列表（过滤空内容）"""
        part = text[start:end]
        if not part or not part.strip():
            return
        if include_meta:
            sections.append(
                {
                    "content": part.strip(),
                    "start_line": text.count("\n", 0, start),
                    "end_line": text.count("\n", 0, end),
                }
            )
        else:
            sections.append(part.strip())

    def _extract_delimited_elements(self, text, delimiters, include_meta=False):
        """按自定义分隔符切割文本，保护区域内的分隔符不被切割

        这是配合 RAGFlow 自定义分隔符机制的通用切割方法。
        分隔符参数由上层（如 parser_txt）传入。

        Args:
            text: 待切割的文本
            delimiters: 分隔符正则表达式字符串
            include_meta: 是否包含元数据（起止行号）

        Returns:
            切割后的文本段列表
        """
        sections = []
        pattern = re.compile(delimiters)
        protected_ranges = self._protected_ranges(text)
        if protected_ranges:
            logging.debug("markdown_parser: detected %d protected ranges for delimiter extraction", len(protected_ranges))
        protected_idx = 0
        last_end = 0

        for match in pattern.finditer(text):
            # 跳过已处理的保护区域
            while protected_idx < len(protected_ranges) and protected_ranges[protected_idx][1] <= match.start():
                protected_idx += 1

            # 如果匹配位置在保护区域内，跳过此次分隔
            if protected_idx < len(protected_ranges):
                start, end = protected_ranges[protected_idx]
                if start <= match.start() < end:
                    logging.debug(
                        "markdown_parser: skipped delimiter match at pos=%d delimiter=%r inside fenced range %s",
                        match.start(),
                        match.group(),
                        (start, end),
                    )
                    continue

            self._append_delimited_section(sections, text, last_end, match.start(), include_meta)
            last_end = match.end()

        # 追加最后一个分隔符之后的内容
        self._append_delimited_section(sections, text, last_end, len(text), include_meta)
        return sections

    def extract_elements(self, delimiter=None, include_meta=False):
        """按 Markdown 语法元素拆分文档

        识别顺序：标题 > 代码块 > 列表 > 引用 > 文本段落
        如果指定了 delimiter，则使用自定义分隔符切割（保护区域机制生效）。

        Args:
            delimiter: 自定义分隔符字符串（如 "`### `"）
            include_meta: 是否包含元素元数据（类型、起止行号）

        Returns:
            元素列表（纯文本字符串列表或含元数据的字典列表）
        """
        sections = []
        i = 0
        dels = ""
        if delimiter:
            dels = self.get_delimiters(delimiter)
        if len(dels) > 0:
            text = "\n".join(self.lines)
            return self._extract_delimited_elements(text, dels, include_meta)
        while i < len(self.lines):
            line = self.lines[i]

            if re.match(r"^#{1,6}\s+.*$", line):
                # 标题 (h1-h6)
                element = self._extract_header(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif self._get_fence_marker(line):
                # 代码块（fenced code block）
                element = self._extract_code_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line):
                # 列表（无序列表或有序列表）
                element = self._extract_list_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith(">"):
                # 引用块（blockquote）
                element = self._extract_blockquote(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip():
                # 文本段落（普通段落和内联元素）
                element = self._extract_text_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            else:
                i += 1

        if include_meta:
            sections = [section for section in sections if section["content"].strip()]
        else:
            sections = [section for section in sections if section.strip()]
        return sections

    def _extract_header(self, start_pos):
        """提取标题元素（单行）"""
        return {
            "type": "header",
            "content": self.lines[start_pos],
            "start_line": start_pos,
            "end_line": start_pos,
        }

    def _extract_code_block(self, start_pos):
        """提取代码块（从开始围栏到闭合围栏的所有行）"""
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]
        fence_char, fence_len = self._get_fence_marker(self.lines[start_pos])

        for i in range(start_pos + 1, len(self.lines)):
            content_lines.append(self.lines[i])
            end_pos = i
            if self._is_closing_fence(self.lines[i], fence_char, fence_len):
                break

        return {
            "type": "code_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_list_block(self, start_pos):
        """提取列表块（包含所有连续的列表行和子列表）"""
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            # 判断是否属于当前列表块：列表项、空行、缩进延续行
            if (
                re.match(r"^\s*[-*+]\s+.*$", line)
                or re.match(r"^\s*\d+\.\s+.*$", line)
                or (i > start_pos and not line.strip())
                or (i > start_pos and re.match(r"^\s{2,}[-*+]\s+.*$", line))
                or (i > start_pos and re.match(r"^\s{2,}\d+\.\s+.*$", line))
                or (i > start_pos and re.match(r"^\s+\w+.*$", line))
            ):
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "list_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_blockquote(self, start_pos):
        """提取引用块（所有连续的以 > 开头的行）"""
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            if line.strip().startswith(">") or (i > start_pos and not line.strip()):
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "blockquote",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_text_block(self, start_pos):
        """提取文本段落（直到遇到下一个块级元素为止）

        停止条件：
        - 标题（#）、代码块围栏（```/~~~）、列表项（-/+/1.）、引用（>）
        - 检测空行后是否跟随块级元素来决定是否继续
        """
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]

        i = start_pos + 1
        while i < len(self.lines):
            line = self.lines[i]
            # 遇到块级元素则停止
            if re.match(r"^#{1,6}\s+.*$", line) or self._get_fence_marker(line) or re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line) or line.strip().startswith(">"):
                break
            elif not line.strip():
                # 空行：检查下一行是否为块级元素，是则停止
                if i + 1 < len(self.lines) and (
                    re.match(r"^#{1,6}\s+.*$", self.lines[i + 1])
                    or self._get_fence_marker(self.lines[i + 1])
                    or re.match(r"^\s*[-*+]\s+.*$", self.lines[i + 1])
                    or re.match(r"^\s*\d+\.\s+.*$", self.lines[i + 1])
                    or self.lines[i + 1].strip().startswith(">")
                ):
                    break
                else:
                    content_lines.append(line)
                    end_pos = i
                    i += 1
            else:
                content_lines.append(line)
                end_pos = i
                i += 1

        return {
            "type": "text_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }
