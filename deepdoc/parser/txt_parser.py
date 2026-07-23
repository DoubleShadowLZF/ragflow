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
纯文本文件解析器

将 .txt 文件按指定的分隔符（默认为换行和中英文标点：\\n!?;。；！？）进行分割，
并按 token 数量（chunk_token_num）自动合并为合适大小的文本块。

核心逻辑：
1. 读取文件内容并检测编码
2. 将分隔符解析为正则表达式，支持反引号包裹的多字符分隔符
3. 按分隔符切割文本，使用贪心算法将相邻段落合并直到达到 token 限制
4. 返回 [[text, ""], ...] 格式的二元组列表
"""

import re

from deepdoc.parser.utils import get_text
from common.token_utils import num_tokens_from_string


class RAGFlowTxtParser:
    """纯文本文件解析器

    将 txt 文件按分隔符切割，并以 token 数量为阈值将短段落合并为合适大小的块。
    支持通过反引号语法 `` `delim` `` 在分隔符字符串中定义多字符分隔符。
    """

    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        """解析文本文件

        Args:
            fnm: 文件路径
            binary: 文件的二进制内容（优先使用，避免重复读取）
            chunk_token_num: 每个文本块的最大 token 数，默认 128
            delimiter: 分段分隔符字符串，支持反引号包裹多字符分隔符
                       默认：换行、英文标点(!?;) 和中文标点(。；！？)

        Returns:
            [[text, ""], ...] 格式的二元组列表，每个元素为 [段落文本, 空字符串]
        """
        txt = get_text(fnm, binary)
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        """按分隔符切割文本并合并为 token 大小合适的块

        处理流程：
        1. 解析分隔符字符串，反引号内容作为整体分隔符，其余字符逐字符分割
        2. 按正则表达式切割文本
        3. 使用贪心算法：当前块的 token 数 < chunk_token_num 时继续追加；
           超过阈值时新建块

        Args:
            txt: 待解析的文本字符串
            chunk_token_num: 每个块的最大 token 数
            delimiter: 分隔符字符串

        Returns:
            [[text, ""], ...] 格式的文本块列表
        """
        if not isinstance(txt, str):
            raise TypeError("txt type should be str!")
        cks = [""]           # 文本块列表
        tk_nums = [0]        # 对应每个块的 token 数

        # 解析分隔符：将 Python 转义字符串（如 \\n）解码为实际字符
        delimiter = delimiter.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8')

        def add_chunk(t):
            """将一个新的文本片段追加到当前块或创建新块"""
            nonlocal cks, tk_nums, delimiter
            tnum = num_tokens_from_string(t)
            if tk_nums[-1] > chunk_token_num:
                # 当前块已满，新建一个块
                cks.append(t)
                tk_nums.append(tnum)
            else:
                # 当前块还有空间，用换行连接
                if cks[-1]:
                    cks[-1] += "\n" + t
                else:
                    cks[-1] += t
                tk_nums[-1] += tnum

        # 解析分隔符字符串：
        # - 反引号内的内容 `` `...` `` 作为整体分隔符
        # - 反引号外的字符逐个作为单字符分隔符
        dels = []
        s = 0
        for m in re.finditer(r"`([^`]+)`", delimiter, re.I):
            f, t = m.span()
            dels.append(m.group(1))           # 反引号内的多字符分隔符
            dels.extend(list(delimiter[s: f]))  # 反引号前的单字符分隔符
            s = t
        if s < len(delimiter):
            dels.extend(list(delimiter[s:]))   # 最后剩余的单字符分隔符

        # 转义特殊正则字符（如 . * + 等），并构建正则表达式
        dels = [re.escape(d) for d in dels if d]
        dels = [d for d in dels if d]
        dels = "|".join(dels)

        # 按分隔符切割文本（保留分隔符本身以便精确还原）
        secs = re.split(r"(%s)" % dels, txt)
        for sec in secs:
            # 跳过纯分隔符片段
            if re.match(f"^{dels}$", sec):
                continue
            add_chunk(sec)

        return [[c, ""] for c in cks]
