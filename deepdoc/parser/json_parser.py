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
JSON / JSONL 文件解析器

参考 LangChain 的 JSON 分块算法实现。支持两种 JSON 格式：
1. 标准 JSON：单个 JSON 对象，按 max_chunk_size 递归分块，保持结构完整性
2. JSONL（JSON Lines）：每行一个 JSON 对象，逐行处理

核心算法：
- 递归遍历 JSON 树结构，在每个节点判断当前块大小是否超过阈值
- 如果子节点可以放入当前块则合并，否则创建新块
- 通过 set_nested_dict 按路径设置值，保持原始 JSON 层级结构
"""

import json
from typing import Any

from rag.nlp import find_codec


class RAGFlowJsonParser:
    """JSON / JSONL 文档解析器

    将 JSON 或 JSONL 文件按结构感知的方式分割成不超过 max_chunk_size 的多个块。
    支持将 JSON 数组自动转为字典（以索引为 key）后再分块。
    """

    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int | None = None):
        """初始化 JSON 解析器

        Args:
            max_chunk_size: 每个 JSON 块的最大字节数（内部会乘 2 作为实际限制）
            min_chunk_size: 每个块的最小字节数，低于此值则与下一个元素合并
                            未指定时为 max(max_chunk_size - 200, 50)
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size * 2
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else max(max_chunk_size - 200, 50)

    def __call__(self, binary):
        """解析 JSON 文件的入口方法

        Args:
            binary: 文件的二进制内容

        Returns:
            JSON 字符串列表，每项为一个独立的分块
        """
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")

        if self.is_jsonl_format(txt):
            sections = self._parse_jsonl(txt)
        else:
            sections = self._parse_json(txt)
        return sections

    @staticmethod
    def _json_size(data: dict) -> int:
        """计算 JSON 对象序列化后的字节大小"""
        return len(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def _set_nested_dict(d: dict, path: list[str], value: Any) -> None:
        """按路径在嵌套字典中设置值

        Args:
            d: 目标字典
            path: 键路径列表，如 ["a", "b", "c"] 表示 d["a"]["b"]["c"]
            value: 要设置的值
        """
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        """将 JSON 数组递归转换为以字符串索引为 key 的字典

        这是分块前的预处理步骤，解决数组元素无法按路径访问的问题。
        例如 [{"a": 1}, {"a": 2}] 会被转为 {"0": {"a": 1}, "1": {"a": 2}}
        """
        if isinstance(data, dict):
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        elif isinstance(data, list):
            return {str(i): self._list_to_dict_preprocessing(item) for i, item in enumerate(data)}
        else:
            return data

    def _json_split(
        self,
        data,
        current_path: list[str] | None,
        chunks: list[dict] | None,
    ) -> list[dict]:
        """递归分割 JSON 数据，保持结构完整性的同时控制块大小

        算法逻辑：
        1. 对于字典，遍历每个 key-value：
           - 如果当前块还能容纳，直接添加
           - 如果当前块已满，检查是否达到 min_chunk_size：
             * 已达到 → 创建新块
             * 未达到 → 继续递归分割子节点
        2. 对于非字典值，直接按路径设置到当前块中

        Args:
            data: 待分割的 JSON 数据
            current_path: 当前在 JSON 树中的路径
            chunks: 已有的分块列表

        Returns:
            字典列表，每个字典为独立的一个分块
        """
        current_path = current_path or []
        chunks = chunks or [{}]
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                    # 当前块还放得下，直接添加
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                        # 当前块已经足够大，创建新块
                        chunks.append({})
                    # 继续递归分割子节点
                    self._json_split(value, new_path, chunks)
        else:
            # 非字典值（字符串/数字/数组等），直接按路径设置
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(
        self,
        json_data,
        convert_lists: bool = False,
    ) -> list[dict]:
        """将 JSON 数据分割为多个字典块

        Args:
            json_data: 原始 JSON 数据（dict 或其他类型）
            convert_lists: 是否先将数组转为字典再分割

        Returns:
            字典列表，每项为一个分块
        """
        if convert_lists:
            preprocessed_data = self._list_to_dict_preprocessing(json_data)
            chunks = self._json_split(preprocessed_data, None, None)
        else:
            chunks = self._json_split(json_data, None, None)

        # 移除末尾的空块
        if not chunks[-1]:
            chunks.pop()
        return chunks

    def split_text(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,
        ensure_ascii: bool = True,
    ) -> list[str]:
        """将 JSON 数据分割为 JSON 字符串列表（对外接口）

        Args:
            json_data: 原始 JSON 数据
            convert_lists: 是否先转换数组
            ensure_ascii: 是否转义非 ASCII 字符

        Returns:
            JSON 字符串列表
        """
        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]

    def _parse_json(self, content: str) -> list[str]:
        """解析标准 JSON 格式内容"""
        sections = []
        try:
            json_data = json.loads(content)
            chunks = self.split_json(json_data, True)
            sections = [json.dumps(line, ensure_ascii=False) for line in chunks if line]
        except json.JSONDecodeError:
            pass
        return sections

    def _parse_jsonl(self, content: str) -> list[str]:
        """解析 JSONL（JSON Lines）格式内容

        逐行解析，每行视为一个独立的 JSON 对象，分别分块后合并。
        """
        lines = content.strip().splitlines()
        all_chunks = []
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                chunks = self.split_json(data, convert_lists=True)
                all_chunks.extend(json.dumps(chunk, ensure_ascii=False) for chunk in chunks if chunk)
            except json.JSONDecodeError:
                continue
        return all_chunks

    def is_jsonl_format(self, txt: str, sample_limit: int = 10, threshold: float = 0.8) -> bool:
        """判断文本是否为 JSONL 格式

        检测策略：
        1. 如果整体文本可以被 json.loads 解析，则视为标准 JSON（非 JSONL）
        2. 否则采样前 sample_limit 行，若超过 threshold 比例的行能解析为 JSON，
           则认为该文件是 JSONL 格式

        Args:
            txt: 文本内容
            sample_limit: 采样行数上限
            threshold: 判定阈值（有效行比例），默认 0.8

        Returns:
            True 表示 JSONL 格式，False 表示标准 JSON 或无法判断
        """
        lines = [line.strip() for line in txt.strip().splitlines() if line.strip()]
        if not lines:
            return False

        # 如果能整体解析，说明是标准 JSON 而非 JSONL
        try:
            json.loads(txt)
            return False
        except json.JSONDecodeError:
            pass

        sample_limit = min(len(lines), sample_limit)
        sample_lines = lines[:sample_limit]
        valid_lines = sum(1 for line in sample_lines if self._is_valid_json(line))

        if not valid_lines:
            return False

        return (valid_lines / len(sample_lines)) >= threshold

    def _is_valid_json(self, line: str) -> bool:
        """判断单行文本是否为有效的 JSON"""
        try:
            json.loads(line)
            return True
        except json.JSONDecodeError:
            return False
