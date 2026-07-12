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
文档存储抽象基类和搜索表达式定义。

定义了所有文档存储引擎（Elasticsearch、Infinity、OceanBase 等）必须实现的
统一接口，以及搜索时使用的各种匹配表达式类型。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# 向量搜索默认返回数量
DEFAULT_MATCH_VECTOR_TOPN = 10
# 稀疏向量搜索默认返回数量
DEFAULT_MATCH_SPARSE_TOPN = 10
# 向量数据类型（list 或 numpy 数组）
VEC = list | np.ndarray

@dataclass
class SparseVector:
    """稀疏向量数据结构。

    由索引列表和对应的值列表组成，用于稀疏向量检索场景（如 BM25 风格的稀疏嵌入）。
    支持两种序列化格式：to_dict()（{index: value} 映射）和 to_dict_old()（分离的 indices/values 列表）。
    """
    indices: list[int]
    values: list[float] | list[int] | None = None

    def __post_init__(self):
        assert (self.values is None) or (len(self.indices) == len(self.values))

    def to_dict_old(self):
        """旧的序列化格式：分离的 indices 和 values 列表。"""
        d = {"indices": self.indices}
        if self.values is not None:
            d["values"] = self.values
        return d

    def to_dict(self):
        """新的序列化格式：{index: value} 映射字典。"""
        if self.values is None:
            raise ValueError("SparseVector.values is None")
        result = {}
        for i, v in zip(self.indices, self.values):
            result[str(i)] = v
        return result

    @staticmethod
    def from_dict(d):
        """从字典反序列化。"""
        return SparseVector(d["indices"], d.get("values"))

    def __str__(self):
        return f"SparseVector(indices={self.indices}{'' if self.values is None else f', values={self.values}'})"

    def __repr__(self):
        return str(self)

class MatchTextExpr:
    """全文搜索匹配表达式。

    用于构建基于 BM25 等算法的文本搜索条件。
    """
    def __init__(
        self,
        fields: list[str],
        matching_text: str,
        topn: int,
        extra_options: dict | None = None,
    ):
        self.fields = fields
        self.matching_text = matching_text
        self.topn = topn
        self.extra_options = extra_options


class MatchDenseExpr:
    """稠密向量（Dense Vector）匹配表达式。

    用于基于向量相似度（如余弦距离、欧氏距离）的语义搜索。
    """
    def __init__(
        self,
        vector_column_name: str,
        embedding_data: VEC,
        embedding_data_type: str,
        distance_type: str,
        topn: int = DEFAULT_MATCH_VECTOR_TOPN,
        extra_options: dict | None = None,
    ):
        self.vector_column_name = vector_column_name
        self.embedding_data = embedding_data
        self.embedding_data_type = embedding_data_type
        self.distance_type = distance_type
        self.topn = topn
        self.extra_options = extra_options


class MatchSparseExpr:
    """稀疏向量（Sparse Vector）匹配表达式。

    用于基于稀疏嵌入的检索，常见于混合搜索中结合稠密向量和关键词匹配。
    """
    def __init__(
        self,
        vector_column_name: str,
        sparse_data: SparseVector | dict,
        distance_type: str,
        topn: int,
        opt_params: dict | None = None,
    ):
        self.vector_column_name = vector_column_name
        self.sparse_data = sparse_data
        self.distance_type = distance_type
        self.topn = topn
        self.opt_params = opt_params


class MatchTensorExpr:
    """张量匹配表达式。

    用于基于张量数据的搜索，适用于 ColBERT 等多向量检索场景。
    """
    def __init__(
        self,
        column_name: str,
        query_data: VEC,
        query_data_type: str,
        topn: int,
        extra_option: dict | None = None,
    ):
        self.column_name = column_name
        self.query_data = query_data
        self.query_data_type = query_data_type
        self.topn = topn
        self.extra_option = extra_option


class FusionExpr:
    """融合表达式。

    用于组合多个搜索子表达式的结果，支持加权求和（weighted_sum）等融合策略。
    通常在混合搜索中用于合并文本匹配和向量匹配的排序结果。
    """
    def __init__(self, method: str, topn: int, fusion_params: dict | None = None):
        self.method = method
        self.topn = topn
        self.fusion_params = fusion_params


# 匹配表达式的联合类型：支持文本、稠密向量、稀疏向量、张量和融合五种搜索方式
MatchExpr = MatchTextExpr | MatchDenseExpr | MatchSparseExpr | MatchTensorExpr | FusionExpr


class OrderByExpr:
    """排序表达式。

    支持链式调用添加多个排序字段：
        OrderByExpr().asc("field1").desc("field2")
    """
    def __init__(self):
        self.fields = list()
    def asc(self, field: str):
        self.fields.append((field, 0))
        return self
    def desc(self, field: str):
        self.fields.append((field, 1))
        return self
    def fields(self):
        return self.fields


class DocStoreConnection(ABC):
    """
    文档存储连接抽象基类。

    定义了所有文档存储引擎必须实现的统一接口，包括：
    - 数据库操作（db_type, health）
    - 表操作（create_idx, delete_idx, index_exist）
    - CRUD 操作（search, get, insert, update, delete）
    - 搜索结果辅助函数（get_total, get_doc_ids, get_fields, get_highlight, get_aggregation）
    - SQL 执行（sql）
    """

    @abstractmethod
    def db_type(self) -> str:
        """
        Return the type of the database.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def health(self) -> dict:
        """
        Return the health status of the database.
        """
        raise NotImplementedError("Not implemented")

    """
    表 / 索引操作
    """

    @abstractmethod
    def create_idx(self, index_name: str, dataset_id: str, vector_size: int, parser_id: str = None):
        """
        Create an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete_idx(self, index_name: str, dataset_id: str):
        """
        Delete an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def index_exist(self, index_name: str, dataset_id: str) -> bool:
        """
        Check if an index with given name exists
        """
        raise NotImplementedError("Not implemented")

    """
    CRUD 操作
    """

    @abstractmethod
    def search(
        self, select_fields: list[str],
            highlight_fields: list[str],
            condition: dict,
            match_expressions: list[MatchExpr],
            order_by: OrderByExpr,
            offset: int,
            limit: int,
            index_names: str|list[str],
            dataset_ids: list[str],
            agg_fields: list[str] | None = None,
            rank_feature: dict | None = None
    ):
        """
        Search with given conjunctive equivalent filtering condition and return all fields of matched documents
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get(self, data_id: str, index_name: str, dataset_ids: list[str]) -> dict | None:
        """
        Get single chunk with given id
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def insert(self, rows: list[dict], index_name: str, dataset_id: str = None) -> list[str]:
        """
        Update or insert a bulk of rows
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self, condition: dict, new_value: dict, index_name: str, dataset_id: str) -> bool:
        """
        Update rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, condition: dict, index_name: str, dataset_id: str) -> int:
        """
        Delete rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")

    """
    搜索结果的辅助函数
    """

    @abstractmethod
    def get_total(self, res):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_doc_ids(self, res):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_highlight(self, res, keywords: list[str], field_name: str):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_aggregation(self, res, field_name: str):
        raise NotImplementedError("Not implemented")

    """
    SQL
    """
    @abstractmethod
    def sql(self, sql: str, fetch_size: int, format: str):
        """
        Run the sql generated by text-to-sql
        """
        raise NotImplementedError("Not implemented")
