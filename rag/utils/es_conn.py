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

import re
import json
import time

import copy
from elasticsearch_dsl import UpdateByQuery, Q, Search
from elastic_transport import ConnectionTimeout
from common.decorator import singleton
from common.doc_store.doc_store_base import MatchTextExpr, OrderByExpr, MatchExpr, MatchDenseExpr, FusionExpr
from common.doc_store.es_conn_base import ESConnectionBase
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD

ATTEMPT_TIME = 2
MAX_RESULT_WINDOW = 10000
SEARCH_AFTER_BATCH_SIZE = 1000

# Single-document atomic pagerank_fea adjust (chunk feedback). Clamps using params.min_w / max_w;
# removes field at zero for rank_feature compatibility.
_PAGERANK_FEA_ADJUST_SCRIPT = """
double cur = 0.0;
if (ctx._source.containsKey(params.pf)) {
  Object v = ctx._source[params.pf];
  if (v != null) {
    if (v instanceof Number) {
      cur = ((Number)v).doubleValue();
    } else {
      try { cur = Double.parseDouble(v.toString()); } catch (Exception e) { cur = 0.0; }
    }
  }
}
double nw = cur + params.delta;
if (nw < params.min_w) { nw = params.min_w; }
if (nw > params.max_w) { nw = params.max_w; }
if (nw <= 0.0) {
  if (ctx._source.containsKey(params.pf)) {
    ctx._source.remove(params.pf);
  }
} else {
  ctx._source[params.pf] = nw;
}
"""


@singleton
class ESConnection(ESConnectionBase):
    """
    CRUD operations
    """

    def _es_search_once(self, index_names: list[str], query: dict, track_total_hits: bool):
        """
        单次ES搜索
        """
        # 支持超时设置（600秒）
        # 控制是否跟踪总命中数
        return self.es.search(
            index=index_names,
            body=query,
            timeout="600s",
            track_total_hits=track_total_hits
        )

    def _search_with_search_after(self, index_names: list[str], query: dict, offset: int, limit: int):
        """
        深度分页方法 : ES默认的from+size分页有深度限制（默认10000条），对于大数据集分页会报错。
        工作原理:
        第一次请求: 获取前N条，记录最后一条的sort值作为游标
        后续请求: 使用search_after参数从游标位置继续获取
        """
        q_base = copy.deepcopy(query)
        q_base.pop("from", None)
        q_base.pop("size", None)

        search_after = None
        template_res = None
        collected_hits = []
        remaining_skip = max(0, offset)
        remaining_take = max(0, limit)
        with_aggs = True

        # 阶段1：跳过前offset条（跳过阶段）
        # 每次从上一批次的最后一条文档获取 sort 值作为新的游标
        # 累进减少 remaining_skip
        # 第一次请求保存 template_res（包含 total 信息）
        # 后续请求禁用聚合（with_aggs=False）以提升性能
        while remaining_skip > 0:
            batch = min(SEARCH_AFTER_BATCH_SIZE, remaining_skip)
            q_iter = copy.deepcopy(q_base)
            q_iter["size"] = batch
            if search_after is not None:
                q_iter["search_after"] = search_after
            if not with_aggs:
                q_iter.pop("aggs", None)
            res = self._es_search_once(index_names, q_iter, track_total_hits=template_res is None)

            # 第一次请求保存template_res（包含总数等元数据）
            if template_res is None:
                template_res = res

            # 获取本次结果
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                break

            # 更新游标到最后一条
            next_search_after = hits[-1].get("sort")
            if not next_search_after or next_search_after == search_after:
                break
            search_after = next_search_after
            remaining_skip -= len(hits)
            # 优化技巧：
            # 跳过阶段禁用聚合（with_aggs=False）提升性能
            # 只在第一次请求时计算总数
            # 分批获取，避免内存溢出
            with_aggs = False
            if len(hits) < batch:
                break

        # 阶段2：获取实际数据（取数阶段）
        while remaining_skip <= 0 and remaining_take > 0:
            batch = min(SEARCH_AFTER_BATCH_SIZE, remaining_take)
            q_iter = copy.deepcopy(q_base)
            q_iter["size"] = batch
            if search_after is not None:
                q_iter["search_after"] = search_after
            if not with_aggs:
                q_iter.pop("aggs", None)
            res = self._es_search_once(index_names, q_iter, track_total_hits=template_res is None)
            if template_res is None:
                template_res = res
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                break
            # 同样使用search_after继续获取
            collected_hits.extend(hits)
            remaining_take -= len(hits)
            next_search_after = hits[-1].get("sort")
            if not next_search_after or next_search_after == search_after:
                break
            search_after = next_search_after
            with_aggs = False
            if len(hits) < batch:
                break

        if template_res is None:
            q_count = copy.deepcopy(q_base)
            q_count["size"] = 0
            template_res = self._es_search_once(index_names, q_count, track_total_hits=True)
        template_res["hits"]["hits"] = collected_hits
        return template_res

    def search(
            self, select_fields: list[str],
            highlight_fields: list[str],
            condition: dict,
            match_expressions: list[MatchExpr],
            order_by: OrderByExpr,
            offset: int,
            limit: int,
            index_names: str | list[str],
            knowledgebase_ids: list[str],
            agg_fields: list[str] | None = None,
            rank_feature: dict | None = None
    ):
        """
        Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html

        主搜索接口

        搜索策略决策树
        是否超过深度分页限制(10000)？
        ├─ 否 → 使用普通分页 (from+size)
        └─ 是 → 是否有排序条件？
            ├─ 否 → 普通分页（可能报错）
            └─ 是 → 是否为向量搜索？
                ├─ 是 → 普通分页（knn不支持search_after）
                └─ 否 → 使用search_after深度分页

        :param select_fields:                  # 要返回的字段
        :param highlight_fields:               # 高亮字段
        :param condition:                      # 过滤条件
        :param match_expressions:              # 匹配表达式（文本/向量）
        :param order_by:                       # 排序规则
        :param offset:                         # 偏移量
        :param limit:                          # 限制条数
        :param index_names:                    # 索引名
        :param knowledgebase_ids:              # 知识库ID（数据隔离）
        :param agg_fields:                     # 聚合字段
        :param rank_feature:                   # 排序特征
        :return:
        """
        if isinstance(index_names, str):
            index_names = index_names.split(",")
        assert isinstance(index_names, list) and len(index_names) > 0
        assert "_id" not in condition

        # 模块1：构建过滤条件（Bool Query）
        bool_query = Q("bool", must=[])
        condition["kb_id"] = knowledgebase_ids # 强制数据隔离
        for k, v in condition.items():
            if k == "available_int":
                # 特殊处理可用状态
                if v == 0:
                    bool_query.filter.append(Q("range", available_int={"lt": 1}))
                else:
                    bool_query.filter.append(
                        Q("bool", must_not=Q("range", available_int={"lt": 1})))
                continue
            if k == "id":
                # 支持同时查询id和_id字段
                if not v:
                    continue
                if isinstance(v, list):
                    bool_query.filter.append(
                        Q("bool", should=[Q("terms", id=v), Q("terms", _id=v)], minimum_should_match=1))
                elif isinstance(v, str) or isinstance(v, int):
                    bool_query.filter.append(
                        Q("bool", should=[Q("term", id=v), Q("term", _id=v)], minimum_should_match=1))
                continue
            if not v:
                continue
            # 普通字段条件
            if isinstance(v, list):
                bool_query.filter.append(Q("terms", **{k: v}))
            elif isinstance(v, str) or isinstance(v, int):
                bool_query.filter.append(Q("term", **{k: v}))
            else:
                raise Exception(
                    f"Condition `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str or list.")

        s = Search()
        vector_similarity_weight = 0.5
        # 模块3：混合搜索权重计算 : 实现文本和向量的加权混合搜索。
        for m in match_expressions:
            # 从FusionExpr中提取权重
            if isinstance(m, FusionExpr) and m.method == "weighted_sum" and "weights" in m.fusion_params:
                assert len(match_expressions) == 3 and isinstance(match_expressions[0], MatchTextExpr) and isinstance(
                    match_expressions[1],
                    MatchDenseExpr) and isinstance(
                    match_expressions[2], FusionExpr)
                weights = m.fusion_params["weights"]
                vector_similarity_weight = get_float(weights.split(",")[1])
        # 模块2：处理匹配表达式
        for m in match_expressions:
            if isinstance(m, MatchTextExpr):
                minimum_should_match = m.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    minimum_should_match = str(int(minimum_should_match * 100)) + "%"
                bool_query.must.append(Q("query_string", fields=m.fields,
                                         type="best_fields", query=m.matching_text,
                                         minimum_should_match=minimum_should_match,
                                         boost=1))
                # 文本查询的boost设置为(1 - 向量权重)
                bool_query.boost = 1.0 - vector_similarity_weight

            # 向量匹配（MatchDenseExpr）
            elif isinstance(m, MatchDenseExpr):
                assert (bool_query is not None)
                similarity = 0.0
                if "similarity" in m.extra_options:
                    similarity = m.extra_options["similarity"]
                s = s.knn(m.vector_column_name,     # 向量字段名
                          m.topn,                   # 返回数量
                          m.topn * 2,               # 候选数量
                          query_vector=list(m.embedding_data),
                          filter=bool_query.to_dict(),
                          similarity=similarity,
                          )

        if bool_query and rank_feature:
            for fld, sc in rank_feature.items():
                if fld != PAGERANK_FLD:
                    fld = f"{TAG_FLD}.{fld}"
                bool_query.should.append(Q("rank_feature", field=fld, linear={}, boost=sc))

        if bool_query:
            s = s.query(bool_query)
        for field in highlight_fields:
            s = s.highlight(field)

        if order_by:
            orders = list()
            # 模块4：排序和分页
            for field, order in order_by.fields:
                order = "asc" if order == 0 else "desc"
                if field in ["page_num_int", "top_int"]:
                    order_info = {"order": order, "unmapped_type": "float",
                                  "mode": "avg", "numeric_type": "double"}
                elif field.endswith("_int") or field.endswith("_flt"):
                    order_info = {"order": order, "unmapped_type": "float"}
                elif field == "id":
                    continue # id as "text", not a "keyword", order by it will cause error
                else:
                    order_info = {"order": order, "unmapped_type": "keyword"}
                orders.append({field: order_info})
            s = s.sort(*orders)
        if agg_fields:
            for fld in agg_fields:
                s.aggs.bucket(f'aggs_{fld}', 'terms', field=fld, size=1000000)

        has_dense = any(isinstance(m, MatchDenseExpr) for m in match_expressions)
        has_explicit_sort = bool(order_by and order_by.fields)
        # 判断是否使用search_after
        use_search_after = (
            limit > 0
            and (offset + limit > MAX_RESULT_WINDOW)        # 超过深度限制
            and has_explicit_sort                           # 有明确排序
            and not has_dense                               # 不是向量搜索
        )

        if limit > 0 and not use_search_after:
            s = s[offset:offset + limit]
        # Filter _source to only requested fields for efficiency, and add vector
        # fields to "fields" param so they appear in hit.fields when ES 9.x
        # exclude_source_vectors is enabled (dense_vector not in _source).
        if select_fields:
            s = s.source(select_fields)
        q = s.to_dict()
        # ES 9.x: dense_vector fields excluded from _source; request them via fields.
        # Note: knn does NOT have a "fields" parameter - adding it inside the knn
        # object causes BadRequestError on ES 9.x. We add "fields" at top level.
        # 模块5：向量字段处理（ES 9.x兼容）
        vector_fields = [f for f in (select_fields or []) if f.endswith("_vec")]
        if vector_fields:
            q["fields"] = vector_fields     # dense_vector字段需要从fields获取
        self.logger.debug(f"ESConnection.search {str(index_names)} query: " + json.dumps(q))

        for i in range(ATTEMPT_TIME):
            try:
                if use_search_after:
                    res = self._search_with_search_after(index_names, q, offset, limit)
                else:
                    # print(json.dumps(q, ensure_ascii=False))
                    res = self._es_search_once(index_names, q, track_total_hits=True)
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                self.logger.debug(f"ESConnection.search {str(index_names)} res: " + str(res))
                return res
            except ConnectionTimeout:
                self.logger.exception("ES request timeout")
                self._connect()
                continue
            except Exception as e:
                # Only log debug for NotFoundError(accepted when metadata index doesn't exist)
                if 'NotFound' in str(e):
                    self.logger.debug(f"ESConnection.search {str(index_names)} query: " + str(q) + " - " + str(e))
                else:
                    self.logger.exception(f"ESConnection.search {str(index_names)} query: " + str(q) + str(e))
                raise e

        self.logger.error(f"ESConnection.search timeout for {ATTEMPT_TIME} times!")
        raise Exception("ESConnection.search timeout.")

    def insert(self, documents: list[dict], index_name: str, knowledgebase_id: str = None) -> list[str]:
        """
        Elasticsearch批量插入操作的实现，用于向ES索引中批量添加文档
        :param documents:           要插入的文档列表（字典格式）
        :param index_name:          ES索引名称
        :param knowledgebase_id:    知识库ID（可选，用于数据隔离）
        :return:                    错误列表（空列表表示全部成功）
        """
        # Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html
        # 构建批量操作（Bulk Operations）
        # # operations列表格式：[操作元数据, 文档数据, 操作元数据, 文档数据, ...]
        # [
        #     {"index": {"_index": "my_index", "_id": "chunk1"}},
        #     {"id": "chunk1", "content": "text", "kb_id": "kb123"},
        #     {"index": {"_index": "my_index", "_id": "chunk2"}},
        #     {"id": "chunk2", "content": "text2", "kb_id": "kb123"}
        # ]
        operations = []
        for d in documents:
            assert "_id" not in d   # 不允许使用ES内部_id
            assert "id" in d        # 必须有业务ID
            d_copy = copy.deepcopy(d) #  深拷贝避免修改原数据
            d_copy["kb_id"] = knowledgebase_id # 添加租户隔离字段
            # Use id as _id for uniqueness, also keep "id" as a regular field for sorting
            meta_id = d_copy.get("id", "")
            operations.append(
                {"index": {"_index": index_name, "_id": meta_id}})
            operations.append(d_copy)

        res = []
        for _ in range(ATTEMPT_TIME):
            try:
                res = []
                # 执行批量插入（带重试）
                # 1.refresh="wait_for"：等待索引刷新后再返回，确保数据立即可查
                # 2.timeout="60s"：操作超时时间
                #
                # 批量操作优势
                # 性能提升：一次网络请求插入多条数据
                # 原子性：部分失败不影响其他文档
                # 错误明细：能精确定位失败的文档
                r = self.es.bulk(index=index_name, operations=operations,
                                 refresh="wait_for", timeout="60s")
                # 错误处理 : 检查整体状态
                if re.search(r"False", str(r["errors"]), re.IGNORECASE):
                    return res # 无错误，返回空列表

                # 提取详细错误
                # 1.遍历每个操作的结果
                # 2.检查是否包含错误信息
                # 3.返回格式："文档ID:错误信息"
                for item in r["items"]:
                    for action in ["create", "delete", "index", "update"]:
                        if action in item and "error" in item[action]:
                            res.append(str(item[action]["_id"]) + ":" + str(item[action]["error"]))
                return res
            # 异常处理
            except ConnectionTimeout:
                self.logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                res.append(str(e))
                self.logger.warning("ESConnection.insert got exception: " + str(e))

        return res

    def update(self, condition: dict, new_value: dict, index_name: str, knowledgebase_id: str) -> bool:
        """
        Elasticsearch更新操作的底层实现，支持单文档更新和批量更新
        1. 更新字段（Update）
        new_value = {
            "content_with_weight": "新内容",
            "available_int": 1
        }
        生成脚本：ctx._source.content_with_weight=params.pp_content_with_weight;ctx._source.available_int=1;

        2. 删除字段（Remove）
        new_value = {
            "remove": "old_field"  # 或 ["field1", "field2"]
        }
        生成脚本：ctx._source.remove('old_field');

        3. 删除数组元素（Remove from array）
        python
        new_value = {
            "remove": {"tag_kwd": "旧标签"}  # 从标签数组中移除特定标签
        }
        生成脚本：
        int i=ctx._source.tag_kwd.indexOf(params.p_tag_kwd);
        ctx._source.tag_kwd.remove(i);

        4. 添加数组元素（Add to array）
        new_value = {
            "add": {"tag_kwd": "新标签"}
        }
        生成脚本：ctx._source.tag_kwd.add(params.pp_tag_kwd);

        :param condition:           查询条件（指定要更新的文档）
        :param new_value:           要更新的字段和值
        :param index_name:          ES索引名
        :param knowledgebase_id:    知识库ID（数据隔离）
        :return:                    成功/失败
        """
        doc = copy.deepcopy(new_value)
        doc.pop("id", None)
        condition["kb_id"] = knowledgebase_id
        # 模式一：单文档更新（精确更新）
        if "id" in condition and isinstance(condition["id"], str):
            # update specific single document
            # 通过ID精确更新单个文档
            chunk_id = condition["id"]
            for i in range(ATTEMPT_TIME):
                doc_part = copy.deepcopy(doc)
                # 步骤2：处理删除操作（remove）
                remove_value = doc_part.pop("remove", None)
                remove_field = remove_value if isinstance(remove_value, str) else None
                remove_dict = remove_value if isinstance(remove_value, dict) else None
                # 步骤1：处理特殊字段（feas字段）
                # 遍历所有字段
                # 识别以_feas结尾的字段（特征字段）
                # 先执行删除操作（如果存在）
                for k in doc_part.keys():
                    if "feas" != k.split("_")[-1]:
                        continue
                    try:
                        self.es.update(index=index_name, id=chunk_id, script=f"ctx._source.remove(\"{k}\");")
                    except Exception:
                        self.logger.exception(
                            f"ESConnection.update(index={index_name}, id={chunk_id}, doc={json.dumps(condition, ensure_ascii=False)}) got exception")
                try:
                    if remove_field is not None:
                        # 删除单个字段
                        self.es.update(
                            index=index_name,
                            id=chunk_id,
                            script=f"ctx._source.remove('{remove_field}');",
                        )
                    if remove_dict is not None:
                        # 从数组中删除特定元素
                        scripts = []
                        params = {}
                        for kk, vv in remove_dict.items():
                            scripts.append(
                                f"if (ctx._source.containsKey('{kk}') && ctx._source.{kk} != null) "
                                f"{{ int i = ctx._source.{kk}.indexOf(params.p_{kk}); "
                                f"if (i >= 0) {{ ctx._source.{kk}.remove(i); }} }}"
                            )
                            params[f"p_{kk}"] = vv
                        if scripts:
                            self.es.update(
                                index=index_name,
                                id=chunk_id,
                                script={"source": "".join(scripts), "params": params},
                            )
                    #  执行文档更新
                    # 使用ES的doc参数进行部分更新
                    # 只更新提供的字段
                    if doc_part:
                        self.es.update(index=index_name, id=chunk_id, doc=doc_part)
                    if remove_field is not None or remove_dict is not None or doc_part:
                        return True
                except Exception as e:
                    self.logger.exception(
                        f"ESConnection.update(index={index_name}, id={chunk_id}, doc={json.dumps(condition, ensure_ascii=False)}) got exception: " + str(
                            e))
                    break
            return False

        # update unspecific maybe-multiple documents
        # 模式二：批量更新流程
        # 步骤1：构建查询条件
        # 支持terms查询（列表匹配）
        # 支持term查询（精确匹配）
        # 支持exists查询（字段存在）
        bool_query = Q("bool")
        for k, v in condition.items():
            if not isinstance(k, str) or not v:
                continue
            if k == "exists":
                bool_query.filter.append(Q("exists", field=v))
                continue
            if isinstance(v, list):
                bool_query.filter.append(Q("terms", **{k: v}))
            elif isinstance(v, str) or isinstance(v, int):
                bool_query.filter.append(Q("term", **{k: v}))
            else:
                raise Exception(
                    f"Condition `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str or list.")
        # 步骤2：构建更新脚本
        scripts = []
        params = {}
        for k, v in new_value.items():
            if k == "remove":
                # 删除字段或数组元素
                if isinstance(v, str):
                    scripts.append(f"ctx._source.remove('{v}');")
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        scripts.append(f"int i=ctx._source.{kk}.indexOf(params.p_{kk});ctx._source.{kk}.remove(i);")
                        params[f"p_{kk}"] = vv
                continue
            if k == "add":
                # 向数组添加元素
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        scripts.append(f"ctx._source.{kk}.add(params.pp_{kk});")
                        params[f"pp_{kk}"] = vv.strip()
                continue
            if (not isinstance(k, str) or not v) and k != "available_int":
                continue
            # 普通字段更新
            if isinstance(v, str):
                v = re.sub(r"(['\n\r]|\\.)", " ", v) # 清理特殊字符
                params[f"pp_{k}"] = v
                scripts.append(f"ctx._source.{k}=params.pp_{k};")
            elif isinstance(v, int) or isinstance(v, float):
                scripts.append(f"ctx._source.{k}={v};")
            elif isinstance(v, list):
                scripts.append(f"ctx._source.{k}=params.pp_{k};")
                params[f"pp_{k}"] = json.dumps(v, ensure_ascii=False)
            else:
                raise Exception(
                    f"newValue `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str.")
        # 步骤3：执行批量更新
        ubq = UpdateByQuery(
            index=index_name).using(
            self.es).query(bool_query)
        ubq = ubq.script(source="".join(scripts), params=params)
        ubq = ubq.params(refresh=True)
        ubq = ubq.params(slices=5) # 并行分片
        ubq = ubq.params(conflicts="proceed") # 冲突时继续

        for _ in range(ATTEMPT_TIME):
            try:
                _ = ubq.execute()
                return True
            except ConnectionTimeout:
                self.logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                self.logger.error("ESConnection.update got exception: " + str(e) + "\n".join(scripts))
                break
        return False

    def adjust_chunk_pagerank_fea(
        self,
        chunk_id: str,
        index_name: str,
        knowledgebase_id: str,
        delta: float,
        min_w: float = 0.0,
        max_w: float = 100.0,
        row_id: int | None = None,
    ) -> bool:
        """Atomically adjust pagerank_fea on one chunk (painless script)."""
        _ = row_id
        for _ in range(ATTEMPT_TIME):
            try:
                self.es.update(
                    index=index_name,
                    id=chunk_id,
                    retry_on_conflict=3,
                    script={
                        "source": _PAGERANK_FEA_ADJUST_SCRIPT.strip(),
                        "lang": "painless",
                        "params": {
                            "pf": PAGERANK_FLD,
                            "delta": float(delta),
                            "min_w": float(min_w),
                            "max_w": float(max_w),
                        },
                    },
                )
                self.logger.debug(
                    "ESConnection.adjust_chunk_pagerank_fea(index=%s, id=%s, delta=%s) succeeded",
                    index_name,
                    chunk_id,
                    delta,
                )
                return True
            except ConnectionTimeout:
                self.logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                self.logger.exception(
                    "ESConnection.adjust_chunk_pagerank_fea(index=%s, id=%s): %s",
                    index_name,
                    chunk_id,
                    e,
                )
                if re.search(r"connection", str(e).lower()):
                    time.sleep(3)
                    self._connect()
                    continue
                break
        return False

    def delete(self, condition: dict, index_name: str, knowledgebase_id: str) -> int:
        """
        Elasticsearch删除操作的实现，属于数据存储层的方法
        :param condition:           删除条件字典
        :param index_name:          ES索引名称
        :param knowledgebase_id:    知识库ID（用于数据隔离）
        :return:                    删除的文档数量（int）
        """
        # 条件预处理
        # 断言：不允许直接使用_id（ES内部ID）
        assert "_id" not in condition
        # 强制添加kb_id条件，确保数据隔离（多租户安全）
        condition["kb_id"] = knowledgebase_id

        # Build a bool query that combines id filter with other conditions
        # 构建布尔查询（Bool Query）
        # 1.使用Elasticsearch的Bool Query组合多个条件
        # 2.包含：filter（过滤）、must（必须匹配）、must_not（必须不匹配）
        bool_query = Q("bool")

        # Handle chunk IDs if present
        # 处理块ID（特殊逻辑）
        # 1.id字段：业务ID（不是ES的_id）
        # 2.使用ids查询进行精确匹配
        # 3.如果是空列表，不添加此条件（相当于忽略）
        if "id" in condition:
            chunk_ids = condition["id"]
            if not isinstance(chunk_ids, list):
                chunk_ids = [chunk_ids]
            if chunk_ids:
                # Filter by specific chunk IDs
                bool_query.filter.append(Q("ids", values=chunk_ids))
            # If chunk_ids is empty, we don't add an ids filter - rely on other conditions

        # Add all other conditions as filters
        # 处理其他条件
        # 支持的条件类型：
        # 条件类型	说明	        ES查询类型
        # exists	字段存在	    exists query
        # must_not	否定条件	    must_not 子句
        # list	    多值匹配	    terms query
        # str/int	精确匹配	    term query
        for k, v in condition.items():
            if k == "id":
                continue  # Already handled above
            if k == "exists":
                bool_query.filter.append(Q("exists", field=v))
            elif k == "must_not":
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk == "exists":
                            bool_query.must_not.append(Q("exists", field=vv))
            elif isinstance(v, list):
                bool_query.must.append(Q("terms", **{k: v}))
            elif isinstance(v, str) or isinstance(v, int):
                bool_query.must.append(Q("term", **{k: v}))
            elif v is not None:
                raise Exception("Condition value must be int, str or list.")

        # If no filters were added, use match_all (for tenant-wide operations)
        # 查询构建
        # 1.如果没有条件，使用match_all（删除所有文档）
        # 2.否则使用构建的bool查询
        if not bool_query.filter and not bool_query.must and not bool_query.must_not:
            qry = Q("match_all")
        else:
            qry = bool_query
        self.logger.debug("ESConnection.delete query: " + json.dumps(qry.to_dict()))
        # 执行删除（带重试机制）
        # 重试机制：
        # 连接超时：等待3秒后重连，继续重试
        # 索引不存在：返回0（幂等性）
        # 其他异常：记录日志后返回0
        #
        # 性能优化
        # 使用refresh=True：删除后立即刷新，保证一致性
        # 使用delete_by_query：批量删除，比逐条删除高效
        for _ in range(ATTEMPT_TIME):
            try:
                res = self.es.delete_by_query(
                    index=index_name,
                    body=Search().query(qry).to_dict(),
                    refresh=True)
                return res["deleted"]
            except ConnectionTimeout:
                self.logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                self.logger.warning("ESConnection.delete got exception: " + str(e))
                if re.search(r"(not_found)", str(e), re.IGNORECASE):
                    return 0
        return 0

    """
    Helper functions for search result
    """

    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        res_fields = {}
        if not fields:
            return {}
        hits = res.get("hits", {}).get("hits", [])
        for hit in hits:
            doc_id = hit.get("_id")
            d = hit.get("_source", {})
            # Also extract fields from ES "fields" response (used by dense_vector in ES 9.x)
            hit_fields = hit.get("fields", {})
            m = {}
            for n in fields:
                # First check _source
                if d.get(n) is not None:
                    m[n] = d.get(n)
                # Then check fields (ES 9.x stores dense_vector here, not in _source)
                elif n in hit_fields:
                    vals = hit_fields[n]
                    # ES fields response wraps dense_vector in 2 levels: [[v1,v2,...]] -> [v1,v2,...]
                    if isinstance(vals, list) and len(vals) == 1:
                        vals = vals[0]
                    m[n] = vals
            for n, v in m.items():
                if isinstance(v, list):
                    m[n] = v
                    continue
                if n == "available_int" and isinstance(v, (int, float)):
                    m[n] = v
                    continue
                if not isinstance(v, str):
                    m[n] = str(m[n])
                # if n.find("tks") > 0:
                #     m[n] = remove_redundant_spaces(m[n])

            if m:
                res_fields[doc_id] = m
        return res_fields
