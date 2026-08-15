use_sql 是 RAGFlow 聊天里的结构化数据查询引擎：当知识库配置了 field_map（字段映射，通常来自
Excel/表格类文档）时，用「自然语言 → SQL →
直接查文档索引」的方式回答，而不是走向量检索。下面是它的运行流程与设计方案。                             
# 一、触发位置
（在 async_chat 中） use_sql 是在常规向量检索之前被尝试的，且失败会自动回退： mermaid

```mermaid                                                                                                                                                                                                                                                                                                                                                                                                                        
  flowchart TD
    A["async_chat 收到用户问题"] --> B["field_map = KnowledgebaseService.get_field_map(kb_ids)"]
    B --> C{"field_map 非空?"}
    C -- 否 --> V["走向量/全文/图谱检索"]
    C -- 是 --> D["ans = await use_sql(...)"]
    D --> E{"ans 有效?<br/>(有 chunks 或 answer)"}
    E -- 是 --> F["yield ans 并 return<br/>（不再走向量检索）"]
    E -- 否 --> G["记录 fallback 日志"] --> V  
```                                                                                                                                                                                                                                                                                                                                                                                                                 

- field_map 来源：knowledgebase_service.get_field_map，合并各 KB 的 parser_config["field_map"]（键=字段名，值=类型/显示名）。
- 关键判断：if ans and (ans["reference"]["chunks"] or ans["answer"]) —— 聚合查询即使 chunks 为空但 answer 有效也算成功。

# 二、use_sql 内部运行流程

```mermaid                                                                                                                                                                                                                                                                                                                                                                                                                        
  flowchart TD
    S["use_sql(question, field_map, tenant_id, chat_mdl, quota, kb_ids)"] --> E{"检测文档引擎<br/>DOC_ENGINE_*"}
    E -->|" Infinity "| T1["表名 ragflow_{tenant}_{kb_id}<br/>docnm 列名=docnm"]
    E -->|" OceanBase "| T2["表名 ragflow_{tenant}<br/>docnm 列名=docnm_kwd"]
    E -->|" Elasticsearch "| T3["表名 ragflow_{tenant}<br/>docnm 列名=docnm_kwd<br/>kb_id 走 WHERE"]
    T1 & T2 & T3 --> G["按引擎构造 sys/user prompt"]
    G --> R{"行数问题?<br/>is_row_count_question"}
    R -- 是 --> OV["直接 override 为<br/>SELECT COUNT(*) AS rows"]
    R -- 否 --> LLM["chat_mdl.async_chat 生成 SQL"]
    OV & LLM --> N["normalize_sql：去 &lt;think&gt;/代码块/分号"]
    N --> K["add_kb_filter：ES/OS 注入 kb_id<br/>（UUID 校验防注入）"]
    K --> Q["retriever.sql_retrieval(sql, format='json')"]
    Q --> C{"执行成功?"}
    C -- 否 --> RTRY["带错误信息重写 SQL 重试一次"]
    RTRY --> C2{"重试成功?"}
    C2 -- 否 --> RET["return None（回退向量检索）"]
    C -- 是 --> R0{"rows 为空?"}
    C2 -- 是 --> R0
    R0 -- 是 --> RET
    R0 -- 否 --> AGG{"非聚合且缺 doc_id/docnm?"}
    AGG -- 是 --> REP["repair_table_for_missing_source_columns<br/>LLM 重写补列（仅一次）"]
    AGG -- 否 --> BUILD
    REP --> BUILD["构建 Markdown 表格 + 列名映射显示名 + Source 列 ##idx$$"]
    BUILD --> SRC{"结果含 doc_id 与 docnm?"}
    SRC -- 否且聚合 --> CHK["按同一 WHERE 再查一次<br/>select doc_id,docnm_kwd ... limit 20"]
    SRC -- 是 --> OUT["返回 {answer, reference{chunks, doc_aggs}, prompt}"]
    CHK --> OUT                                                                                                                                                                                                                                                                                                                                                                                                                 
```                                                                                                                                                                                                                                                                                                                                                                                                                             

# 三、设计方案要点

1. 多引擎适配（表名策略不同是核心差异）

- Infinity：每个 KB 一张表，kb_id 编码进表名 ragflow_{tenant}_{kb_id}，且 docnm 列名不带 _kwd 后缀。
- ES/OpenSearch/OceanBase：共享基础索引 ragflow_{tenant}，kb_id 用 WHERE 过滤（单 KB =，多 KB OR）。
- JSON 字段引擎（Infinity/OceanBase）用 json_extract_string(chunk_data, '$.Field')，ES 用直接字段名。

2. SQL 注入防护

- 所有 kb_id 在拼进 SQL 前经 _assert_valid_uuid 校验为规范 UUID，非法直接抛 ValueError。
- 表名由 index_name(tenant_id) + UUID 派生，不接受外部原始输入。

3. 自我修复闭环（三次 LLM 调用机会，各有明确边界）

- 第 1 次：正常生成 SQL。
- 第 2 次：执行报错 → 把错误信息回喂给 LLM 重写。
- 第 3 次：执行成功但非聚合查询缺 doc_id/docnm（没法做引用）→ 用 repair_prompt 补列。这一层是最近加的（对应
  test_dialog_service_use_sql_source_columns.py），且只重试一次。

4. 聚合 vs 非聚合 分开处理

- is_aggregate_sql() 用正则匹配 count/sum/avg/max/min/distinct(。
- 聚合查询天然没有 doc_id，所以表格照常返回，但会额外按原 WHERE 再查一条 select doc_id, docnm_kwd ... limit 20 来补 source
  chunks（否则引用为空）。

5. 引用追踪（与向量检索复用同一套机制）

- 表格每行追加 Source 列 + ##idx$$ 标记。
- reference.chunks 里带 doc_id/docnm_kwd/kb_id；多 KB 时 kb_id 来自行内 kb_id 列，单 KB 时直接取 kb_ids[0]（_
  chunk_kb_id_for_doc）。
- doc_aggs 按 doc 聚合计数，供上游做来源统计；prompt 字段透传 sys_prompt。

6. 关键状态位

- quota 参数目前传进去但 rows 拼接逻辑里 if quota 与 else 分支完全一致，属于遗留/占位，未真正实现 token 配额截断。

需要的话我可以再画一张 field_map 从 parser_config 一路到 SQL 提示词的时序图，或把某个分支（如聚合查询补 chunks 的细节）展开。 

# 4.提示词模板
# 4.1. infinity/oceanbase 提示词模板
``` 
You are a Database Administrator. Write SQL for a table with JSON 'chunk_data' column.

JSON Extraction: json_extract_string(chunk_data, '$.FieldName')
Numeric Cast: CAST(json_extract_string(chunk_data, '$.FieldName') AS INTEGER/FLOAT)
NULL Check: json_extract_isnull(chunk_data, '$.FieldName') == false

RULES:
1. Use EXACT field names (case-sensitive) from the list below
2. For SELECT: include doc_id, docnm_kwd, and json_extract_string() for requested fields
3. For COUNT: use COUNT(*) or COUNT(DISTINCT json_extract_string(...))
4. Add AS alias for extracted field names
5. DO NOT select 'content' field
6. Only add NULL check (json_extract_isnull() == false) in WHERE clause when:
   - Question asks to "show me" or "display" specific columns
   - Question mentions "not null" or "excluding null"
   - Add NULL check for count specific column
   - DO NOT add NULL check for COUNT(*) queries (COUNT(*) counts all rows including nulls)
7. Output ONLY the SQL, no explanations"""
        user_prompt = """Table: {}
Fields (EXACT case): {}
{}
Question: {}
Write SQL using json_extract_string() with exact field names. Include doc_id, docnm_kwd for data queries. Only SQL.
``` 
根据您提供的SQL生成规则，这是一个**针对包含JSON列`chunk_data`的数据库表，根据用户问题生成SQL查询语句的提示词模板**。

## 整体作用

这个提示词用于指导LLM（大语言模型）将**自然语言问题**转换为**SQL查询语句**，专门针对存储JSON数据的表结构。

---

## 逐部分解析

### 1. 系统角色设定
```
You are a Database Administrator. Write SQL for a table with JSON 'chunk_data' column.
```
- 设定AI角色为**数据库管理员**
- 目标表包含一个名为 `chunk_data` 的JSON类型列

---

### 2. JSON数据提取方法
| 函数 | 用途 | 示例 |
|------|------|------|
| `json_extract_string(chunk_data, '$.FieldName')` | 从JSON中提取字符串值 | `json_extract_string(chunk_data, '$.title')` |
| `CAST(json_extract_string(...) AS INTEGER/FLOAT)` | 提取并转换为数值类型 | `CAST(json_extract_string(chunk_data, '$.age') AS INTEGER)` |
| `json_extract_isnull(chunk_data, '$.FieldName') == false` | 判断字段是否为NULL（过滤条件） | 用于WHERE子句中排除NULL值 |

---

### 3. 核心规则（6条）

| 规则 | 说明 |
|------|------|
| **① 精确字段名** | 使用列表中的字段名，**区分大小写**（JSON字段通常大小写敏感） |
| **② SELECT语句** | 必须包含 `doc_id`、`docnm` 以及用 `json_extract_string()` 提取的请求字段，并添加别名 |
| **③ COUNT语句** | 使用 `COUNT(*)` 或 `COUNT(DISTINCT json_extract_string(...))` |
| **④ 添加别名** | 为提取的JSON字段添加 `AS alias_name` |
| **⑤ 禁止选择content** | 不查询 `content` 字段（可能是大文本字段，避免性能问题） |
| **⑥ NULL检查条件** | 仅在以下情况添加 `json_extract_isnull() == false`：<br>• 问题要求"显示"或"展示"特定列<br>• 问题提到"非空"或"排除NULL"<br>• 对**特定列**进行计数时<br>• **COUNT(*) 不加NULL检查**（因为COUNT(*)统计所有行） |

---

### 4. 输出要求
```
Output ONLY the SQL, no explanations
```
- 只输出SQL语句，不要有任何解释或额外文本

---

### 5. 用户输入占位符
```
Table: {}          # 表名
Fields (EXACT case): {}  # 可用字段列表（区分大小写）
{}                 # 额外的SQL提示/条件
Question: {}       # 用户的自然语言问题
Write SQL using json_extract_string() with exact field names. Include doc_id, docnm for data queries. Only SQL.
```

---

## 使用示例

**输入：**
```
Table: documents
Fields (EXACT case): title, author, pages, publish_date
Question: 显示所有title非空的文档的标题和作者
```

**预期输出SQL：**
```sql
SELECT 
    doc_id, 
    docnm, 
    json_extract_string(chunk_data, '$.title') AS title,
    json_extract_string(chunk_data, '$.author') AS author
FROM documents
WHERE json_extract_isnull(chunk_data, '$.title') == false
```

---

## 在您项目中的位置

这个提示词很可能用于：
- **DeepResearcher** 或 **TreeStructuredQueryDecompositionRetrieval** 中
- 当需要从结构化JSON数据中检索信息时
- 作为LLM生成检索/查询语句的指令模板
- 与知识库检索（`kb_retrieve`）配合使用，实现语义查询到SQL的转换

# 4.2.ES/OS提示词模板
```
You are a Database Administrator. Write SQL queries.

RULES:
1. Use EXACT field names from the schema below (e.g., product_tks, not product)
2. Quote field names starting with digit: "123_field"
3. Add IS NOT NULL in WHERE clause when:
   - Question asks to "show me" or "display" specific columns
4. Include doc_id/docnm in non-aggregate statement
5. Output ONLY the SQL, no explanations"""
        user_prompt = """Table: {}
Available fields:
{}
Question: {}
Write SQL using exact field names above. Include doc_id, docnm_kwd for data queries. Only SQL.
```
## 提示词模板解析

这是一个**简化版SQL生成提示词模板**，专门用于从**结构化表格数据**（非JSON）生成SQL查询语句。

---

### 规则1：使用精确字段名
```
Use EXACT field names from the schema below (e.g., product_tks, not product)
```

| 要点 | 说明 |
|------|------|
| **必须使用** | Schema中定义的字段名，不能使用别名或简称 |
| **示例** | Schema里是 `product_tks`，就不能写成 `product` |
| **原因** | 避免字段名不匹配导致的SQL执行错误 |

---

### 规则2：字段名以数字开头需加引号
```
Quote field names starting with digit: "123_field"
```

| 要点 | 说明 |
|------|------|
| **问题场景** | 字段名以数字开头（如 `123_field`、`2024_year`） |
| **解决方法** | 用双引号包裹：`SELECT "123_field" FROM table` |
| **原因** | 大多数数据库不允许标识符以数字开头，加引号可转义 |
| **注意** | 其他特殊字符（空格、连字符）也可能需要引号，但规则只强调了数字开头 |

---

### 规则3：NULL检查条件
```
Add IS NOT NULL in WHERE clause when:
- Question asks to "show me" or "display" specific columns
```

| 场景 | 是否添加 IS NOT NULL | 示例 |
|------|---------------------|------|
| 问题含"显示"、"展示" | ✅ 添加 | "显示所有产品名称" → `WHERE product_name IS NOT NULL` |
| 问题含"查询"、"统计" | ❌ 不添加 | "统计总销售额" → 不加NULL检查 |
| 问题含"非空"、"不为空" | ✅ 添加 | "找出有价格的商品" → `WHERE price IS NOT NULL` |

**逻辑**：当用户要"看"数据时，过滤掉NULL值展示更干净；做聚合统计时，保留NULL（或根据业务需求决定）。

---

### 规则4：非聚合查询必须包含doc_id/docnm
```
Include doc_id/docnm in non-aggregate statement
```

| 类型 | 要求 | 说明 |
|------|------|------|
| 非聚合查询（SELECT明细） | ✅ 必须包含 | 确保每条记录可追溯来源文档 |
| 聚合查询（COUNT/SUM等） | ❌ 不要求 | 聚合结果不涉及具体记录 |

**示例：**
```sql
-- ✅ 正确（非聚合）
SELECT doc_id, docnm_kwd, product_name, price
FROM table
WHERE product_name IS NOT NULL;

-- ❌ 错误（缺少doc_id）
SELECT product_name, price
FROM table;

-- ✅ 聚合查询可以不包含
SELECT COUNT(*) AS total
FROM table;
```

---

### 规则5：只输出SQL
```
Output ONLY the SQL, no explanations
```

| 要求 | 说明 |
|------|------|
| **不要解释** | 不输出"这是您的SQL"等说明文字 |
| **不要注释** | 除非必要，不添加SQL注释 |
| **纯文本SQL** | 输出可直接执行的SQL语句 |
