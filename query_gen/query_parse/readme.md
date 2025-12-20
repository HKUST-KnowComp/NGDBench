# Cypher Parser & AST Generator

## 简介

这是基于 **ANTLR4 Cypher 语法** 的轻量级 Cypher 解析器，用于将 **Cypher 查询语句** 转换为一个 **结构化、可序列化的 JSON AST（抽象语法树）**。

该 AST 并不是完整复刻 Neo4j 的执行语义，而是一个 面向分析、理解、重写与下游处理（如 LLM Prompt、图模式抽取、查询解释）的中间表示。


---

## 整体流程

```text
Cypher Query (string)
        ↓
ANTLR Lexer / Parser
        ↓
Parse Tree (Concrete Syntax Tree)
        ↓
CypherASTVisitor (Visitor Pattern)
        ↓
JSON AST (Python dict)
```

核心入口函数：

```python
parse_cypher_to_ast(query: str) -> dict
```

---

## AST 总体结构

解析完成后，返回的 AST 结构如下：

```json
{
  "type": "query",
  "clauses": [...],
  "return": [...],
  "limit": 20
}
```

### 顶层字段说明

| 字段名       | 类型         | 含义                               |
| --------- | ---------- | -------------------------------- |
| `type`    | string     | AST 根节点类型，固定为 `query`            |
| `clauses` | list       | 查询中的 MATCH / OPTIONAL MATCH 子句列表 |
| `return`  | list       | RETURN 子句中的投影项（字符串形式）            |
| `limit`   | int / null | LIMIT 的数值，如果不存在则为 `null`         |

---

## MATCH / OPTIONAL MATCH Clause

每一个 `MATCH` 或 `OPTIONAL MATCH` 会生成一个 clause 对象：

```json
{
  "type": "match",
  "optional": false,
  "patterns": [...]
}
```

### 字段说明

| 字段名        | 类型      | 含义                 |
| ---------- | ------- | ------------------ |
| `type`     | string  | 固定为 `match`        |
| `optional` | boolean | 是否为 OPTIONAL MATCH |
| `patterns` | list    | 该 MATCH 子句中的图模式    |

---

## Pattern（图模式）

`patterns` 中的元素来自 `patternPart`，可能是：

* 普通匿名模式：`(a)-[:R]->(b)`
* 带路径变量的模式：`p = (a)-[:R]->(b)`

### 1️⃣ Path Pattern（带路径变量）

```json
{
  "type": "path",
  "variable": "p",
  "elements": [...]
}
```

| 字段名        | 类型     | 含义              |
| ---------- | ------ | --------------- |
| `type`     | string | 固定为 `path`      |
| `variable` | string | 路径变量名（如 `p`）    |
| `elements` | list   | 路径上的节点与关系，按顺序排列 |

`elements` 的顺序示例：

```text
(node) → (relationship) → (node) → (relationship) → (node)
```

---

### 2️⃣ Node（节点模式）

```json
{
  "type": "node",
  "variable": "a",
  "labels": ["Drug"],
  "properties": {
    "id": "1904862"
  }
}
```

| 字段名          | 类型            | 含义                       |
| ------------ | ------------- | ------------------------ |
| `type`       | string        | 固定为 `node`               |
| `variable`   | string / null | 节点变量名，如 `a`，匿名节点为 `null` |
| `labels`     | list[string]  | 节点标签，如 `:Drug`           |
| `properties` | dict          | 节点属性键值对（当前为字符串简化版）       |

> ⚠️ 注意：当前实现中属性值直接使用 `getText()`，未做表达式 AST 化。

---

### 3️⃣ Relationship（关系模式）

```json
{
  "type": "relationship",
  "labels": ["contraindication"],
  "direction": "->"
}
```

| 字段名         | 类型           | 含义                          |
| ----------- | ------------ | --------------------------- |
| `type`      | string       | 固定为 `relationship`          |
| `labels`    | list[string] | 关系类型（如 `:contraindication`） |
| `direction` | string       | 关系方向（当前简化为 `->`）            |

> 当前版本未区分 `<-`、`-`、`->`，但结构已预留扩展空间。

---

## RETURN 子句

```json
"return": ["p"]
```

* 每一项是 `projectionItem.getText()` 的结果
* 保留原始 Cypher 表达式形式

示例：

```cypher
RETURN DISTINCT p, a.name
```

可能解析为：

```json
"return": ["DISTINCT p", "a.name"]
```

---

## LIMIT 子句

```json
"limit": 20
```

* 如果存在 `LIMIT`，解析为整数
* 如果不存在，值为 `null`

---

## 示例

### 输入 Cypher

```cypher
MATCH (a {id: '1904862'})
MATCH (b {id: 'D014025'})
MATCH p = (a)-[:contraindication]->()-[:indication]->()-[:disease_phenotype_negative]->(b)
RETURN p LIMIT 20
```

### 输出 AST（简化）

```json
{
  "type": "query",
  "clauses": [ ... ],
  "return": ["p"],
  "limit": 20
}
```

---

## 设计取舍与局限

* ✅ 关注 **结构信息**，而非执行语义
* ✅ 非常适合 LLM / 图模式抽取
* ❌ 未完整支持 WHERE、WITH、ORDER BY
* ❌ 属性值、表达式未构建完整 AST
* ❌ 关系方向、长度暂未解析

---

## 未来可扩展方向

* WHERE → 条件表达式 AST
* WITH / ORDER BY / DISTINCT 支持
* 关系方向与可变长度路径
* Expression 子树（函数、比较、逻辑）
* AST → 图 Schema / Prompt 自动生成

---

## 总结一句话

> 这是一个 **面向结构理解而非执行** 的 Cypher 解析器，将复杂的图查询转换为清晰、稳定、可编程处理的 JSON AST。
