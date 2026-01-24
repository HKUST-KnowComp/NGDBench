# NGDB Benchmark

## 目录

- [用户指南](#用户指南)
- [Neo4j 使用](#neo4j-使用)
- [数据生成模块状态](#数据生成模块状态)
- [使用指南](#使用指南)
- [生成的数据集](#生成的数据集)

## 用户指南

### 1. 数据转换为图格式

将数据转换为图的形式（`.gpickle` 或 `.graphml`）：

```bash
cd data_gen/graph_gen
python run.py
```

### 2. 模拟噪声图

生成噪声图并记录噪声点的位置：

```bash
cd data_gen
python graph_generator.py
```

### 3. 构建数据库容器

如果需要自己建数据库容器（可以先用已建好的）：

```bash
cd pipeline/db_builder
python test_build.py
```

### 4. 生成检测查询

在噪声图上生成检测查询（噪声点和干净点上的复杂查询检测），在干净图上生成增删改相关的查询。

查询分为几类：
- **complex1**: 复杂查询类型1
- **complex2**: 复杂查询类型2（判断题）
- **management**: 管理查询（增删改）

```bash
cd pipeline/query_gen
python qgen_test_noise
python management_test.py
```

### 5. 清洗查询结果

清洗查询结果数据。

### 6. 生成 NLP 描述

```bash
cd pipeline/handler
python translate.py
```

**注意**: 记得修改文件名。

## Neo4j 使用

### 基本用法

详见 `pipeline/query_module/db_base.py`

```python
uri = "bolt://localhost:7693"
user = "neo4j"
password = "fei123456"

# 输入和输出文件路径
input_json_file = "../query_gen/query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned.json"
output_json_file = "noise_execution_step1_ldbcfin_results.json"

# 创建数据库执行器
executor = DatabaseExecutor(uri, user, password)

try:
    # 连接数据库
    executor.connect()
    
    # 读取查询
    queries = executor.read_queries_from_json(input_json_file)
    
    # 执行查询并比较结果，启用增量保存（一边执行一边记录）
    results = executor.execute_queries_batch(
        queries, 
        compare_with_original=True,
        incremental_save=True,  # 启用增量保存
        output_file_path=output_json_file
    )
finally:
    executor.close()
```

### Docker 容器配置

目前已有的数据库容器如下：

#### 金融文档数据

```bash
docker run -d \
  --name neo4j-520 \
  -p 7689:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```

#### MCP 数据

```bash
docker run -d \
  --name neo4j-mcp \
  -p 7690:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```

#### LDBC BI 数据

```bash
docker run -d \
  --name neo4j-ldbcbi \
  -p 7691:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```

#### LDBC Fin 数据

```bash
docker run -d \
  --name neo4j-ldbcfin \
  -p 7692:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```

#### LDBC Fin Noise 数据

```bash
docker run -d \
  --name neo4j-ldbcfin-noise \
  -p 7693:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```

#### LDBC Fin Manage 数据

```bash
docker run -d \
  --name neo4j-ldbcfin-manage \
  -p 7694:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```

## 数据生成模块状态

### 已完成功能

- ✅ 随机不完整性生成已完成
- ✅ 随机噪声生成已完成
- ✅ 语义扰动已完成（仅在 PrimeKG 数据集上测试过）
- ✅ `pipeline/data_analyser`（包含数据加载器）正在工作
  - 关于如何使用数据加载器，可以参考 `dataload_toolkit.py` 文件中的测试代码
  - 在 `data_analyser` 模块中，有一个 `buffer` 目录存储 `ldbc_snb_bi_graph.gpickle`，但由于文件过大已被 gitignore
  - 可以在 CPU8 机器上访问：`/data/ylivm/ngdb_benchmark/pipeline/data_analyser/buffer`
  - 实际上从头生成 gpickle 文件只需要几分钟

### 待完成功能

- 🚧 查询生成模块（下一步重要工作）
- 🚧 拓扑扰动（当前阶段暂不考虑）

## 使用指南

关于如何使用 `data_gen` 模块的详细说明，请参阅 [data_gen/readme.md](data_gen/readme.md)。

## 生成的数据集

当前生成的数据集存储在：

- **GPU8**: `/data/ylivm/ngdb_benchmark/data_gen/perturbed_dataset`
- **扰动记录**: `/data/ylivm/ngdb_benchmark/data_gen/perturb_record`

在 `data_analyser` 模块中，有一个 `buffer` 目录存储 `ldbc_snb_bi_graph.gpickle`，但由于文件过大已被 gitignore。可以在机器上访问（实际上从头生成 gpickle 文件只需要几分钟）。
