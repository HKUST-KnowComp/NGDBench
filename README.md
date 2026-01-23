# NADB Benchmark Status
## User Guide
- 首先把数据转换为图的形式 .gpickle或者.graphml
cd data_gen/graph_gen
python run.py

- 模拟噪声图，记录噪声点的位置：
cd data_gen
python graph_generator.py

- 生成检测查询（噪声图上：噪声点和干净点上的复杂查询检测；干净图上：增删改相关的查询生成）：
  查询分为几类：
  complex1，complex2(判断题), management
cd pipeline/query_gen
python qgen_test_noise

- 清洗查询结果

- 得到nlp描述
cd pipeline/handler
python translate.py
(记得修改文件名)

## neo4j的使用
详见pipeline/query_module/db_base.py

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
    ```
目前已有几个数据库容器如下：
金融文档数据：
```bash
# 金融文档数据：
docker run \
  --name neo4j-520 \
  -p 7689:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0

# mcp数据：
docker run \
  --name neo4j-mcp \
  -p 7690:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0

# ldbcbi数据：
docker run \
  --name neo4j-ldbcbi \
  -p 7691:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0

# ldbcfin数据：
docker run \
  --name neo4j-ldbcfin \
  -p 7692:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0

# ldbcfin-noise 数据：
docker run \
  --name neo4j-ldbcfin-noise \
  -p 7693:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0

# ldbcfin-manage 数据：
docker run \
  --name neo4j-ldbcfin-manage \
  -p 7694:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
```
  
## Data Generation Module Status
- ✅ Random incompleteness generation is done.
- ✅ Random noise generation is done.
- ✅ Semantic perturbation is done. (But I only tested it on Primekg dataset)
- ✅ pipeline/data_analyser(containing the dataloader) is working
  - On how to use the dataloader, you can refer to the test code in file dataload_toolkit.py.
  - In the data_analyser module, there is a buffer directory which stores the ldbc_snb_bi_graph.gpickle, but it is gitignored due to the huge size. You can access it on the machine CPU8`/data/ylivm/ngdb_benchmark/pipeline/data_analyser/buffer` (but actually generating the gpickle file from scratch takes few minutes.)
- 🚧 Coming soon:
  - The next important step is to construct the query generation module.
  - topology_perturbation is not considered in current stage.

## Usage Guide
For detailed instructions on using the data_gen module, please see [data_gen/readme.md](data_gen/readme.md).

## Generated Datasets
The currently generated datasets are stored at:
GPU8 `/data/ylivm/ngdb_benchmark/data_gen/perturbed_dataset`, you can refer to `/data/ylivm/ngdb_benchmark/data_gen/perturb_record` for what have happened.

In the data_analyser module, there is a buffer directory which stores the ldbc_snb_bi_graph.gpickle, but it is gitignored due to the huge size. You can access it on the machine(but actually generating the gpickle file from scratch takes few minutes.)