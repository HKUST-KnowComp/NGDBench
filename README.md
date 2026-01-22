# NADB Benchmark Status
## User Guide
首先把数据转换为图的形式 .gpickle或者.graphml
cd data_gen/graph_gen
python run.py

模拟噪声图，记录噪声点的位置：
cd data_gen
python graph_generator.py

生成检测查询（噪声图上：噪声点和干净点上的复杂查询检测；干净图上：增删改相关的查询生成）：
查询分为几类：
complex1，complex2(判断题), management

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