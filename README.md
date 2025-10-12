# NGDB Framework - 图算法基准测试框架

NGDB (Noisy Graph Database) Framework 是一个专门用于评估图算法在不完整和噪声数据环境下性能的综合基准测试框架。

## 功能特性

### 核心功能
- **多样化数据源支持**: 支持文件加载和图生成器
- **系统性数据扰动**: 随机、语义和拓扑三种扰动策略
- **多种算法支持**: 传统图算法、GraphRAG和图神经网络
- **标准化查询集**: 读查询和更新查询的标准化测试集
- **全面评估体系**: 准确性、鲁棒性和性能三维评估
- **结构化报告**: 不完整性、噪声、更新和综合报告

### 框架架构

```
NGDB Framework
├── 数据准备和扰动
│   ├── 数据源 (Data Source)
│   └── 扰动生成器 (Perturbation Generator)
├── 核心评估和算法模块
│   ├── 算法执行 (Methodology)
│   └── 查询模块 (Query Module)
└── 评估和报告
    ├── 评估框架 (Evaluation Harness)
    └── 指标报告 (Metric Report)
```

## 安装

### 环境要求
- Python 3.8+
- NetworkX 2.6+
- NumPy 1.21+
- Pandas 1.3+
- PyYAML 5.4+

### 安装步骤

1. 克隆仓库:
```bash
git clone https://github.com/your-org/ngdb-framework.git
cd ngdb-framework
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 安装框架:
```bash
pip install -e .
```

## 快速开始

### 基本使用

```python
from ngdb_framework import NGDBFramework

# 定义配置
config = {
    "data_source": {
        "type": "generator",
        "generator_type": "karate_club"
    },
    "perturbation": {
        "type": "random",
        "node_removal_ratio": 0.1,
        "edge_removal_ratio": 0.15
    },
    "methodology": {
        "type": "graph_algorithm",
        "algorithm_type": "pagerank"
    },
    "evaluation": {
        "accuracy": {"enabled": True},
        "robustness": {"enabled": True},
        "performance": {"enabled": True}
    }
}

# 创建框架实例并运行
framework = NGDBFramework(config)
results = framework.run_benchmark()

print(f"总体性能评分: {results['reports']['comprehensive_report']['executive_summary']['overall_performance']['score']:.3f}")
```

### 使用配置文件

```bash
# 使用默认配置
python -m ngdb_framework --config configs/default_config.yaml --output results.json

# 使用自定义配置
python -m ngdb_framework --config my_config.yaml --output my_results.json --verbose
```

## 配置说明

### 数据源配置

```yaml
data_source:
  type: "generator"  # "file" 或 "generator"
  
  # 生成器配置
  generator_type: "barabasi_albert"
  generator_params:
    n: 100
    m: 3
    seed: 42
  
  # 文件配置
  file_path: "data/graph.gml"
  file_format: "gml"
```

### 扰动配置

```yaml
perturbation:
  type: "random"  # "random", "semantic", "topology"
  
  # 随机扰动
  remove_nodes: true
  node_removal_ratio: 0.1
  remove_edges: true
  edge_removal_ratio: 0.15
  add_noise: true
  noise_ratio: 0.05
```

### 算法配置

```yaml
methodology:
  type: "graph_algorithm"  # "graph_algorithm", "graph_rag", "gnn"
  algorithm_type: "pagerank"
  algorithm_params:
    alpha: 0.85
    max_iter: 100
```

### 评估配置

```yaml
evaluation:
  accuracy:
    enabled: true
    tolerance: 1e-6
  
  robustness:
    enabled: true
    robustness_threshold: 0.8
  
  performance:
    enabled: true
```

## 支持的算法

### 传统图算法
- PageRank
- 介数中心性 (Betweenness Centrality)
- 接近中心性 (Closeness Centrality)
- 特征向量中心性 (Eigenvector Centrality)
- 聚类系数 (Clustering Coefficient)
- 最短路径 (Shortest Path)
- 连通分量 (Connected Components)
- 社区检测 (Community Detection)

### GraphRAG算法
- 实体检索 (Entity Retrieval)
- 关系查询 (Relation Query)
- 路径推理 (Path Reasoning)
- 子图问答 (Subgraph QA)
- 相似性搜索 (Similarity Search)
- 知识补全 (Knowledge Completion)

### 图神经网络
- 图卷积网络 (GCN)
- 图注意力网络 (GAT)
- GraphSAGE
- 图同构网络 (GIN)

## 扰动策略

### 随机扰动
- 随机删除节点和边
- 随机修改属性值
- 添加随机噪声数据

### 语义扰动
- 基于语义规则删除数据
- 添加语义不一致的噪声
- 模拟现实世界的数据质量问题

### 拓扑扰动
- 基于度数和中心性删除关键节点
- 基于社区结构的扰动
- 破坏图的拓扑特性

## 评估指标

### 准确性指标
- 精确匹配率 (Exact Match Rate)
- 数值准确性 (Numerical Accuracy)
- 排名相关性 (Rank Correlation)
- 分类准确性 (Classification Accuracy)

### 鲁棒性指标
- 输出一致性 (Output Consistency)
- 结构保持性 (Structure Preservation)
- 性能退化率 (Performance Degradation)
- 稳定性评分 (Stability Score)

### 性能指标
- 执行时间 (Execution Time)
- 内存使用 (Memory Usage)
- 吞吐量 (Throughput)
- 可扩展性 (Scalability)

## 报告类型

### 综合报告
包含所有评估维度的完整分析，提供执行摘要、详细结果、比较分析和改进建议。

### 不完整性报告
专注于算法处理缺失数据的能力评估，分析节点/边删除的影响和信息恢复能力。

### 噪声报告
专注于算法对噪声数据的鲁棒性评估，分析不同噪声类型的影响和过滤效果。

### 更新报告
专注于算法处理动态更新的性能评估，分析更新操作的效率和一致性维护。

## 示例

### 基本示例
```bash
cd examples
python basic_usage.py
```

### 配置文件示例
```bash
# 默认配置
python -m ngdb_framework -c configs/default_config.yaml

# 文件数据源配置
python -m ngdb_framework -c configs/example_file_config.yaml

# GNN算法配置
python -m ngdb_framework -c configs/gnn_config.yaml
```

## 扩展开发

### 添加新的数据源
```python
from ngdb_framework.data_source import BaseDataSource

class MyDataSource(BaseDataSource):
    def load_data(self):
        # 实现数据加载逻辑
        pass
    
    def get_metadata(self):
        # 返回数据元信息
        pass
```

### 添加新的算法
```python
from ngdb_framework.methodology import BaseMethodology

class MyAlgorithm(BaseMethodology):
    def execute(self, graph, queries=None):
        # 实现算法执行逻辑
        pass
    
    def validate_input(self, graph):
        # 验证输入数据
        pass
```

### 添加新的评估器
```python
from ngdb_framework.evaluation_harness import BaseEvaluationHarness

class MyEvaluator(BaseEvaluationHarness):
    def evaluate(self, ground_truth, algorithm_output):
        # 实现评估逻辑
        pass
```

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 引用

如果您在研究中使用了 NGDB Framework，请引用：

```bibtex
@software{ngdb_framework,
  title={NGDB Framework: A Comprehensive Benchmark for Graph Algorithms on Noisy and Incomplete Data},
  author={NGDB Team},
  year={2024},
  url={https://github.com/your-org/ngdb-framework}
}
```

## 联系方式

- 项目主页: https://github.com/your-org/ngdb-framework
- 问题报告: https://github.com/your-org/ngdb-framework/issues
- 邮箱: ngdb-team@example.com

## 更新日志

### v0.1.0 (2024-01-01)
- 初始版本发布
- 支持基本的图算法评估
- 实现三种扰动策略
- 提供综合评估报告
