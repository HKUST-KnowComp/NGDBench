# NGDB Framework - 图算法基准测试框架

NGDB (Noisy Graph Database) Framework 是一个专门用于评估图算法在不完整和噪声数据环境下性能的综合基准测试框架。

## 代码架构说明

data_gen专门负责和gnd数据库不同的扰动数据的生成，从原本src里面解耦出来。所以src里面ngdb_framework目前不能要，得重构
### data_gen说明


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


