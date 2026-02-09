"""
NGDB Framework - 图算法基准测试框架
"""

__version__ = "0.1.0"
__author__ = "NGDB Team"

# 为了在部分组件缺失时仍然可以导入子模块（例如仅使用 handler 工具），
# 这里对各个组件的导入做了容错处理。这样即使某些模块不存在，也不会在
# `import ngdb_benchmark.pipeline.handler...` 时直接报错。
try:
    from .data_source import DataSource  # type: ignore
except Exception:  # pragma: no cover - 容错导入
    DataSource = None  # type: ignore

try:
    from .perturbation_generator import PerturbationGenerator  # type: ignore
except Exception:  # pragma: no cover
    PerturbationGenerator = None  # type: ignore

try:
    from .methodology import Methodology  # type: ignore
except Exception:  # pragma: no cover
    Methodology = None  # type: ignore

try:
    from .query_module import QueryModule  # type: ignore
except Exception:  # pragma: no cover
    QueryModule = None  # type: ignore

try:
    from .evaluation_harness import EvaluationHarness  # type: ignore
except Exception:  # pragma: no cover
    EvaluationHarness = None  # type: ignore

try:
    from .metric_report import MetricReport  # type: ignore
except Exception:  # pragma: no cover
    MetricReport = None  # type: ignore

__all__ = [
    "DataSource",
    "PerturbationGenerator",
    "Methodology",
    "QueryModule",
    "EvaluationHarness",
    "MetricReport",
]
