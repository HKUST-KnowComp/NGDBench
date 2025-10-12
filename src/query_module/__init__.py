"""
查询模块 - 提供标准化的查询集合来测试算法性能
"""

from .base import BaseQueryModule
from .read_queries import ReadQueryModule
from .update_queries import UpdateQueryModule
from .query_generator import QueryGenerator

__all__ = [
    "BaseQueryModule",
    "ReadQueryModule",
    "UpdateQueryModule",
    "QueryGenerator"
]
