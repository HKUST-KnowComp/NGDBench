"""
算法执行模块 - 执行用户提供的图算法或模型
"""

from .base import BaseMethodology
from .graph_algorithm import GraphAlgorithmMethodology
from .graph_rag import GraphRAGMethodology
from .gnn_methodology import GNNMethodology

__all__ = [
    "BaseMethodology",
    "GraphAlgorithmMethodology",
    "GraphRAGMethodology", 
    "GNNMethodology"
]
