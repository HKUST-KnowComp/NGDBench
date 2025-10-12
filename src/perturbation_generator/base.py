"""
扰动生成器基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np
import copy


class BasePerturbationGenerator(ABC):
    """扰动生成器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化扰动生成器
        
        Args:
            config: 扰动配置参数
        """
        self.config = config
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
        
    @abstractmethod
    def apply_perturbation(self, graph: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        应用扰动到图数据
        
        Args:
            graph: 原始图数据
            
        Returns:
            扰动后的图数据和扰动信息
        """
        pass
    
    def create_groundtruth(self, graph: nx.Graph) -> nx.Graph:
        """
        创建真实基准数据（原始数据的副本）
        
        Args:
            graph: 原始图数据
            
        Returns:
            真实基准图数据
        """
        return copy.deepcopy(graph)
    
    def get_perturbation_stats(self, original: nx.Graph, perturbed: nx.Graph) -> Dict[str, Any]:
        """
        计算扰动统计信息
        
        Args:
            original: 原始图
            perturbed: 扰动后的图
            
        Returns:
            扰动统计信息
        """
        return {
            "original_nodes": original.number_of_nodes(),
            "original_edges": original.number_of_edges(),
            "perturbed_nodes": perturbed.number_of_nodes(),
            "perturbed_edges": perturbed.number_of_edges(),
            "nodes_removed": original.number_of_nodes() - perturbed.number_of_nodes(),
            "edges_removed": original.number_of_edges() - perturbed.number_of_edges(),
            "perturbation_ratio_nodes": 1 - (perturbed.number_of_nodes() / original.number_of_nodes()),
            "perturbation_ratio_edges": 1 - (perturbed.number_of_edges() / original.number_of_edges())
        }
    
    def validate_perturbation(self, original: nx.Graph, perturbed: nx.Graph) -> bool:
        """
        验证扰动结果的有效性
        
        Args:
            original: 原始图
            perturbed: 扰动后的图
            
        Returns:
            验证结果
        """
        # 基本验证：扰动后的图不应该比原图更大
        if (perturbed.number_of_nodes() > original.number_of_nodes() or 
            perturbed.number_of_edges() > original.number_of_edges()):
            return False
        
        # 扰动后的图应该仍然是有效的图结构
        if perturbed.number_of_nodes() == 0:
            return False
            
        return True
