"""
生成器数据源实现
"""

from typing import Any, Dict, Optional
import networkx as nx
import numpy as np
from .base import BaseDataSource


class GeneratorDataSource(BaseDataSource):
    """使用图生成算法创建数据的数据源"""
    
    def __init__(self, generator_type: str, **params):
        """
        初始化生成器数据源
        
        Args:
            generator_type: 生成器类型 ('erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'karate_club')
            **params: 生成器参数
        """
        super().__init__(params)
        self.generator_type = generator_type
        self.params = params
        
        # 验证生成器类型
        supported_generators = {
            'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 
            'karate_club', 'complete', 'cycle', 'path'
        }
        if generator_type not in supported_generators:
            raise ValueError(f"不支持的生成器类型: {generator_type}")
    
    def load_data(self) -> nx.Graph:
        """使用生成器创建图数据"""
        if self.generator_type == "erdos_renyi":
            n = self.params.get('n', 100)
            p = self.params.get('p', 0.1)
            graph = nx.erdos_renyi_graph(n, p)
            
        elif self.generator_type == "barabasi_albert":
            n = self.params.get('n', 100)
            m = self.params.get('m', 3)
            graph = nx.barabasi_albert_graph(n, m)
            
        elif self.generator_type == "watts_strogatz":
            n = self.params.get('n', 100)
            k = self.params.get('k', 6)
            p = self.params.get('p', 0.1)
            graph = nx.watts_strogatz_graph(n, k, p)
            
        elif self.generator_type == "karate_club":
            graph = nx.karate_club_graph()
            
        elif self.generator_type == "complete":
            n = self.params.get('n', 10)
            graph = nx.complete_graph(n)
            
        elif self.generator_type == "cycle":
            n = self.params.get('n', 10)
            graph = nx.cycle_graph(n)
            
        elif self.generator_type == "path":
            n = self.params.get('n', 10)
            graph = nx.path_graph(n)
            
        else:
            raise ValueError(f"未实现的生成器: {self.generator_type}")
        
        # 添加节点和边属性
        self._add_attributes(graph)
        
        if not self.validate_data(graph):
            raise ValueError("生成的数据验证失败")
        
        return graph
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取生成器数据集元信息"""
        graph = self.graph
        return {
            "generator_type": self.generator_type,
            "generator_params": self.params,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "is_directed": graph.is_directed(),
            "density": nx.density(graph),
            "is_connected": nx.is_connected(graph) if not graph.is_directed() else nx.is_weakly_connected(graph)
        }
    
    def _add_attributes(self, graph: nx.Graph):
        """为节点和边添加随机属性"""
        np.random.seed(self.params.get('seed', 42))
        
        # 添加节点属性
        for node in graph.nodes():
            graph.nodes[node]['value'] = np.random.randint(1, 100)
            graph.nodes[node]['category'] = np.random.choice(['A', 'B', 'C'])
            graph.nodes[node]['weight'] = np.random.uniform(0.1, 1.0)
        
        # 添加边属性
        for edge in graph.edges():
            graph.edges[edge]['weight'] = np.random.uniform(0.1, 1.0)
            graph.edges[edge]['type'] = np.random.choice(['friend', 'colleague', 'family'])
