"""
只读查询模块实现
"""

from typing import Any, Dict, List
import networkx as nx
import random
import numpy as np
from .base import BaseQueryModule


class ReadQueryModule(BaseQueryModule):
    """只读查询模块 - 生成各种读取和分析查询"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化只读查询模块
        
        Args:
            config: 查询配置参数
        """
        super().__init__(config)
        self.num_queries = config.get('num_queries', 20)
        self.query_distribution = config.get('query_distribution', {
            'node_query': 0.3,
            'edge_query': 0.2,
            'path_query': 0.2,
            'subgraph_query': 0.15,
            'centrality_query': 0.1,
            'community_query': 0.05
        })
        
    def generate_queries(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成只读查询集合"""
        queries = []
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        
        if not nodes:
            return queries
        
        # 根据分布生成不同类型的查询
        for query_type, ratio in self.query_distribution.items():
            num_queries = int(self.num_queries * ratio)
            
            if query_type == 'node_query':
                queries.extend(self._generate_node_queries(graph, nodes, num_queries))
            elif query_type == 'edge_query':
                queries.extend(self._generate_edge_queries(graph, edges, num_queries))
            elif query_type == 'path_query':
                queries.extend(self._generate_path_queries(graph, nodes, num_queries))
            elif query_type == 'subgraph_query':
                queries.extend(self._generate_subgraph_queries(graph, nodes, num_queries))
            elif query_type == 'centrality_query':
                queries.extend(self._generate_centrality_queries(graph, num_queries))
            elif query_type == 'community_query':
                queries.extend(self._generate_community_queries(graph, num_queries))
        
        # 随机打乱查询顺序
        random.shuffle(queries)
        
        # 验证查询
        validated_queries = [q for q in queries if self.validate_query(q)]
        
        self._generated_queries = validated_queries
        return validated_queries
    
    def validate_query(self, query: Dict[str, Any]) -> bool:
        """验证查询的有效性"""
        required_fields = ['type', 'parameters']
        
        for field in required_fields:
            if field not in query:
                return False
        
        # 检查查询类型是否支持
        if query['type'] not in self.get_supported_query_types():
            return False
        
        return True
    
    def get_supported_query_types(self) -> List[str]:
        """获取支持的查询类型"""
        return [
            'node_attribute_query', 'node_degree_query', 'node_neighbors_query',
            'edge_attribute_query', 'edge_existence_query',
            'shortest_path_query', 'all_paths_query', 'path_length_query',
            'subgraph_extraction', 'subgraph_analysis', 'induced_subgraph',
            'betweenness_centrality', 'closeness_centrality', 'pagerank',
            'community_detection', 'modularity_calculation'
        ]
    
    def _generate_node_queries(self, graph: nx.Graph, nodes: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成节点相关查询"""
        queries = []
        
        for _ in range(num_queries):
            if not nodes:
                break
                
            node = random.choice(nodes)
            query_type = random.choice(['node_attribute_query', 'node_degree_query', 'node_neighbors_query'])
            
            if query_type == 'node_attribute_query':
                queries.append({
                    "type": "node_attribute_query",
                    "parameters": {"node": node},
                    "description": f"获取节点 {node} 的所有属性",
                    "complexity": "low",
                    "expected_result_type": "dict"
                })
            
            elif query_type == 'node_degree_query':
                queries.append({
                    "type": "node_degree_query",
                    "parameters": {"node": node},
                    "description": f"获取节点 {node} 的度数",
                    "complexity": "low",
                    "expected_result_type": "int"
                })
            
            elif query_type == 'node_neighbors_query':
                queries.append({
                    "type": "node_neighbors_query",
                    "parameters": {"node": node, "radius": random.randint(1, 3)},
                    "description": f"获取节点 {node} 的邻居节点",
                    "complexity": "low",
                    "expected_result_type": "list"
                })
        
        return queries
    
    def _generate_edge_queries(self, graph: nx.Graph, edges: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成边相关查询"""
        queries = []
        nodes = list(graph.nodes())
        
        for _ in range(num_queries):
            query_type = random.choice(['edge_attribute_query', 'edge_existence_query'])
            
            if query_type == 'edge_attribute_query' and edges:
                edge = random.choice(edges)
                queries.append({
                    "type": "edge_attribute_query",
                    "parameters": {"source": edge[0], "target": edge[1]},
                    "description": f"获取边 {edge[0]}-{edge[1]} 的属性",
                    "complexity": "low",
                    "expected_result_type": "dict"
                })
            
            elif query_type == 'edge_existence_query' and len(nodes) >= 2:
                source, target = random.sample(nodes, 2)
                queries.append({
                    "type": "edge_existence_query",
                    "parameters": {"source": source, "target": target},
                    "description": f"检查边 {source}-{target} 是否存在",
                    "complexity": "low",
                    "expected_result_type": "bool"
                })
        
        return queries
    
    def _generate_path_queries(self, graph: nx.Graph, nodes: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成路径相关查询"""
        queries = []
        
        for _ in range(num_queries):
            if len(nodes) < 2:
                break
                
            source, target = random.sample(nodes, 2)
            query_type = random.choice(['shortest_path_query', 'all_paths_query', 'path_length_query'])
            
            if query_type == 'shortest_path_query':
                queries.append({
                    "type": "shortest_path_query",
                    "parameters": {"source": source, "target": target},
                    "description": f"查找从 {source} 到 {target} 的最短路径",
                    "complexity": "medium",
                    "expected_result_type": "list"
                })
            
            elif query_type == 'all_paths_query':
                max_length = random.randint(3, 5)
                queries.append({
                    "type": "all_paths_query",
                    "parameters": {"source": source, "target": target, "max_length": max_length},
                    "description": f"查找从 {source} 到 {target} 的所有路径（最大长度 {max_length}）",
                    "complexity": "high",
                    "expected_result_type": "list"
                })
            
            elif query_type == 'path_length_query':
                queries.append({
                    "type": "path_length_query",
                    "parameters": {"source": source, "target": target},
                    "description": f"计算从 {source} 到 {target} 的最短路径长度",
                    "complexity": "medium",
                    "expected_result_type": "int"
                })
        
        return queries
    
    def _generate_subgraph_queries(self, graph: nx.Graph, nodes: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成子图相关查询"""
        queries = []
        
        for _ in range(num_queries):
            if not nodes:
                break
                
            query_type = random.choice(['subgraph_extraction', 'subgraph_analysis', 'induced_subgraph'])
            
            if query_type == 'subgraph_extraction':
                # 随机选择一些节点
                sample_size = min(random.randint(3, 10), len(nodes))
                selected_nodes = random.sample(nodes, sample_size)
                
                queries.append({
                    "type": "subgraph_extraction",
                    "parameters": {"nodes": selected_nodes},
                    "description": f"提取包含节点 {selected_nodes[:3]}... 的子图",
                    "complexity": "medium",
                    "expected_result_type": "graph"
                })
            
            elif query_type == 'subgraph_analysis':
                center_node = random.choice(nodes)
                radius = random.randint(1, 3)
                
                queries.append({
                    "type": "subgraph_analysis",
                    "parameters": {"center_node": center_node, "radius": radius},
                    "description": f"分析以 {center_node} 为中心半径为 {radius} 的子图",
                    "complexity": "medium",
                    "expected_result_type": "dict"
                })
            
            elif query_type == 'induced_subgraph':
                sample_size = min(random.randint(5, 15), len(nodes))
                selected_nodes = random.sample(nodes, sample_size)
                
                queries.append({
                    "type": "induced_subgraph",
                    "parameters": {"nodes": selected_nodes},
                    "description": f"生成由节点集合诱导的子图",
                    "complexity": "low",
                    "expected_result_type": "graph"
                })
        
        return queries
    
    def _generate_centrality_queries(self, graph: nx.Graph, num_queries: int) -> List[Dict[str, Any]]:
        """生成中心性相关查询"""
        queries = []
        centrality_types = ['betweenness_centrality', 'closeness_centrality', 'pagerank']
        
        for _ in range(num_queries):
            centrality_type = random.choice(centrality_types)
            
            if centrality_type == 'betweenness_centrality':
                queries.append({
                    "type": "betweenness_centrality",
                    "parameters": {"normalized": random.choice([True, False])},
                    "description": "计算所有节点的介数中心性",
                    "complexity": "high",
                    "expected_result_type": "dict"
                })
            
            elif centrality_type == 'closeness_centrality':
                queries.append({
                    "type": "closeness_centrality",
                    "parameters": {"normalized": random.choice([True, False])},
                    "description": "计算所有节点的接近中心性",
                    "complexity": "medium",
                    "expected_result_type": "dict"
                })
            
            elif centrality_type == 'pagerank':
                alpha = random.uniform(0.8, 0.9)
                queries.append({
                    "type": "pagerank",
                    "parameters": {"alpha": alpha, "max_iter": 100},
                    "description": f"计算PageRank值（alpha={alpha:.2f}）",
                    "complexity": "medium",
                    "expected_result_type": "dict"
                })
        
        return queries
    
    def _generate_community_queries(self, graph: nx.Graph, num_queries: int) -> List[Dict[str, Any]]:
        """生成社区检测相关查询"""
        queries = []
        
        for _ in range(num_queries):
            query_type = random.choice(['community_detection', 'modularity_calculation'])
            
            if query_type == 'community_detection':
                method = random.choice(['greedy_modularity', 'label_propagation'])
                queries.append({
                    "type": "community_detection",
                    "parameters": {"method": method},
                    "description": f"使用 {method} 方法进行社区检测",
                    "complexity": "high",
                    "expected_result_type": "list"
                })
            
            elif query_type == 'modularity_calculation':
                queries.append({
                    "type": "modularity_calculation",
                    "parameters": {},
                    "description": "计算图的模块度",
                    "complexity": "medium",
                    "expected_result_type": "float"
                })
        
        return queries
    
    def generate_benchmark_queries(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成基准测试查询集合"""
        benchmark_queries = []
        nodes = list(graph.nodes())
        
        if not nodes:
            return benchmark_queries
        
        # 基本查询
        if len(nodes) >= 2:
            source, target = random.sample(nodes, 2)
            benchmark_queries.extend([
                {
                    "type": "shortest_path_query",
                    "parameters": {"source": source, "target": target},
                    "description": "基准：最短路径查询",
                    "complexity": "medium",
                    "expected_result_type": "list",
                    "is_benchmark": True
                },
                {
                    "type": "node_neighbors_query",
                    "parameters": {"node": source, "radius": 2},
                    "description": "基准：邻居查询",
                    "complexity": "low",
                    "expected_result_type": "list",
                    "is_benchmark": True
                }
            ])
        
        # 中心性查询
        benchmark_queries.append({
            "type": "pagerank",
            "parameters": {"alpha": 0.85, "max_iter": 100},
            "description": "基准：PageRank计算",
            "complexity": "medium",
            "expected_result_type": "dict",
            "is_benchmark": True
        })
        
        # 社区检测
        benchmark_queries.append({
            "type": "community_detection",
            "parameters": {"method": "greedy_modularity"},
            "description": "基准：社区检测",
            "complexity": "high",
            "expected_result_type": "list",
            "is_benchmark": True
        })
        
        return benchmark_queries
