"""
图算法执行器实现
"""

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from .base import BaseMethodology


class GraphAlgorithmMethodology(BaseMethodology):
    """传统图算法执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化图算法执行器
        
        Args:
            config: 算法配置，包含algorithm_type等参数
        """
        super().__init__(config)
        self.algorithm_type = config.get('algorithm_type', 'pagerank')
        
        # 支持的算法类型
        self.supported_algorithms = {
            'pagerank', 'betweenness_centrality', 'closeness_centrality',
            'eigenvector_centrality', 'clustering', 'shortest_path',
            'connected_components', 'community_detection', 'node_classification'
        }
        
        if self.algorithm_type not in self.supported_algorithms:
            raise ValueError(f"不支持的算法类型: {self.algorithm_type}")
    
    def execute(self, graph: nx.Graph, queries: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """执行图算法"""
        if not self.validate_input(graph):
            raise ValueError("输入图数据验证失败")
        
        # 预处理图数据
        processed_graph = self.preprocess_graph(graph)
        
        # 执行主算法
        algorithm_results = self._execute_algorithm(processed_graph)
        
        # 处理查询
        query_results = {}
        if queries:
            query_results = self._process_queries(processed_graph, queries, algorithm_results)
        
        # 组合结果
        results = {
            "algorithm_results": algorithm_results,
            "query_results": query_results,
            "graph_stats": self._get_graph_statistics(processed_graph)
        }
        
        return self.postprocess_results(results)
    
    def validate_input(self, graph: nx.Graph) -> bool:
        """验证输入图数据"""
        if graph is None or graph.number_of_nodes() == 0:
            return False
        
        # 检查算法特定的要求
        if self.algorithm_type in ['eigenvector_centrality'] and not nx.is_connected(graph):
            # 特征向量中心性需要连通图
            return False
        
        return True
    
    def get_supported_query_types(self) -> List[str]:
        """获取支持的查询类型"""
        return [
            'node_ranking', 'top_k_nodes', 'node_score', 'subgraph_analysis',
            'path_query', 'neighborhood_query', 'similarity_query'
        ]
    
    def _execute_algorithm(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行具体的图算法"""
        if self.algorithm_type == 'pagerank':
            return self._execute_pagerank(graph)
        elif self.algorithm_type == 'betweenness_centrality':
            return self._execute_betweenness_centrality(graph)
        elif self.algorithm_type == 'closeness_centrality':
            return self._execute_closeness_centrality(graph)
        elif self.algorithm_type == 'eigenvector_centrality':
            return self._execute_eigenvector_centrality(graph)
        elif self.algorithm_type == 'clustering':
            return self._execute_clustering(graph)
        elif self.algorithm_type == 'shortest_path':
            return self._execute_shortest_path(graph)
        elif self.algorithm_type == 'connected_components':
            return self._execute_connected_components(graph)
        elif self.algorithm_type == 'community_detection':
            return self._execute_community_detection(graph)
        elif self.algorithm_type == 'node_classification':
            return self._execute_node_classification(graph)
        else:
            raise ValueError(f"未实现的算法: {self.algorithm_type}")
    
    def _execute_pagerank(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行PageRank算法"""
        alpha = self.algorithm_params.get('alpha', 0.85)
        max_iter = self.algorithm_params.get('max_iter', 100)
        tol = self.algorithm_params.get('tol', 1e-6)
        
        pagerank_scores = nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol)
        
        return {
            "type": "pagerank",
            "scores": pagerank_scores,
            "top_nodes": sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            "parameters": {"alpha": alpha, "max_iter": max_iter, "tol": tol}
        }
    
    def _execute_betweenness_centrality(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行介数中心性算法"""
        normalized = self.algorithm_params.get('normalized', True)
        k = self.algorithm_params.get('k', None)  # 采样节点数
        
        centrality_scores = nx.betweenness_centrality(graph, normalized=normalized, k=k)
        
        return {
            "type": "betweenness_centrality",
            "scores": centrality_scores,
            "top_nodes": sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            "parameters": {"normalized": normalized, "k": k}
        }
    
    def _execute_closeness_centrality(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行接近中心性算法"""
        normalized = self.algorithm_params.get('normalized', True)
        
        centrality_scores = nx.closeness_centrality(graph, normalized=normalized)
        
        return {
            "type": "closeness_centrality",
            "scores": centrality_scores,
            "top_nodes": sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            "parameters": {"normalized": normalized}
        }
    
    def _execute_eigenvector_centrality(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行特征向量中心性算法"""
        max_iter = self.algorithm_params.get('max_iter', 100)
        tol = self.algorithm_params.get('tol', 1e-6)
        
        try:
            centrality_scores = nx.eigenvector_centrality(graph, max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            # 如果收敛失败，使用度中心性作为替代
            centrality_scores = nx.degree_centrality(graph)
        
        return {
            "type": "eigenvector_centrality",
            "scores": centrality_scores,
            "top_nodes": sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            "parameters": {"max_iter": max_iter, "tol": tol}
        }
    
    def _execute_clustering(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行聚类系数计算"""
        clustering_coeffs = nx.clustering(graph)
        avg_clustering = nx.average_clustering(graph)
        
        return {
            "type": "clustering",
            "node_clustering": clustering_coeffs,
            "average_clustering": avg_clustering,
            "top_clustered_nodes": sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _execute_shortest_path(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行最短路径算法"""
        source = self.algorithm_params.get('source')
        target = self.algorithm_params.get('target')
        
        if source and target:
            # 计算特定源目标的最短路径
            try:
                path = nx.shortest_path(graph, source, target)
                length = nx.shortest_path_length(graph, source, target)
                return {
                    "type": "shortest_path",
                    "source": source,
                    "target": target,
                    "path": path,
                    "length": length
                }
            except nx.NetworkXNoPath:
                return {
                    "type": "shortest_path",
                    "source": source,
                    "target": target,
                    "path": None,
                    "length": float('inf')
                }
        else:
            # 计算所有节点对的最短路径长度
            try:
                path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
                avg_path_length = nx.average_shortest_path_length(graph)
                return {
                    "type": "all_pairs_shortest_path",
                    "path_lengths": path_lengths,
                    "average_path_length": avg_path_length
                }
            except nx.NetworkXError:
                # 图不连通
                return {
                    "type": "all_pairs_shortest_path",
                    "path_lengths": {},
                    "average_path_length": float('inf'),
                    "error": "Graph is not connected"
                }
    
    def _execute_connected_components(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行连通分量分析"""
        if graph.is_directed():
            components = list(nx.strongly_connected_components(graph))
            weak_components = list(nx.weakly_connected_components(graph))
            return {
                "type": "connected_components",
                "strongly_connected_components": [list(comp) for comp in components],
                "weakly_connected_components": [list(comp) for comp in weak_components],
                "num_strongly_connected": len(components),
                "num_weakly_connected": len(weak_components)
            }
        else:
            components = list(nx.connected_components(graph))
            return {
                "type": "connected_components",
                "connected_components": [list(comp) for comp in components],
                "num_components": len(components),
                "largest_component_size": len(max(components, key=len)) if components else 0
            }
    
    def _execute_community_detection(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行社区检测"""
        method = self.algorithm_params.get('method', 'greedy_modularity')
        
        if method == 'greedy_modularity':
            communities = list(nx.community.greedy_modularity_communities(graph))
        elif method == 'label_propagation':
            communities = list(nx.community.label_propagation_communities(graph))
        else:
            communities = list(nx.community.greedy_modularity_communities(graph))
        
        modularity = nx.community.modularity(graph, communities)
        
        return {
            "type": "community_detection",
            "communities": [list(comm) for comm in communities],
            "num_communities": len(communities),
            "modularity": modularity,
            "method": method
        }
    
    def _execute_node_classification(self, graph: nx.Graph) -> Dict[str, Any]:
        """执行节点分类（基于图结构特征）"""
        # 计算节点特征
        features = {}
        
        # 度特征
        degrees = dict(graph.degree())
        
        # 中心性特征
        try:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            clustering = nx.clustering(graph)
        except:
            betweenness = {node: 0 for node in graph.nodes()}
            closeness = {node: 0 for node in graph.nodes()}
            clustering = {node: 0 for node in graph.nodes()}
        
        # 组合特征
        for node in graph.nodes():
            features[node] = {
                "degree": degrees.get(node, 0),
                "betweenness": betweenness.get(node, 0),
                "closeness": closeness.get(node, 0),
                "clustering": clustering.get(node, 0)
            }
        
        # 简单的基于度的分类
        classifications = {}
        degree_threshold_low = np.percentile(list(degrees.values()), 33)
        degree_threshold_high = np.percentile(list(degrees.values()), 67)
        
        for node, degree in degrees.items():
            if degree <= degree_threshold_low:
                classifications[node] = "low_degree"
            elif degree >= degree_threshold_high:
                classifications[node] = "high_degree"
            else:
                classifications[node] = "medium_degree"
        
        return {
            "type": "node_classification",
            "features": features,
            "classifications": classifications,
            "thresholds": {
                "low": degree_threshold_low,
                "high": degree_threshold_high
            }
        }
    
    def _process_queries(self, graph: nx.Graph, queries: List[Dict], 
                        algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询请求"""
        query_results = {}
        
        for i, query in enumerate(queries):
            if not self.is_query_supported(query):
                query_results[f"query_{i}"] = {"error": f"不支持的查询类型: {query.get('type')}"}
                continue
            
            try:
                result = self._execute_single_query(graph, query, algorithm_results)
                query_results[f"query_{i}"] = result
            except Exception as e:
                query_results[f"query_{i}"] = {"error": str(e)}
        
        return query_results
    
    def _execute_single_query(self, graph: nx.Graph, query: Dict[str, Any], 
                             algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个查询"""
        query_type = query['type']
        
        if query_type == 'node_ranking':
            return self._query_node_ranking(algorithm_results, query)
        elif query_type == 'top_k_nodes':
            return self._query_top_k_nodes(algorithm_results, query)
        elif query_type == 'node_score':
            return self._query_node_score(algorithm_results, query)
        elif query_type == 'subgraph_analysis':
            return self._query_subgraph_analysis(graph, query)
        elif query_type == 'path_query':
            return self._query_path(graph, query)
        elif query_type == 'neighborhood_query':
            return self._query_neighborhood(graph, query)
        elif query_type == 'similarity_query':
            return self._query_similarity(graph, algorithm_results, query)
        else:
            raise ValueError(f"未实现的查询类型: {query_type}")
    
    def _query_node_ranking(self, algorithm_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """节点排名查询"""
        if 'scores' in algorithm_results:
            scores = algorithm_results['scores']
            ranked_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return {
                "type": "node_ranking",
                "ranked_nodes": ranked_nodes,
                "total_nodes": len(ranked_nodes)
            }
        else:
            return {"error": "算法结果中没有分数信息"}
    
    def _query_top_k_nodes(self, algorithm_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """Top-K节点查询"""
        k = query.get('k', 10)
        if 'scores' in algorithm_results:
            scores = algorithm_results['scores']
            top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            return {
                "type": "top_k_nodes",
                "k": k,
                "top_nodes": top_k
            }
        else:
            return {"error": "算法结果中没有分数信息"}
    
    def _query_node_score(self, algorithm_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """节点分数查询"""
        node = query.get('node')
        if not node:
            return {"error": "查询中缺少节点参数"}
        
        if 'scores' in algorithm_results:
            scores = algorithm_results['scores']
            score = scores.get(node, None)
            return {
                "type": "node_score",
                "node": node,
                "score": score
            }
        else:
            return {"error": "算法结果中没有分数信息"}
    
    def _query_subgraph_analysis(self, graph: nx.Graph, query: Dict[str, Any]) -> Dict[str, Any]:
        """子图分析查询"""
        nodes = query.get('nodes', [])
        if not nodes:
            return {"error": "查询中缺少节点列表"}
        
        subgraph = graph.subgraph(nodes)
        return {
            "type": "subgraph_analysis",
            "nodes": nodes,
            "subgraph_nodes": subgraph.number_of_nodes(),
            "subgraph_edges": subgraph.number_of_edges(),
            "density": nx.density(subgraph),
            "is_connected": nx.is_connected(subgraph) if not subgraph.is_directed() else nx.is_weakly_connected(subgraph)
        }
    
    def _query_path(self, graph: nx.Graph, query: Dict[str, Any]) -> Dict[str, Any]:
        """路径查询"""
        source = query.get('source')
        target = query.get('target')
        
        if not source or not target:
            return {"error": "查询中缺少源节点或目标节点"}
        
        try:
            path = nx.shortest_path(graph, source, target)
            length = len(path) - 1
            return {
                "type": "path_query",
                "source": source,
                "target": target,
                "path": path,
                "length": length
            }
        except nx.NetworkXNoPath:
            return {
                "type": "path_query",
                "source": source,
                "target": target,
                "path": None,
                "length": float('inf'),
                "error": "No path exists"
            }
    
    def _query_neighborhood(self, graph: nx.Graph, query: Dict[str, Any]) -> Dict[str, Any]:
        """邻域查询"""
        node = query.get('node')
        radius = query.get('radius', 1)
        
        if not node:
            return {"error": "查询中缺少节点参数"}
        
        if node not in graph.nodes():
            return {"error": f"节点 {node} 不存在于图中"}
        
        # 获取指定半径内的邻域
        neighborhood = set([node])
        current_level = set([node])
        
        for _ in range(radius):
            next_level = set()
            for n in current_level:
                next_level.update(graph.neighbors(n))
            neighborhood.update(next_level)
            current_level = next_level
        
        subgraph = graph.subgraph(neighborhood)
        
        return {
            "type": "neighborhood_query",
            "center_node": node,
            "radius": radius,
            "neighborhood_nodes": list(neighborhood),
            "neighborhood_size": len(neighborhood),
            "subgraph_edges": subgraph.number_of_edges()
        }
    
    def _query_similarity(self, graph: nx.Graph, algorithm_results: Dict[str, Any], 
                         query: Dict[str, Any]) -> Dict[str, Any]:
        """相似性查询"""
        node = query.get('node')
        top_k = query.get('k', 5)
        
        if not node:
            return {"error": "查询中缺少节点参数"}
        
        if 'scores' in algorithm_results:
            scores = algorithm_results['scores']
            if node not in scores:
                return {"error": f"节点 {node} 不在算法结果中"}
            
            target_score = scores[node]
            
            # 计算与其他节点的相似性（基于分数差异）
            similarities = []
            for other_node, other_score in scores.items():
                if other_node != node:
                    similarity = 1.0 / (1.0 + abs(target_score - other_score))
                    similarities.append((other_node, similarity))
            
            # 排序并取前k个
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:top_k]
            
            return {
                "type": "similarity_query",
                "target_node": node,
                "target_score": target_score,
                "similar_nodes": top_similar,
                "k": top_k
            }
        else:
            return {"error": "算法结果中没有分数信息"}
    
    def _get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """获取图统计信息"""
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed()
        }
        
        try:
            if not graph.is_directed():
                stats["is_connected"] = nx.is_connected(graph)
                if nx.is_connected(graph):
                    stats["diameter"] = nx.diameter(graph)
                    stats["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
            else:
                stats["is_strongly_connected"] = nx.is_strongly_connected(graph)
                stats["is_weakly_connected"] = nx.is_weakly_connected(graph)
        except:
            pass
        
        return stats
