"""
拓扑扰动生成器实现
"""

from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np
import random
from .base import BasePerturbationGenerator


class TopologyPerturbationGenerator(BasePerturbationGenerator):
    """拓扑扰动生成器 - 基于图的拓扑结构进行扰动"""
    
    def apply_perturbation(self, graph: nx.Graph, perturb_type) -> Tuple[nx.Graph, Dict[str, Any]]:
        """应用拓扑扰动"""
        perturbed_graph = graph.copy()
        perturbation_info = {
            "type": "topology",
            "operations": []
        }
        if perturb_type == 'incompleteness':
        # 基于度数的节点删除
            if self.config.get('degree_based_removal', False):
                perturbed_graph, degree_ops = self._degree_based_removal(perturbed_graph)
                perturbation_info["operations"].extend(degree_ops)
            
            # 基于中心性的节点删除
            if self.config.get('centrality_based_removal', False):
                perturbed_graph, centrality_ops = self._centrality_based_removal(perturbed_graph)
                perturbation_info["operations"].extend(centrality_ops)
            
            # 基于社区结构的扰动
            if self.config.get('community_based_perturbation', False):
                perturbed_graph, community_ops = self._community_based_perturbation(perturbed_graph)
                perturbation_info["operations"].extend(community_ops)
        elif perturb_type == 'noise':
        # 添加拓扑噪声
            if self.config.get('topology_noise', False):
                perturbed_graph, noise_ops = self._add_topology_noise(perturbed_graph)
                perturbation_info["operations"].extend(noise_ops)
        else:
            raise ValueError(f"不支持的扰动类型: {perturb_type}")
        
        if not self.validate_perturbation(graph, perturbed_graph):
            raise ValueError("拓扑扰动验证失败")
        
        return perturbed_graph, perturbation_info
    
    def _degree_based_removal(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """基于节点度数删除节点"""
        strategy = self.config.get('degree_strategy', 'high_degree')  # 'high_degree', 'low_degree', 'random_degree'
        removal_ratio = self.config.get('degree_removal_ratio', 0.1)
        
        # 计算所有节点的度数
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        num_to_remove = int(len(sorted_nodes) * removal_ratio)
        operations = []
        
        if strategy == 'high_degree':
            # 删除高度数节点（关键节点）
            nodes_to_remove = [node for node, _ in sorted_nodes[:num_to_remove]]
        elif strategy == 'low_degree':
            # 删除低度数节点（边缘节点）
            nodes_to_remove = [node for node, _ in sorted_nodes[-num_to_remove:]]
        else:  # random_degree
            # 随机删除不同度数的节点
            nodes_to_remove = random.sample([node for node, _ in sorted_nodes], num_to_remove)
        
        for node in nodes_to_remove:
            if node in graph.nodes():
                operations.append({
                    "operation": "degree_based_remove_node",
                    "target": node,
                    "degree": degrees[node],
                    "strategy": strategy,
                    "original_data": dict(graph.nodes[node])
                })
                graph.remove_node(node)
        
        return graph, operations
    
    def _centrality_based_removal(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """基于中心性指标删除节点"""
        centrality_type = self.config.get('centrality_type', 'betweenness')  # 'betweenness', 'closeness', 'eigenvector'
        strategy = self.config.get('centrality_strategy', 'high_centrality')  # 'high_centrality', 'low_centrality'
        removal_ratio = self.config.get('centrality_removal_ratio', 0.1)
        
        # 计算中心性指标
        if centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(graph)
        elif centrality_type == 'closeness':
            centrality = nx.closeness_centrality(graph)
        elif centrality_type == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            except:
                # 如果特征向量中心性计算失败，使用度中心性
                centrality = nx.degree_centrality(graph)
        else:
            centrality = nx.degree_centrality(graph)
        
        # 排序节点
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        num_to_remove = int(len(sorted_nodes) * removal_ratio)
        operations = []
        
        if strategy == 'high_centrality':
            nodes_to_remove = [node for node, _ in sorted_nodes[:num_to_remove]]
        else:  # low_centrality
            nodes_to_remove = [node for node, _ in sorted_nodes[-num_to_remove:]]
        
        for node in nodes_to_remove:
            if node in graph.nodes():
                operations.append({
                    "operation": "centrality_based_remove_node",
                    "target": node,
                    "centrality_type": centrality_type,
                    "centrality_value": centrality[node],
                    "strategy": strategy,
                    "original_data": dict(graph.nodes[node])
                })
                graph.remove_node(node)
        
        return graph, operations
    
    def _community_based_perturbation(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """基于社区结构的扰动"""
        perturbation_type = self.config.get('community_perturbation_type', 'inter_community_edges')
        operations = []
        
        try:
            # 检测社区结构
            communities = list(nx.community.greedy_modularity_communities(graph))
            
            if perturbation_type == 'inter_community_edges':
                # 删除社区间的边
                operations = self._remove_inter_community_edges(graph, communities)
            elif perturbation_type == 'intra_community_edges':
                # 删除社区内的边
                operations = self._remove_intra_community_edges(graph, communities)
            elif perturbation_type == 'community_isolation':
                # 孤立某些社区
                operations = self._isolate_communities(graph, communities)
            
        except Exception as e:
            # 如果社区检测失败，使用简单的拓扑扰动
            operations = self._simple_topology_perturbation(graph)
        
        return graph, operations
    
    def _remove_inter_community_edges(self, graph: nx.Graph, communities: List) -> List[Dict]:
        """删除社区间的边"""
        # 创建节点到社区的映射
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        
        # 找到社区间的边
        inter_edges = []
        for edge in graph.edges():
            if (edge[0] in node_to_community and edge[1] in node_to_community and
                node_to_community[edge[0]] != node_to_community[edge[1]]):
                inter_edges.append(edge)
        
        # 删除部分社区间的边
        removal_ratio = self.config.get('inter_edge_removal_ratio', 0.3)
        num_to_remove = int(len(inter_edges) * removal_ratio)
        edges_to_remove = random.sample(inter_edges, min(num_to_remove, len(inter_edges)))
        
        operations = []
        for edge in edges_to_remove:
            operations.append({
                "operation": "remove_inter_community_edge",
                "target": edge,
                "source_community": node_to_community[edge[0]],
                "target_community": node_to_community[edge[1]],
                "original_data": dict(graph.edges[edge])
            })
            graph.remove_edge(*edge)
        
        return operations
    
    def _remove_intra_community_edges(self, graph: nx.Graph, communities: List) -> List[Dict]:
        """删除社区内的边"""
        operations = []
        removal_ratio = self.config.get('intra_edge_removal_ratio', 0.2)
        
        for i, community in enumerate(communities):
            community_nodes = set(community)
            intra_edges = [(u, v) for u, v in graph.edges() 
                          if u in community_nodes and v in community_nodes]
            
            num_to_remove = int(len(intra_edges) * removal_ratio)
            edges_to_remove = random.sample(intra_edges, min(num_to_remove, len(intra_edges)))
            
            for edge in edges_to_remove:
                operations.append({
                    "operation": "remove_intra_community_edge",
                    "target": edge,
                    "community": i,
                    "original_data": dict(graph.edges[edge])
                })
                graph.remove_edge(*edge)
        
        return operations
    
    def _isolate_communities(self, graph: nx.Graph, communities: List) -> List[Dict]:
        """孤立某些社区"""
        isolation_ratio = self.config.get('community_isolation_ratio', 0.2)
        num_to_isolate = int(len(communities) * isolation_ratio)
        communities_to_isolate = random.sample(range(len(communities)), 
                                             min(num_to_isolate, len(communities)))
        
        operations = []
        for community_idx in communities_to_isolate:
            community_nodes = set(communities[community_idx])
            
            # 删除该社区与其他社区的所有连接
            edges_to_remove = []
            for edge in graph.edges():
                if ((edge[0] in community_nodes and edge[1] not in community_nodes) or
                    (edge[1] in community_nodes and edge[0] not in community_nodes)):
                    edges_to_remove.append(edge)
            
            for edge in edges_to_remove:
                operations.append({
                    "operation": "isolate_community",
                    "target": edge,
                    "isolated_community": community_idx,
                    "original_data": dict(graph.edges[edge])
                })
                graph.remove_edge(*edge)
        
        return operations
    
    def _add_topology_noise(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """添加拓扑噪声"""
        noise_type = self.config.get('topology_noise_type', 'random_edges')
        operations = []
        
        if noise_type == 'random_edges':
            operations = self._add_random_edges(graph)
        elif noise_type == 'hub_creation':
            operations = self._create_artificial_hubs(graph)
        elif noise_type == 'bridge_disruption':
            operations = self._disrupt_bridges(graph)
        
        return graph, operations
    
    def _add_random_edges(self, graph: nx.Graph) -> List[Dict]:
        """添加随机边作为拓扑噪声"""
        noise_ratio = self.config.get('random_edge_noise_ratio', 0.05)
        num_noise_edges = int(graph.number_of_edges() * noise_ratio)
        
        nodes = list(graph.nodes())
        operations = []
        
        for _ in range(num_noise_edges):
            if len(nodes) >= 2:
                source, target = random.sample(nodes, 2)
                if not graph.has_edge(source, target):
                    # 添加噪声边属性
                    noise_attrs = {
                        "weight": np.random.uniform(0.1, 1.0),
                        "is_noise": True,
                        "noise_type": "topology"
                    }
                    
                    graph.add_edge(source, target, **noise_attrs)
                    operations.append({
                        "operation": "add_topology_noise_edge",
                        "target": (source, target),
                        "attributes": noise_attrs
                    })
        
        return operations
    
    def _create_artificial_hubs(self, graph: nx.Graph) -> List[Dict]:
        """创建人工枢纽节点"""
        num_hubs = self.config.get('artificial_hubs', 2)
        hub_degree = self.config.get('hub_degree', 10)
        
        nodes = list(graph.nodes())
        operations = []
        
        for i in range(num_hubs):
            # 创建新的枢纽节点
            hub_node = f"artificial_hub_{i}"
            hub_attrs = {
                "is_artificial_hub": True,
                "hub_id": i,
                "value": 999
            }
            
            graph.add_node(hub_node, **hub_attrs)
            
            # 连接到随机节点
            target_nodes = random.sample(nodes, min(hub_degree, len(nodes)))
            for target in target_nodes:
                edge_attrs = {
                    "weight": np.random.uniform(0.5, 1.0),
                    "is_artificial": True
                }
                graph.add_edge(hub_node, target, **edge_attrs)
            
            operations.append({
                "operation": "create_artificial_hub",
                "target": hub_node,
                "connected_nodes": target_nodes,
                "hub_attributes": hub_attrs
            })
        
        return operations
    
    def _disrupt_bridges(self, graph: nx.Graph) -> List[Dict]:
        """破坏桥边（关键连接）"""
        try:
            bridges = list(nx.bridges(graph))
            disruption_ratio = self.config.get('bridge_disruption_ratio', 0.3)
            num_to_disrupt = int(len(bridges) * disruption_ratio)
            
            bridges_to_remove = random.sample(bridges, min(num_to_disrupt, len(bridges)))
            operations = []
            
            for bridge in bridges_to_remove:
                operations.append({
                    "operation": "disrupt_bridge",
                    "target": bridge,
                    "original_data": dict(graph.edges[bridge])
                })
                graph.remove_edge(*bridge)
            
            return operations
        except:
            return []
    
    def _simple_topology_perturbation(self, graph: nx.Graph) -> List[Dict]:
        """简单的拓扑扰动（当社区检测失败时使用）"""
        operations = []
        
        # 随机删除一些边
        removal_ratio = self.config.get('simple_edge_removal_ratio', 0.1)
        num_to_remove = int(graph.number_of_edges() * removal_ratio)
        edges_to_remove = random.sample(list(graph.edges()), 
                                      min(num_to_remove, graph.number_of_edges()))
        
        for edge in edges_to_remove:
            operations.append({
                "operation": "simple_topology_remove_edge",
                "target": edge,
                "original_data": dict(graph.edges[edge])
            })
            graph.remove_edge(*edge)
        
        return operations
