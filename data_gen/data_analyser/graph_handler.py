"""
Graph handler for graph analysis
"""
import networkx as nx
from collections import defaultdict


class GraphInspector:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def in_degree(self, node: str) -> int:
        # get the in degree of a node
        return self.graph.in_degree(node)

    def out_degree(self, node: str) -> int:
        # get the out degree of a node
        return self.graph.out_degree(node)

    def degree(self, node: str) -> int:
        # get the degree of a node
        return self.graph.degree(node)

    def in_edges(self, node: str):
        # get the in edges of a node
        return list(self.graph.in_edges(node, data=True))

    def out_edges(self, node: str):
        # get the out edges of a node
        return list(self.graph.out_edges(node, data=True))

    def out_degree_by_relation(self, node: str):
        # get the out degree of a node by relation
        rel_outdegree = defaultdict(int)
        for _, _, data in self.graph.out_edges(node, data=True):
            rel = data.get('label')
            rel_outdegree[rel] += 1
        return dict(rel_outdegree)

    def edges_by_relation(self, node: str, relation_label: str):
        # get the edges of a node by relation
        edges = []
        for u, v, data in self.graph.out_edges(node, data=True):
            if data.get('label') == relation_label:
                edges.append((u, v))
        return edges

    def summary(self):
        """显示图的基本统计信息"""
        G = self.graph
        degrees = [d for _, d in G.degree()]
        
        print("=" * 60)
        print("Graph Statistics")
        print("=" * 60)
        print(f"Number of nodes: {G.number_of_nodes():,}")
        print(f"Number of edges: {G.number_of_edges():,}")
        if degrees:
            print(f"Average degree: {sum(degrees) / len(degrees):.2f}")
            print(f"Max degree: {max(degrees)}")
            print(f"Min degree: {min(degrees)}")
        print(f"Directed: {G.is_directed()}")
        print()

    def node_type_distribution(self):
        """统计节点类型分布"""
        G = self.graph
        type_count = defaultdict(int)
        
        for node, data in G.nodes(data=True):
            raw_type = data.get('label') or data.get('type') or data.get('node_type') or 'Unknown'

            # 将节点类型统一转换为可哈希的字符串，避免 list 之类导致报错
            if isinstance(raw_type, list):
                # 如果是列表，拼成一个逗号分隔的字符串；根据需要也可以改成只取第一个元素
                node_type = ",".join(map(str, raw_type)) if raw_type else "Unknown"
            else:
                node_type = str(raw_type)

            type_count[node_type] += 1
        
        print("=" * 60)
        print("Node Type Distribution")
        print("=" * 60)
        for node_type, count in sorted(type_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / G.number_of_nodes()) * 100
            print(f"  {node_type}: {count:,} ({percentage:.2f}%)")
        print()
        return dict(type_count)

    def edge_type_distribution(self):
        """统计边（关系）类型分布"""
        G = self.graph
        rel_count = defaultdict(int)
        
        for u, v, data in G.edges(data=True):
            rel_type = data.get('label') or data.get('relation') or data.get('type') or 'Unknown'
            rel_count[rel_type] += 1
        
        print("=" * 60)
        print("Edge/Relation Type Distribution")
        print("=" * 60)
        for rel_type, count in sorted(rel_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / G.number_of_edges()) * 100
            print(f"  {rel_type}: {count:,} ({percentage:.2f}%)")
        print()
        return dict(rel_count)

    def node_attributes_statistics(self):
        """统计节点属性分布"""
        G = self.graph
        attr_count = defaultdict(int)  # 统计每个属性出现的节点数
        attr_types = defaultdict(set)  # 统计每个属性的值类型
        attr_samples = defaultdict(list)  # 保存每个属性的示例值
        
        for node, data in G.nodes(data=True):
            for attr_key, attr_value in data.items():
                attr_count[attr_key] += 1
                attr_types[attr_key].add(type(attr_value).__name__)
                # 保存前3个示例值
                if len(attr_samples[attr_key]) < 3:
                    attr_samples[attr_key].append(attr_value)
        
        print("=" * 60)
        print("Node Attributes Statistics")
        print("=" * 60)
        print(f"Total number of nodes: {G.number_of_nodes():,}")
        print(f"Total number of unique attributes: {len(attr_count)}")
        print()
        
        # 按出现频率排序
        sorted_attrs = sorted(attr_count.items(), key=lambda x: x[1], reverse=True)
        
        for attr_key, count in sorted_attrs:
            percentage = (count / G.number_of_nodes()) * 100
            types_str = ", ".join(sorted(attr_types[attr_key]))
            samples_str = ", ".join([str(s)[:50] for s in attr_samples[attr_key][:3]])
            
            print(f"  {attr_key}:")
            print(f"    - 出现节点数: {count:,} ({percentage:.2f}%)")
            print(f"    - 值类型: {types_str}")
            if samples_str:
                print(f"    - 示例值: {samples_str}")
            print()
        
        print("=" * 60 + "\n")
        
        return {
            "attribute_counts": dict(attr_count),
            "attribute_types": {k: list(v) for k, v in attr_types.items()},
            "attribute_samples": {k: v for k, v in attr_samples.items()}
        }

    def check_connectivity(self, show_details: bool = True):
        """
        检查图的连通性
        对于有向图，检查弱连通性（忽略边的方向）
        
        Args:
            show_details: 是否显示详细的连通分量信息
            
        Returns:
            dict: 包含连通性信息的字典
        """
        G = self.graph
        
        # 检查弱连通性（忽略边方向）
        is_weakly_connected = nx.is_weakly_connected(G)
        num_components = nx.number_weakly_connected_components(G)
        
        # 获取所有弱连通分量
        components = list(nx.weakly_connected_components(G))
        component_sizes = sorted([len(c) for c in components], reverse=True)
        
        print("=" * 60)
        print("Connectivity Analysis")
        print("=" * 60)
        
        if is_weakly_connected:
            print("[Connected] 图是弱连通的（所有节点在忽略边方向后相互可达）")
        else:
            print("[Disconnected] 图不是连通的，包含多个分离的子图")
        
        print(f"\n连通分量统计:")
        print(f"  - 连通分量数量: {num_components}")
        print(f"  - 最大分量节点数: {component_sizes[0]:,}")
        print(f"  - 最小分量节点数: {component_sizes[-1]:,}")
        
        if num_components > 1:
            print(f"\n各连通分量大小分布:")
            # 显示前10个最大的分量
            for i, size in enumerate(component_sizes[:10], 1):
                percentage = (size / G.number_of_nodes()) * 100
                print(f"  {i}. 分量大小: {size:,} 节点 ({percentage:.2f}%)")
            
            if len(component_sizes) > 10:
                print(f"  ... 还有 {len(component_sizes) - 10} 个更小的分量")
        
        if show_details and num_components > 1 and num_components <= 20:
            print(f"\n连通分量详情（前5个最大分量的采样节点）:")
            for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5], 1):
                sample_nodes = list(comp)[:3]
                print(f"  分量 {i} ({len(comp)} 节点): {sample_nodes}")
        
        print("=" * 60 + "\n")
        
        return {
            "is_connected": is_weakly_connected,
            "num_components": num_components,
            "component_sizes": component_sizes,
            "largest_component_size": component_sizes[0],
            "smallest_component_size": component_sizes[-1]
        }

    def degree_distribution(self):
        """分析度数分布"""
        G = self.graph
        degrees = [d for _, d in G.degree()]
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        print("=" * 60)
        print("Degree Distribution")
        print("=" * 60)
        print(f"Total Degree  - Mean: {sum(degrees)/len(degrees):.2f}, Max: {max(degrees)}, Min: {min(degrees)}")
        print(f"In Degree     - Mean: {sum(in_degrees)/len(in_degrees):.2f}, Max: {max(in_degrees)}, Min: {min(in_degrees)}")
        print(f"Out Degree    - Mean: {sum(out_degrees)/len(out_degrees):.2f}, Max: {max(out_degrees)}, Min: {min(out_degrees)}")
        
        # 找出度数最高的节点
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n度数最高的5个节点:")
        for node, deg in top_nodes:
            node_data = G.nodes[node]
            node_label = node_data.get('label') or node_data.get('name') or str(node)[:50]
            print(f"  {node_label}: degree={deg}")
        print()

    def sample_nodes(self, n: int = 30, output_file: str = None):
        """随机采样节点并显示详情

        Args:
            n (int): 采样节点数
            output_file (str): 如果给定，则把输出保存到此文件，否则输出到控制台
        """
        import random
        import sys
        from contextlib import redirect_stdout

        def _run_sample_nodes():
            G = self.graph
            all_nodes = list(G.nodes())
            sample_size = min(n, len(all_nodes))
            sampled = random.sample(all_nodes, sample_size)
            
            print("=" * 60)
            print(f"Sample Nodes ({sample_size} nodes)")
            print("=" * 60)
            
            for node in sampled:
                data = G.nodes[node]
                print(f"\nNode: {node}")
                print(f"  Attributes: {dict(data)}")
                print(f"  In-degree: {G.in_degree(node)}, Out-degree: {G.out_degree(node)}")
                
                # 显示部分出边
                out_edges = list(G.out_edges(node, data=True))[:3]
                if out_edges:
                    print(f"  Sample out-edges:")
                    for u, v, edata in out_edges:
                        rel = edata.get('label') or edata.get('relation') or 'N/A'
                        print(f"    --[{rel}]--> {v}")
            print()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    _run_sample_nodes()
        else:
            _run_sample_nodes()

    def full_analysis(self, output_file: str = None):
        """
        执行完整的图分析
        
        Args:
            output_file: 输出文件路径，如果为 None 则输出到控制台
        """
        from contextlib import redirect_stdout
        from datetime import datetime
        
        def _run_analysis():
            print("\n" + "=" * 60)
            print("          FULL GRAPH ANALYSIS")
            print(f"          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60 + "\n")
            
            self.summary()
            self.node_type_distribution()
            self.edge_type_distribution()
            self.degree_distribution()
            self.check_connectivity()
            self.sample_nodes()

            self.node_attributes_statistics()
            print("=" * 60)
            print("          ANALYSIS COMPLETE")
            print("=" * 60)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    _run_analysis()
            print(f"分析结果已保存到: {output_file}")
        else:
            _run_analysis()