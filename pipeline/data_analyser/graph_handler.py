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
            node_type = data.get('label') or data.get('type') or data.get('node_type') or 'Unknown'
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

    def sample_nodes(self, n: int = 5):
        """随机采样节点并显示详情"""
        import random
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