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
        # get the summary of the graph
        G = self.graph
        degrees = [d for _, d in G.degree()]
        print(
            "📊 Graph Statistics:\n"
            f"Number of nodes: {G.number_of_nodes()}\n"
            f"Number of edges: {G.number_of_edges()}\n"
            f"Average degree: {sum(degrees) / len(degrees):.2f}\n"
            f"Directed: {G.is_directed()}"
        )
        print()