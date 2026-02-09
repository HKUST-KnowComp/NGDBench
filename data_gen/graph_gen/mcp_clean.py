from dataload_clean import *
import pickle
import json

def _convert_list_attributes_to_string(g):
    """
    将图中所有节点和边的列表类型属性转换为字符串（JSON格式），
    因为 GraphML 不支持列表类型的属性值。
    """
    import networkx as nx
    
    # 处理节点属性
    for node in g.nodes():
        node_attrs = g.nodes[node]
        for key, value in list(node_attrs.items()):
            if isinstance(value, list):
                # 将列表转换为 JSON 字符串
                node_attrs[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # 将字典也转换为 JSON 字符串（GraphML 可能也不支持复杂字典）
                node_attrs[key] = json.dumps(value, ensure_ascii=False)
    
    # 处理边属性
    for u, v, key in g.edges(keys=True):
        edge_attrs = g[u][v][key] if g.is_multigraph() else g[u][v]
        for attr_key, attr_value in list(edge_attrs.items()):
            if isinstance(attr_value, list):
                # 将列表转换为 JSON 字符串
                edge_attrs[attr_key] = json.dumps(attr_value, ensure_ascii=False)
            elif isinstance(attr_value, dict):
                # 将字典也转换为 JSON 字符串
                edge_attrs[attr_key] = json.dumps(attr_value, ensure_ascii=False)

def mcp_clean(input_file, output_file):
    import os
    import networkx as nx

    # 读取输入文件
    if input_file.endswith(".graphml"):
        g = nx.read_graphml(input_file)
    else:
        with open(input_file, "rb") as f:
            g = pickle.load(f)
    
    # 处理图
    g, stats = remove_concept_nodes_and_annotate_neighbors(g)
    
    # 根据输出文件扩展名选择保存格式
    if output_file.endswith(".graphml"):
        # 输出为 GraphML 格式
        # GraphML 不支持列表和字典类型的属性值，需要先转换
        _convert_list_attributes_to_string(g)
        nx.write_graphml(g, output_file)
    else:
        # 输出为 pickle 格式
        with open(output_file, "wb") as f:
            pickle.dump(g, f)

if __name__ == "__main__":
    mcp_clean("graph_buffer/multi_financial_graph.graphml", "graph_buffer/multi_financial_graph_concepts_aligned.gpickle")