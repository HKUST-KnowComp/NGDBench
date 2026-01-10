"""
基本图查询算子实现
支持在 .gpickle 格式的 NetworkX 图上进行查询操作
"""

import pickle
import networkx as nx
from typing import List, Dict, Any, Set, Optional, Union
import re

# 全局图对象（支持有向图和无向图）
_graph: Optional[Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]] = None


def load_graph(graph_path: str) -> Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]:
    """加载图文件"""
    global _graph
    with open(graph_path, 'rb') as f:
        _graph = pickle.load(f)
    print(f"✅ 已加载图: {graph_path}")
    print(f"   - 节点数: {_graph.number_of_nodes():,}")
    print(f"   - 边数: {_graph.number_of_edges():,}")
    print(f"   - 图类型: {'有向图' if _graph.is_directed() else '无向图'}")
    return _graph


def get_graph() -> Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]:
    """获取全局图对象"""
    if _graph is None:
        raise RuntimeError("图未加载，请先调用 load_graph()")
    return _graph

def save_graph(G):

    nx.write_gpickle(G, "graph.gpickle")

def Scan(type: str, mode = "normal") -> List[Any]:
    """
    匹配特定类型的所有节点
    
    Args:
        type: 节点类型（节点属性中的 'type' 或 'label' 字段）
    
    Returns:
        匹配的节点ID列表
    """
    G = get_graph()
    matched_nodes = []
    if mode == "normal":
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_type = node_data.get('type') or node_data.get('label') or node_data.get('node_type')
            if node_type == type:
                matched_nodes.append(node_id)
    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
    
    return matched_nodes


def Filter(input: Union[List[Any], Set[Any]], filters: Dict[str, Any], mode = "normal") -> List[Any]:
    """
    根据属性过滤节点
    
    Args:
        input: 输入节点集合（节点ID列表）
        filters: 过滤条件字典，格式为 {属性名: 属性值}
                 支持比较操作，如 ">0.9" 表示大于0.9
    
    Returns:
        过滤后的节点ID列表
    """
    G = get_graph()
    filtered_nodes = []
    if mode == "normal":
        for node_id in input:
            node_data = G.nodes[node_id]
            match = True
            for prop_key, prop_value in filters.items():
                if prop_key not in node_data:
                    match = False
                    break
            if match:
                filtered_nodes.append(node_id)
    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
    return filtered_nodes



def Project(
    input: Union[List[Any], Set[Any]],
    edge: str,
    edge_filter: Optional[Dict[str, Any]] = None,
    direction: str = "OUT",
    mode = "normal"
) -> List[Any]:
    """
    从输入节点遍历边到达目标节点
    支持有向图和无向图
    
    Args:
        input: 输入节点集合（节点ID列表）
        edge: 边类型/标签（边类型已经决定了目标节点类型）
        edge_filter: 边属性过滤条件（可选）
        direction: 遍历方向
                   - 有向图: "OUT" 表示出边，"IN" 表示入边
                   - 无向图: "OUT" 和 "IN" 效果相同，都是遍历所有连接的边
    
    Returns:
        到达的目标节点ID列表
    """
    G = get_graph()
    result_nodes = []
    visited = set()
    is_directed = G.is_directed()
    if mode == "normal":
        for source_node in input:
            # 根据图类型和方向获取边
            if is_directed:
                # 有向图：根据方向选择出边或入边
                if direction == "OUT":
                    edges = G.out_edges(source_node, data=True)
                else:  # direction == "IN"
                    edges = G.in_edges(source_node, data=True)
            else:
                # 无向图：获取所有连接的边（direction参数在无向图中被忽略）
                edges = G.edges(source_node, data=True)
            
            for edge_data in edges:
                # 解析边的源节点和目标节点
                if is_directed:
                    # 有向图：边的方向是固定的
                    if direction == "OUT":
                        source, target, edge_attrs = edge_data
                    else:  # direction == "IN"
                        target, source, edge_attrs = edge_data
                else:
                    # 无向图：需要确定哪个是源节点，哪个是目标节点
                    u, v, edge_attrs = edge_data
                    if u == source_node:
                        source, target = u, v
                    else:
                        source, target = v, u
                    # 在无向图中，如果指定了direction，我们仍然需要确保逻辑正确
                    # 但通常无向图中，我们总是从source_node出发到另一个节点
                
                # 检查边类型
                edge_type = edge_attrs.get('type') or edge_attrs.get('label') or edge_attrs.get('edge_type') or edge_attrs.get('relation')
                if edge_type != edge:
                    continue
                
                # 检查边属性过滤
                # if edge_filter:
                #     edge_match = True
                #     for prop_key, prop_value in edge_filter.items():
                #         if prop_key not in edge_attrs:
                #             edge_match = False
                #             break
                        
                #         edge_prop_value = edge_attrs[prop_key]
                        
                #         # 处理比较操作
                #         if isinstance(prop_value, str) and prop_value.startswith(('>', '<', '>=', '<=')):
                #             match_obj = re.match(r'([><=]+)(.+)', prop_value)
                #             if match_obj:
                #                 op = match_obj.group(1)
                #                 threshold = float(match_obj.group(2))
                                
                #                 if op == '>':
                #                     if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value > threshold):
                #                         edge_match = False
                #                         break
                #                 elif op == '<':
                #                     if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value < threshold):
                #                         edge_match = False
                #                         break
                #                 elif op == '>=':
                #                     if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value >= threshold):
                #                         edge_match = False
                #                         break
                #                 elif op == '<=':
                #                     if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value <= threshold):
                #                         edge_match = False
                #                         break
                #         else:
                #             if edge_prop_value != prop_value:
                #                 edge_match = False
                #                 break
                    
                #     if not edge_match:
                #         continue
                
                # 添加到结果中（去重）
                if target not in visited:
                    result_nodes.append(target)
                    visited.add(target)
    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
    return result_nodes


def TraverseWithPath(
    input: Union[List[Any], Set[Any]],
    edge: str,
    target_type: Optional[str] = None,
    edge_filter: Optional[Dict[str, Any]] = None,
    direction: str = "OUT"
) -> List[Dict[str, Any]]:
    """
    从输入节点遍历边到达目标节点，返回路径信息（包含源节点和目标节点）
    
    Args:
        input: 输入节点集合（节点ID列表）
        edge: 边类型/标签
        target_type: 目标节点类型（可选）
        edge_filter: 边属性过滤条件（可选）
        direction: 遍历方向
    
    Returns:
        路径列表，每个路径是一个字典：{"source_node": source_id, "target_node": target_id}
    """
    G = get_graph()
    result_paths = []
    visited_edges = set()
    is_directed = G.is_directed()
    
    for source_node in input:
        if is_directed:
            if direction == "OUT":
                edges = G.out_edges(source_node, data=True)
            else:
                edges = G.in_edges(source_node, data=True)
        else:
            edges = G.edges(source_node, data=True)
        
        for edge_data in edges:
            if is_directed:
                if direction == "OUT":
                    source, target, edge_attrs = edge_data
                else:
                    target, source, edge_attrs = edge_data
            else:
                u, v, edge_attrs = edge_data
                if u == source_node:
                    source, target = u, v
                else:
                    source, target = v, u
            
            # 检查边类型
            edge_type = edge_attrs.get('type') or edge_attrs.get('label') or edge_attrs.get('edge_type') or edge_attrs.get('relation')
            if edge_type != edge:
                continue
            
            # 检查边属性过滤
            if edge_filter:
                edge_match = True
                for prop_key, prop_value in edge_filter.items():
                    if prop_key not in edge_attrs:
                        edge_match = False
                        break
                    
                    edge_prop_value = edge_attrs[prop_key]
                    
                    if isinstance(prop_value, str) and prop_value.startswith(('>', '<', '>=', '<=')):
                        match_obj = re.match(r'([><=]+)(.+)', prop_value)
                        if match_obj:
                            op = match_obj.group(1)
                            threshold = float(match_obj.group(2))
                            
                            if op == '>':
                                if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value > threshold):
                                    edge_match = False
                                    break
                            elif op == '<':
                                if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value < threshold):
                                    edge_match = False
                                    break
                            elif op == '>=':
                                if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value >= threshold):
                                    edge_match = False
                                    break
                            elif op == '<=':
                                if not (isinstance(edge_prop_value, (int, float)) and edge_prop_value <= threshold):
                                    edge_match = False
                                    break
                    else:
                        if edge_prop_value != prop_value:
                            edge_match = False
                            break
                
                if not edge_match:
                    continue
            
            # 检查目标节点类型
            if target_type:
                target_data = G.nodes[target]
                target_node_type = target_data.get('type') or target_data.get('label') or target_data.get('node_type')
                if target_node_type != target_type:
                    continue
            
            # 添加到结果中（去重）
            edge_key = (source, target)
            if edge_key not in visited_edges:
                result_paths.append({"source_node": source, "target_node": target})
                visited_edges.add(edge_key)
    
    return result_paths


def Minus(set1: Union[List[Any], Set[Any]], set2: Union[List[Any], Set[Any]], mode = "normal") -> List[Any]:
    """
    集合差运算：返回 set1 中不在 set2 中的元素
    
    Args:
        set1: 第一个集合（节点ID列表）
        set2: 第二个集合（节点ID列表）
    
    Returns:
        差集结果（节点ID列表）
    """
    if mode == "normal":
        set1_set = set(set1) if not isinstance(set1, set) else set1
        set2_set = set(set2) if not isinstance(set2, set) else set2
        result = list(set1_set - set2_set)
        return result
    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
    return result


def Intersection(set1: Union[List[Any], Set[Any]], set2: Union[List[Any], Set[Any]], mode = "normal") -> List[Any]:
    """
    集合交集运算：返回 set1 和 set2 中都存在的元素
    
    Args:
        set1: 第一个集合（节点ID列表）
        set2: 第二个集合（节点ID列表）
        mode: 运算模式，支持 "normal", "neural", "agent"
    
    Returns:
        交集结果（节点ID列表）
    """
    if mode == "normal":
        set1_set = set(set1) if not isinstance(set1, set) else set1
        set2_set = set(set2) if not isinstance(set2, set) else set2
        result = list(set1_set & set2_set)
        return result
    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
    return result


def Union(set1: Union[List[Any], Set[Any]], set2: Union[List[Any], Set[Any]], mode = "normal") -> List[Any]:
    """
    集合并集运算：返回 set1 和 set2 中所有不重复的元素
    
    Args:
        set1: 第一个集合（节点ID列表）
        set2: 第二个集合（节点ID列表）
        mode: 运算模式，支持 "normal", "neural", "agent"
    
    Returns:
        并集结果（节点ID列表）
    """
    if mode == "normal":
        set1_set = set(set1) if not isinstance(set1, set) else set1
        set2_set = set(set2) if not isinstance(set2, set) else set2
        result = list(set1_set | set2_set)
        return result
    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
    return result


def GroupBy(
    input: Union[List[Any], List[Dict[str, Any]]],
    key: str = "target_node"
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    对输入进行分组
    
    Args:
        input: 输入数据
               - 如果是节点列表，会尝试从图中重建路径信息（需要上下文）
               - 如果是路径列表（字典列表，包含 source_node 和 target_node），则按 key 分组
        key: 分组键，可以是 "target_node" 或 "source_node"
    
    Returns:
        分组后的字典，格式为 {group_key: [路径列表]}
    """
    # 如果输入是路径列表（字典列表）
    if input and isinstance(input[0], dict):
        grouped = {}
        for path in input:
            group_key = path.get(key)
            if group_key is not None:
                if group_key not in grouped:
                    grouped[group_key] = []
                grouped[group_key].append(path)
        return grouped
    
    # 如果输入是节点列表，我们需要将其转换为路径格式
    # 这种情况下，我们假设每个节点都是 target_node，source_node 需要从上下文中获取
    # 为了简化，我们创建一个默认的路径结构
    grouped = {}
    for node in input:
        # 创建一个简单的路径结构，target_node 是节点本身
        # 注意：这种情况下 source_node 可能不准确，需要调用者确保使用 TraverseWithPath
        path = {"target_node": node, "source_node": None}
        group_key = path.get(key)
        if group_key is not None:
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(path)
    
    return grouped


def Aggregate(
    input: Union[Dict[Any, List[Dict[str, Any]]], List[Any]],
    func: str = "COUNT",
    target: Optional[str] = None,
    order: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    聚合操作
    
    Args:
        input: 输入数据
              - 如果是字典（GroupBy的结果），格式为 {group_key: [路径列表]}
              - 如果是列表（节点列表或路径列表），会先进行计数
        func: 聚合函数，支持 "COUNT", "MIN", "MAX", "AVG", "SUM"
        target: 聚合目标，可以是 "source_node" 或 "target_node" 或节点属性名
        order: 排序方式，"ASC" 或 "DESC"
        limit: 限制返回结果数量
    
    Returns:
        聚合结果列表，每个元素是 {"key": group_key, "value": aggregated_value}
    """
    G = get_graph()
    results = []
    
    # 如果输入是字典（GroupBy的结果）
    if isinstance(input, dict):
        for group_key, paths in input.items():
            if func.upper() == "COUNT":
                # 如果指定了 target，统计不同 target 的数量
                if target:
                    if paths and isinstance(paths[0], dict):
                        # 统计不同的 target 值
                        unique_targets = set()
                        for path in paths:
                            target_value = path.get(target)
                            if target_value is not None:
                                unique_targets.add(target_value)
                        value = len(unique_targets)
                    else:
                        value = len(paths)
                else:
                    value = len(paths)
            
            elif func.upper() == "MIN":
                if target:
                    values = []
                    for path in paths:
                        if isinstance(path, dict):
                            val = path.get(target)
                        else:
                            # 如果是节点ID，尝试获取节点属性
                            node_data = G.nodes.get(path, {})
                            val = node_data.get(target)
                        if val is not None and isinstance(val, (int, float)):
                            values.append(val)
                    value = min(values) if values else None
                else:
                    value = None
            
            elif func.upper() == "MAX":
                if target:
                    values = []
                    for path in paths:
                        if isinstance(path, dict):
                            val = path.get(target)
                        else:
                            node_data = G.nodes.get(path, {})
                            val = node_data.get(target)
                        if val is not None and isinstance(val, (int, float)):
                            values.append(val)
                    value = max(values) if values else None
                else:
                    value = None
            
            elif func.upper() == "AVG":
                if target:
                    values = []
                    for path in paths:
                        if isinstance(path, dict):
                            val = path.get(target)
                        else:
                            node_data = G.nodes.get(path, {})
                            val = node_data.get(target)
                        if val is not None and isinstance(val, (int, float)):
                            values.append(val)
                    value = sum(values) / len(values) if values else None
                else:
                    value = None
            
            elif func.upper() == "SUM":
                if target:
                    values = []
                    for path in paths:
                        if isinstance(path, dict):
                            val = path.get(target)
                        else:
                            node_data = G.nodes.get(path, {})
                            val = node_data.get(target)
                        if val is not None and isinstance(val, (int, float)):
                            values.append(val)
                    value = sum(values) if values else None
                else:
                    value = None
            
            else:
                value = None
            
            if value is not None:
                results.append({"key": group_key, "value": value})
    
    # 如果输入是列表（节点列表）
    elif isinstance(input, list):
        if func.upper() == "COUNT":
            value = len(input)
            results.append({"key": "total", "value": value})
        else:
            # 对于其他聚合函数，需要指定 target
            if target:
                values = []
                for item in input:
                    if isinstance(item, dict):
                        val = item.get(target)
                    else:
                        node_data = G.nodes.get(item, {})
                        val = node_data.get(target)
                    if val is not None and isinstance(val, (int, float)):
                        values.append(val)
                
                if func.upper() == "MIN":
                    value = min(values) if values else None
                elif func.upper() == "MAX":
                    value = max(values) if values else None
                elif func.upper() == "AVG":
                    value = sum(values) / len(values) if values else None
                elif func.upper() == "SUM":
                    value = sum(values) if values else None
                else:
                    value = None
                
                if value is not None:
                    results.append({"key": "total", "value": value})
    
    # 排序
    if order:
        reverse = (order.upper() == "DESC")
        results.sort(key=lambda x: x["value"], reverse=reverse)
    
    # 限制数量
    if limit is not None:
        results = results[:limit]
    
    return results



def Insert(
    mode: str = "normal",
    insert_type: str = None,   # "node" | "property"
    node_type: str = None,
    properties: dict = None,
    node_id: int = None,
    key: str = None,
    value = None
):
    G = get_graph()

    if mode == "normal":

        # ========== 插入节点 ==========
        if insert_type == "node":
            new_id = generate_new_node_id(G)

            attr = {}
            if node_type is not None:
                attr["type"] = node_type
            if properties is not None:
                attr.update(properties)

            G.add_node(new_id, **attr)
            save_graph(G)
            return new_id

        # ========== 插入属性 ==========
        elif insert_type == "property":
            if node_id not in G:
                raise ValueError(f"Node {node_id} does not exist.")

            if key is None:
                raise ValueError("key must be provided for property insert.")

            G.nodes[node_id][key] = value
            save_graph(G)
            return node_id

        else:
            raise ValueError("Unknown insert_type")

    elif mode == "neural":
        pass
    elif mode == "agent":
        pass

def Update(
    mode: str = "normal",
    update_type: str = None,  # "rename_node" | "node_type" | "property"
    node_id: int = None,
    old_id: int = None,
    new_id: int = None,
    new_type: str = None,
    key: str = None,
    new_value = None
):
    G = get_graph()

    if mode == "normal":

        # ========== 重命名节点 ID ==========
        if update_type == "rename_node":
            if old_id not in G:
                raise ValueError(f"Node {old_id} does not exist.")
            if new_id in G:
                raise ValueError(f"Node {new_id} already exists.")

            attrs = dict(G.nodes[old_id])
            G.add_node(new_id, **attrs)

            for u, v, data in list(G.edges(old_id, data=True)):
                if u == old_id:
                    G.add_edge(new_id, v, **data)
                else:
                    G.add_edge(u, new_id, **data)

            G.remove_node(old_id)
            save_graph(G)
            return new_id

        # ========== 修改节点类型 ==========
        elif update_type == "node_type":
            if node_id not in G:
                raise ValueError(f"Node {node_id} does not exist.")

            G.nodes[node_id]["type"] = new_type
            save_graph(G)
            return node_id

        # ========== 修改属性 ==========
        elif update_type == "property":
            if node_id not in G:
                raise ValueError(f"Node {node_id} does not exist.")

            if key is None:
                raise ValueError("key must be provided.")

            G.nodes[node_id][key] = new_value
            save_graph(G)
            return node_id

        else:
            raise ValueError("Unknown update_type")

    elif mode == "neural":
        pass
    elif mode == "agent":
        pass


def Delete(
    mode: str = "normal",
    delete_type: str = None,   # "node" | "property"
    node_id: int = None,
    key: str = None
):
    G = get_graph()

    if mode == "normal":

        # ========== 删除节点 ==========
        if delete_type == "node":
            if node_id not in G:
                raise ValueError(f"Node {node_id} does not exist.")

            G.remove_node(node_id)
            save_graph(G)
            return

        # ========== 删除属性 ==========
        elif delete_type == "property":
            if node_id not in G:
                raise ValueError(f"Node {node_id} does not exist.")

            if key not in G.nodes[node_id]:
                raise ValueError(f"Property {key} does not exist.")

            del G.nodes[node_id][key]
            save_graph(G)
            return

        else:
            raise ValueError("Unknown delete_type")

    elif mode == "neural":
        pass
    elif mode == "agent":
        pass
