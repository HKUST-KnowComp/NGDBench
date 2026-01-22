import pickle
import networkx as nx
from pathlib import Path
def _ensure_parent_dir(path_like):
    output_path = Path(path_like)
    parent = output_path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return output_path


def remove_isolated_nodes_from_file(input_file, output_file=None):
    """
    Remove isolated nodes (nodes with total degree 0) from a gpickle or GraphML file.

    Parameters
    ----------
    input_file : str or Path
        Path to the input graph file. Supported formats: .gpickle, .graphml.
    output_file : str or Path, optional
        Path to save the cleaned graph. If not provided, a new file will be
        created alongside the input file with suffix `_no_isolates`.
    """
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()

    # Load graph according to file type
    if suffix == ".gpickle":
        # 使用标准 pickle 读入，由于前面写入时就是直接 pickle.dump
        with open(input_path, "rb") as f:
            g = pickle.load(f)
    elif suffix == ".graphml":
        g = nx.read_graphml(input_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only .gpickle and .graphml are supported.")

    # Find and remove isolated nodes
    isolated_nodes = list(nx.isolates(g))
    if isolated_nodes:
        g.remove_nodes_from(isolated_nodes)
        print(f"Removed {len(isolated_nodes)} isolated nodes from {input_path}.")
    else:
        print(f"No isolated nodes found in {input_path}.")

    # Determine output path
    if output_file is None:
        output_path = input_path.with_name(input_path.stem + "_no_isolates" + input_path.suffix)
    else:
        output_path = _ensure_parent_dir(output_file)

    # Save graph back using the same format
    if suffix == ".gpickle":
        with open(output_path, "wb") as f:
            pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
    elif suffix == ".graphml":
        nx.write_graphml(g, output_path, infer_numeric_types=True)

    print(f"Saved graph without isolated nodes to: {output_path}")


def normalize_mcp_relations(
    input_file="mcp_tragectory_cleaned.gpickle",
    output_file=None,
    remove_original_edges=False,
):
    """
    将 MCP 轨迹图中的部分关系统一为源节点上的属性：

    - parameter: has_parameter, has-parameter, hasParameters, hasParameter, 接收参数, takes_parameter, requires
    - return   : 返回, return, returns, returns result, Returns
    - type     : is_type_of, type, isTypeOf
    - description: has_description, has-description, hasDescription, Description:, description, has description
    
    参数
    ----
    input_file : str or Path
        输入 gpickle 文件路径（默认 mcp_tragectory_cleaned.gpickle）
    output_file : str or Path, optional
        输出 gpickle 文件路径；若不提供，则在同目录下加后缀 `_rel_cleaned`
    remove_original_edges : bool
        是否在写完节点属性后删除这些原始关系边（默认 False：保留边，只额外添加属性）
    """
    input_path = Path(input_file)

    # 读图（gpickle）
    with open(input_path, "rb") as f:
        g = pickle.load(f)

    # 归一化函数：忽略大小写、连字符和下划线差异
    def _norm_rel(s: str) -> str:
        return (
            str(s)
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )

    parameter_rels = {
        _norm_rel(r)
        for r in [
            "has_parameter",
            "has-parameter",
            "hasParameters",
            "hasParameter",
            "接收参数",
            "takes_parameter",
            "requires",
        ]
    }
    return_rels = {
        _norm_rel(r)
        for r in [
            "返回",
            "return",
            "returns",
            "returns result",
            "Returns",
        ]
    }
    type_rels = {
        _norm_rel(r)
        for r in [
            "is_type_of",
            "type",
            "isTypeOf",
        ]
    }
    description_rels = {
        _norm_rel(r)
        for r in [
            "has_description",
            "has-description",
            "hasDescription",
            "Description:",
            "description",
            "has description",
        ]
    }

    def _get_relation(data: dict) -> str:
        # 优先使用 'relation'，否则尝试 'type' 或 'label'
        rel = data.get("relation")
        if rel is None:
            rel = data.get("type")
        if rel is None:
            rel = data.get("label")
        return str(rel) if rel is not None else ""

    def _append_attr(node_attrs: dict, attr_name: str, value: str):
        if not value:
            return
        existing = node_attrs.get(attr_name)
        if existing is None:
            node_attrs[attr_name] = [value]
        elif isinstance(existing, list):
            if value not in existing:
                existing.append(value)
        else:
            # 已有单值，转为列表
            if existing != value:
                node_attrs[attr_name] = [existing, value]

    edges_to_remove = []
    cnt_param = cnt_ret = cnt_type = cnt_desc = 0

    # MultiDiGraph / DiGraph 兼容处理
    if isinstance(g, nx.MultiDiGraph):
        edge_iter = g.edges(keys=True, data=True)
        for u, v, k, data in edge_iter:
            rel_raw = _get_relation(data)
            rel_norm = _norm_rel(rel_raw)

            if rel_norm in parameter_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                _append_attr(g.nodes[u], "parameter", value)
                cnt_param += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v, k))
            elif rel_norm in return_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                _append_attr(g.nodes[u], "return", value)
                cnt_ret += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v, k))
            elif rel_norm in type_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                # 使用 type_info 而非 type，避免覆盖节点的原始类型属性
                _append_attr(g.nodes[u], "type_info", value)
                cnt_type += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v, k))
            elif rel_norm in description_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                _append_attr(g.nodes[u], "description", value)
                cnt_desc += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v, k))

        if remove_original_edges and edges_to_remove:
            g.remove_edges_from(edges_to_remove)
    else:
        edge_iter = g.edges(data=True)
        for u, v, data in edge_iter:
            rel_raw = _get_relation(data)
            rel_norm = _norm_rel(rel_raw)

            if rel_norm in parameter_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                _append_attr(g.nodes[u], "parameter", value)
                cnt_param += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v))
            elif rel_norm in return_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                _append_attr(g.nodes[u], "return", value)
                cnt_ret += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v))
            elif rel_norm in type_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                # 使用 type_info 而非 type，避免覆盖节点的原始类型属性
                _append_attr(g.nodes[u], "type_info", value)
                cnt_type += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v))
            elif rel_norm in description_rels:
                value = g.nodes[v].get("id", g.nodes[v].get("name", v))
                _append_attr(g.nodes[u], "description", value)
                cnt_desc += 1
                if remove_original_edges:
                    edges_to_remove.append((u, v))

        if remove_original_edges and edges_to_remove:
            g.remove_edges_from(edges_to_remove)

    print(
        f"Normalized relations -> attributes: "
        f"parameter edges: {cnt_param}, return edges: {cnt_ret}, type edges: {cnt_type}, description edges: {cnt_desc}"
    )

    # 确定输出路径
    if output_file is None:
        output_path = input_path.with_name(
            input_path.stem + "_rel_cleaned" + input_path.suffix
        )
    else:
        output_path = _ensure_parent_dir(output_file)

    with open(output_path, "wb") as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)

    print(f"Saved relation-normalized graph to: {output_path}")
