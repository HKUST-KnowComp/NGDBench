""" 
temporarily done for only one data format, need to be extended to other data formats
"""
import os
import gzip
import pandas as pd
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
from graph_handler import GraphInspector
from pathlib import Path
import pickle
import random

def read_csv_gz(file_path: str) -> pd.DataFrame:
    # read the single .csv.gz file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f)

def process_single_file(file_path: str, folder_name: str, file_format: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    # Process the single file and return the nodes and edges
    import time
    start_time = time.time()
    file_name = os.path.basename(file_path)
    # print(f"📖 开始处理: {file_name}")
    
    nodes = []
    edges = []
    
    try:
        if file_format == ".csv.gz":
            df = read_csv_gz(file_path)
        # elif file_format == ".json":
        #     df = read_json(file_path)
        # elif file_format == ".jsonl":
        #     df = read_jsonl(file_path)
        # elif file_format == ".parquet":
        #     df = read_parquet(file_path)
        # elif file_format == ".feather":
        #     df = read_feather(file_path)
    except Exception as e:
        print(f"skip the file {file_path}, error: {e}")
        return nodes, edges
    
    # check if the file is a node table or an edge table
    if "_" not in folder_name:
        # the file is a node table
        node_type = folder_name
        if 'id' in df.columns:
            # for nid in df['id'].astype(str):
            #     nodes.append((f"{node_type}:{nid}", node_type))
            # 使用向量化操作，比 iterrows 快得多
            node_ids = df['id'].astype(str).values
            nodes = [(f"{node_type}:{nid}", node_type) for nid in node_ids]
    else:
        # the file is an edge table
        rel_type = folder_name
        cols = df.columns.tolist()
        if len(cols) >= 2:
            src_col, dst_col = cols[0], cols[1]
            src_prefix = src_col.split('_')[0]
            dst_prefix = dst_col.split('_')[0]
            
            edges = []

            src_values = df[src_col].values
            dst_values = df[dst_col].values
            edges = [
                (f"{src_prefix}:{src}", f"{dst_prefix}:{dst}", rel_type)
                for src, dst in zip(src_values, dst_values)
            ]
    
    return nodes, edges

def build_graph_from_data(data_path: str, file_format: str) -> nx.MultiDiGraph:
    """
    Build the graph from the data path (sequential processing)
    Suitable for single file or small dataset scenarios
    
    Args:
        data_path: Path to the data directory
        file_format: File format to process (e.g., ".csv.gz", ".csv")
        
    Returns:
        nx.MultiDiGraph: The constructed graph
    """
    import time
    overall_start = time.time()
    
    graph = nx.MultiDiGraph()
    print(f"loading graph data from {data_path}...")
    
    # collect all the files to be processed
    file_tasks = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith(file_format):
                continue
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            file_tasks.append((file_path, folder_name, file_format))
    
    total_files = len(file_tasks)
    print(f"found {total_files} files, start to process...")
    
    # process files sequentially
    processed_files = 0
    for file_path, folder_name, fmt in file_tasks:
        try:
            nodes, edges = process_single_file(file_path, folder_name, fmt)
            
            # add the nodes to the graph
            for node_id, node_type in nodes:
                graph.add_node(node_id, label=node_type)
            
            # add the edges to the graph
            for src, dst, rel_type in edges:
                graph.add_edge(src, dst, label=rel_type)
            
            processed_files += 1
            
            # print progress
            if processed_files % 10 == 0 or processed_files == total_files:
                print(f"progress: {processed_files}/{total_files} files processed")
                
        except Exception as e:
            print(f"error when processing the file {os.path.basename(file_path)}: {e}")
            processed_files += 1
    
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print("graph loaded successfully!")
    print(f"{'='*60}")
    print(f"total time: {overall_elapsed:.2f} seconds")
    print(f"processed files: {processed_files}/{total_files}")
    print(f"number of nodes: {graph.number_of_nodes():,}")
    print(f"number of edges: {graph.number_of_edges():,}")
    print(f"{'='*60}\n")
    return graph

def build_graph_from_data_threaded(data_path: str, file_format: str, max_workers: int = 4) -> nx.MultiDiGraph:
    """
    Build the graph from the data path (parallel processing with thread pool)
    Suitable for large dataset scenarios with multiple files
    
    Args:
        data_path: Path to the data directory
        file_format: File format to process (e.g., ".csv.gz", ".csv")
        max_workers: Maximum number of worker threads
        
    Returns:
        nx.MultiDiGraph: The constructed graph
    """
    import time
    overall_start = time.time()
    
    graph = nx.MultiDiGraph()
    print(f"loading graph data from {data_path}...")
    
    # collect all the files to be processed
    file_tasks = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith(file_format):
                continue
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            file_tasks.append((file_path, folder_name, file_format))
    
    total_files = len(file_tasks)
    print(f"found {total_files} files, start to process...")
    
    # use the thread pool to process the files
    processed_files = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, file_path, folder_name, file_format): file_path
            for file_path, folder_name, file_format in file_tasks
        }
        
        # process the completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                nodes, edges = future.result()
                
                # add the nodes to the graph
                for node_id, node_type in nodes:
                    graph.add_node(node_id, label=node_type)
                
                # add the edges to the graph
                for src, dst, rel_type in edges:
                    graph.add_edge(src, dst, label=rel_type)
                
                processed_files += 1
                    
            except Exception as e:
                print(f"error when processing the file {os.path.basename(file_path)}: {e}")
                processed_files += 1

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print("graph loaded successfully!")
    print(f"{'='*60}")
    print(f"total time: {overall_elapsed:.2f} seconds")
    print(f"processed files: {processed_files}/{total_files}")
    print(f"number of nodes: {graph.number_of_nodes():,}")
    print(f"number of edges: {graph.number_of_edges():,}")
    print(f"{'='*60}\n")
    return graph

def save_graph(graph: nx.MultiDiGraph, path: str):
    # save the graph to the file
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    print(f"graph saved to {path}")

def load_graph(path: str) -> nx.MultiDiGraph:
    # load the graph from the file
    with open(path, "rb") as f:
        graph = pickle.load(f)
    print(f"graph loaded from {path}")
    return graph

def build_graph_from_kg_csv(csv_path: str, save_path: str = None) -> nx.MultiDiGraph:
    """
    从 kg.csv 格式的文件构建图并可选保存为 .gpickle 格式
    
    CSV 格式要求包含以下列：
    - relation: 关系类型
    - x_id, x_type, x_name, x_source: 源节点信息
    - y_id, y_type, y_name, y_source: 目标节点信息
    - version: 可选，版本信息
    
    Args:
        csv_path: kg.csv 文件路径
        save_path: 可选，保存图的路径（.gpickle 格式）
        
    Returns:
        nx.MultiDiGraph: 构建好的图
    """
    import time
    start_time = time.time()
    
    print(f"开始从 {csv_path} 加载知识图谱数据...")
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 创建有向多重图
    graph = nx.MultiDiGraph()
    
    # 处理每一行数据
    for idx, row in df.iterrows():
        # 提取源节点信息
        x_id = str(row['x_id'])
        x_type = row['x_type']
        x_name = row.get('x_name', '')
        x_source = row.get('x_source', '')
        
        # 提取目标节点信息
        y_id = str(row['y_id'])
        y_type = row['y_type']
        y_name = row.get('y_name', '')
        y_source = row.get('y_source', '')
        
        # 提取关系信息
        relation = row['relation']
        display_relation = row.get('display_relation', relation)
        version = row.get('version', None)  # 提取版本信息（如果存在）
        
        # 构建节点ID（格式：类型:ID）
        x_node_id = f"{x_type}:{x_id}"
        y_node_id = f"{y_type}:{y_id}"
        
        # 添加源节点（如果不存在）
        if not graph.has_node(x_node_id):
            graph.add_node(
                x_node_id,
                label=x_type,
                node_type=x_type,
                node_id=x_id,
                name=x_name,
                source=x_source
            )
        
        # 添加目标节点（如果不存在）
        if not graph.has_node(y_node_id):
            graph.add_node(
                y_node_id,
                label=y_type,
                node_type=y_type,
                node_id=y_id,
                name=y_name,
                source=y_source
            )
        
        # 添加边（包含版本信息）
        edge_attrs = {
            'label': relation,
            'relation': relation,
            'display_relation': display_relation
        }
        if version is not None:
            edge_attrs['version'] = version
        
        graph.add_edge(
            x_node_id,
            y_node_id,
            **edge_attrs
        )
        
        # # 进度显示
        # if (idx + 1) % 1000 == 0:
        #     print(f"已处理 {idx + 1}/{len(df)} 条记录...")
    
    elapsed_time = time.time() - start_time
    
    # 显示统计信息
    print(f"\n{'='*60}")
    print("知识图谱加载完成！")
    print(f"{'='*60}")
    print(f"处理时间: {elapsed_time:.2f} 秒")
    print(f"处理记录数: {len(df):,}")
    print(f"节点数量: {graph.number_of_nodes():,}")
    print(f"边数量: {graph.number_of_edges():,}")
    print(f"{'='*60}\n")
    
    # 如果指定了保存路径，则保存图
    if save_path:
        save_graph(graph, save_path)
    
    return graph

if __name__ == "__main__":
    data_path = "/home/ylivm/ngdb/ngdb_benchmark/data_gen/perturbed_dataset/ldbc_snb_bi_2510280002/out-sf1/graphs/csv/bi/composite-projected-fk/initial_snapshot"
    graph_name = "ldbc_snb_bi_2510280002"
    file_format = ".csv.gz"
    graph_path = Path(f"pipeline/data_analyser/buffer/{graph_name}.gpickle")
    if graph_path.exists():
        graph = load_graph(graph_path)
        print(f"loaded graph from {graph_path}")
    else:
        graph = build_graph_from_data_threaded(data_path, file_format)
        save_graph(graph, graph_path)
    
    # 创建图检查器
    graph_inspector = GraphInspector(graph)
    
    # 显示图的统计信息
    print("\n" + "="*60)
    print("【图的整体统计信息】")
    print("="*60)
    graph_inspector.summary()
    
    # 随机采样一些节点进行测试
    all_nodes = list(graph.nodes())
    sample_size = min(5, len(all_nodes))  # 采样5个节点，如果节点数少于5则全部采样
    sampled_nodes = random.sample(all_nodes, sample_size)
    
    print("\n" + "="*60)
    print(f"【随机采样 {sample_size} 个节点进行测试】")
    print("="*60)
    
    for i, node in enumerate(sampled_nodes, 1):
        print(f"\n{'─'*60}")
        print(f"📍 节点 {i}: {node}")
        print(f"{'─'*60}")
        
        # 测试度数相关功能
        in_deg = graph_inspector.in_degree(node)
        out_deg = graph_inspector.out_degree(node)
        total_deg = graph_inspector.degree(node)
        print(f"📥 入度: {in_deg}")
        print(f"📤 出度: {out_deg}")
        print(f"📊 总度数: {total_deg}")
        
        # 测试按关系统计出度
        rel_outdegree = graph_inspector.out_degree_by_relation(node)
        if rel_outdegree:
            print(f"\n🔗 按关系类型统计出度:")                                                                                                                                          
            for rel, count in sorted(rel_outdegree.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {rel}: {count}")
        else:
            print(f"\n🔗 该节点没有出边")
        
        # 测试入边和出边
        in_edges = graph_inspector.in_edges(node)
        out_edges = graph_inspector.out_edges(node)
        
        # 显示部分入边示例（最多显示3条）
        if in_edges:
            print(f"\n📥 入边示例 (共 {len(in_edges)} 条，显示前3条):")
            for src, dst, data in in_edges[:3]:
                print(f"  {src} --[{data.get('label', 'N/A')}]--> {dst}")
        
        # 显示部分出边示例（最多显示3条）
        if out_edges:
            print(f"\n📤 出边示例 (共 {len(out_edges)} 条，显示前3条):")
            for src, dst, data in out_edges[:3]:
                print(f"  {src} --[{data.get('label', 'N/A')}]--> {dst}")
        
        # 如果有关系类型，测试按关系查询边
        if rel_outdegree:
            # 选择出度最高的关系类型
            top_relation = max(rel_outdegree.items(), key=lambda x: x[1])[0]
            edges_of_relation = graph_inspector.edges_by_relation(node, top_relation)
            print(f"\n🎯 关系类型 '{top_relation}' 的边 (共 {len(edges_of_relation)} 条，显示前3条):")
            for src, dst in edges_of_relation[:3]:
                print(f"  {src} --> {dst}")
    
    print("\n" + "="*60)
    print("✅ GraphInspector 功能测试完成！")
    print("="*60)