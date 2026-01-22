""" 
temporarily done for only one data format, need to be extended to other data formats
"""
import os
import sys
# 添加项目根目录到 Python 路径，以便正确导入 pipeline 模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gzip
import pandas as pd
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
from data_analyser.graph_handler import GraphInspector
from pathlib import Path
import pickle
import random
# from torch_geometric.utils import to_networkx

def read_csv_gz(file_path: str) -> pd.DataFrame:
    # read the single .csv.gz file
    # 尝试不同的分隔符，先尝试 | 分隔符（LDBC SNB BI 格式常用）
    # 注意：LDBC SNB BI 格式通常没有 header，使用 header=None
    try:
        # 先尝试 | 分隔符，没有 header
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, sep='|', header=None)
        # 如果只有一列，说明分隔符不对，尝试逗号
        if len(df.columns) == 1:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, sep=',', header=None)
    except Exception:
        # 如果 | 分隔符失败，尝试逗号
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, sep=',', header=None)
        except Exception:
            # 最后尝试自动检测（可能有 header）
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
    return df

def read_csv(file_path: str, sep: str = ',', header=0) -> pd.DataFrame:
    # read the single .csv file
    # header=0 表示第一行是列名（默认行为）
    # 如果 header=None，pandas不会将第一行作为列名，而是使用数字列名
    return pd.read_csv(file_path, sep=sep, encoding='utf-8', header=header)

def process_single_file_ldbcbi(file_path: str, folder_name: str, file_format: str) -> Tuple[List[Tuple[str, str, dict]], List[Tuple[str, str, str]]]:
    # Process the single file and return the nodes and edges (包含所有属性)
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
        else:
            print(f"警告: 不支持的文件格式 {file_format}，跳过文件 {file_path}")
            return nodes, edges
    except Exception as e:
        print(f"skip the file {file_path}, error: {e}")
        return nodes, edges
    
    # 检查 DataFrame 是否为空
    if df is None or df.empty:
        print(f"警告: 文件 {file_path} 为空，跳过")
        return nodes, edges
    
    # check if the file is a node table or an edge table
    if "_" not in folder_name:
        # the file is a node table
        node_type = folder_name
        # 检查是否有 'id' 列，或者第一列（当没有header时，列名是0）
        id_col = None
        if 'id' in df.columns:
            id_col = 'id'
        elif len(df.columns) > 0:
            # 如果没有 'id' 列，使用第一列作为 id
            id_col = df.columns[0]
        
        if id_col is not None:
            # 遍历每一行，构建节点及其所有属性
            for idx, row in df.iterrows():
                node_id_value = str(row[id_col])
                node_id = f"{node_type}:{node_id_value}"
                
                # 构建属性字典，包含所有列的值
                attributes = {}
                for col in df.columns:
                    value = row[col]
                    # 处理 NaN 值
                    if pd.isna(value):
                        attributes[col] = None
                    else:
                        # 保持原始类型，但确保可以序列化
                        attributes[col] = value
                
                nodes.append((node_id, node_type, attributes))
    else:
        # the file is an edge table
        rel_type = folder_name
        cols = df.columns.tolist()
        if len(cols) >= 2:
            # 处理不同列数的边表：
            # - 2列：静态边表，直接是源节点ID和目标节点ID（列0和列1）
            # - 3列或更多：动态边表，第一列是时间戳，第二列和第三列是源节点ID和目标节点ID（列1和列2）
            if len(cols) == 2:
                # 静态边表：使用第0列和第1列
                src_col, dst_col = cols[0], cols[1]
            else:
                # 动态边表：跳过第一列（时间戳），使用第1列和第2列
                src_col, dst_col = cols[1], cols[2]
            
            # 从关系名（文件夹名）中提取源节点类型和目标节点类型
            # 格式：SourceType_RelationName_TargetType
            # 例如：Place_isPartOf_Place -> 源类型：Place，目标类型：Place
            #      Tag_hasType_TagClass -> 源类型：Tag，目标类型：TagClass
            #      Comment_isLocatedIn_Country -> 源类型：Comment，目标类型：Country
            parts = folder_name.split('_')
            if len(parts) >= 3:
                # 源节点类型是第一部分
                src_prefix = parts[0]
                # 目标节点类型是最后一部分
                dst_prefix = parts[-1]
                
                # 处理特殊情况：Country、City、University、Company 等可能是 Place 或 Organisation 的子类型
                # 在 composite-projected-fk 格式中，这些节点实际上存储在 Place 或 Organisation 节点表中
                # 将目标节点类型映射到实际的节点类型
                type_mapping = {
                    'Country': 'Place',
                    'City': 'Place',
                    'University': 'Organisation',  # 或者可能是 Place，需要根据实际情况调整
                    'Company': 'Organisation',
                }
                if dst_prefix in type_mapping:
                    dst_prefix = type_mapping[dst_prefix]
            else:
                # 如果格式不符合预期，尝试从列名中提取
                if isinstance(src_col, (int, str)) and str(src_col).isdigit():
                    src_prefix = "node"  # 默认前缀
                else:
                    src_prefix = str(src_col).split('_')[0] if '_' in str(src_col) else "node"
                
                if isinstance(dst_col, (int, str)) and str(dst_col).isdigit():
                    dst_prefix = "node"  # 默认前缀
                else:
                    dst_prefix = str(dst_col).split('_')[0] if '_' in str(dst_col) else "node"
            
            edges = []

            src_values = df[src_col].astype(str).values
            dst_values = df[dst_col].astype(str).values
            edges = [
                (f"{src_prefix}:{src}", f"{dst_prefix}:{dst}", rel_type)
                for src, dst in zip(src_values, dst_values)
            ]
    
    return nodes, edges

def build_graph_from_data_ldbcbi(data_path: str, file_format: str) -> nx.MultiDiGraph:
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
    total_nodes_added = 0
    total_edges_added = 0
    for file_path, folder_name, fmt in file_tasks:
        try:
            nodes, edges = process_single_file_ldbcbi(file_path, folder_name, fmt)
            
            # add the nodes to the graph (包含所有属性)
            for node_id, node_type, attributes in nodes:
                # 添加节点，包含所有属性
                # 将 label 作为单独属性，同时保留所有其他属性
                node_attrs = {'label': node_type}
                node_attrs.update(attributes)
                graph.add_node(node_id, **node_attrs)
            
            # add the edges to the graph
            for src, dst, rel_type in edges:
                graph.add_edge(src, dst, label=rel_type)
            
            nodes_count = len(nodes)
            edges_count = len(edges)
            total_nodes_added += nodes_count
            total_edges_added += edges_count
            
            processed_files += 1
            
            # print progress with details
            if processed_files % 10 == 0 or processed_files == total_files:
                print(f"progress: {processed_files}/{total_files} files processed (nodes: {total_nodes_added:,}, edges: {total_edges_added:,})")
            elif nodes_count > 0 or edges_count > 0:
                # 打印有内容的文件
                print(f"  {os.path.basename(file_path)}: {nodes_count} nodes, {edges_count} edges")
                
        except Exception as e:
            print(f"error when processing the file {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()
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
            executor.submit(process_single_file_ldbcbi, file_path, folder_name, file_format): file_path
            for file_path, folder_name, file_format in file_tasks
        }
        
        # process the completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                nodes, edges = future.result()
                
                # add the nodes to the graph (包含所有属性)
                for node_id, node_type, attributes in nodes:
                    # 添加节点，包含所有属性
                    # 将 label 作为单独属性，同时保留所有其他属性
                    node_attrs = {'label': node_type}
                    node_attrs.update(attributes)
                    graph.add_node(node_id, **node_attrs)
                
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


def is_camel_case(filename: str) -> bool:
    """
    判断文件名是否为驼峰命名法（关系文件）
    如果文件名包含大写字母（除了首字母），则认为是驼峰命名
    """
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    # 检查是否有大写字母（除了首字母）
    return any(c.isupper() for c in name_without_ext[1:])


def parse_relation_filename(filename: str) -> Tuple[str, str, str]:
    """
    解析关系文件名，提取源节点类型、关系名和目标节点类型
    
    例如：
    - AccountTransferAccount -> (Account, Transfer, Account)
    - PersonInvestCompany -> (Person, Invest, Company)
    - CompanyOwnAccount -> (Company, Own, Account)
    
    Args:
        filename: 文件名（不含扩展名）
        
    Returns:
        (源节点类型, 关系名, 目标节点类型)
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # 找到所有大写字母的位置
    uppercase_positions = [i for i, c in enumerate(name_without_ext) if c.isupper()]
    
    if len(uppercase_positions) < 2:
        # 如果只有一个大写字母（首字母），无法解析
        return None, None, None
    
    # 第一个大写字母位置是0（首字母）
    # 找到第二个大写字母的位置，这通常是源节点类型和关系名的分界
    first_break = uppercase_positions[1] if len(uppercase_positions) > 1 else len(name_without_ext)
    
    # 源节点类型：从开头到第一个分界点
    src_type = name_without_ext[:first_break]
    
    # 找到最后一个大写字母的位置，这通常是关系名和目标节点类型的分界
    if len(uppercase_positions) >= 3:
        # 有多个大写字母，最后一个分界点是倒数第二个大写字母
        last_break = uppercase_positions[-1]
        # 关系名：从第一个分界点到最后一个分界点
        rel_name = name_without_ext[first_break:last_break]
        # 目标节点类型：从最后一个分界点到结尾
        dst_type = name_without_ext[last_break:]
    else:
        # 只有两个大写字母，说明是 SourceTarget 格式
        # 这种情况下，中间部分可能是关系名，但通常关系名会被省略
        # 例如：AccountAccount 可能是 Account -> Account 的自环关系
        # 我们假设第二个大写字母开始是目标节点类型
        last_break = uppercase_positions[1]
        rel_name = name_without_ext[first_break:last_break] if first_break < last_break else name_without_ext[first_break:]
        dst_type = name_without_ext[last_break:] if last_break < len(name_without_ext) else src_type
    
    return src_type, rel_name, dst_type


def process_node_file_ldbcfin(file_path: str) -> List[Tuple[str, str, dict]]:
    """
    处理节点文件，返回节点列表（包含所有属性）
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        节点列表，格式为 [(node_id, node_type, attributes_dict), ...]
        其中 attributes_dict 包含节点的所有属性（包括ID列）
    """
    nodes = []
    try:
        # 读取CSV文件，使用 | 分隔符
        # 注意：LDBC FinBench 的 CSV 文件第一行是列名，所以使用 header=0（默认值）
        df = read_csv(file_path, sep='|', header=0)
        
        if df.empty:
            return nodes
        
        # 节点类型是文件名（不含扩展名）
        node_type = os.path.splitext(os.path.basename(file_path))[0]
        
        # 查找ID列：通常是第一列，或者包含'id'的列（不区分大小写）
        id_col = None
        for col in df.columns:
            if 'id' in str(col).lower():
                id_col = col
                break
        
        if id_col is None and len(df.columns) > 0:
            # 如果没有找到id列，使用第一列
            id_col = df.columns[0]
        
        if id_col is not None:
            # 遍历每一行，构建节点及其属性
            for idx, row in df.iterrows():
                node_id_value = str(row[id_col])
                node_id = f"{node_type}:{node_id_value}"
                
                # 构建属性字典，包含所有列的值
                attributes = {}
                for col in df.columns:
                    value = row[col]
                    # 处理 NaN 值
                    if pd.isna(value):
                        attributes[col] = None
                    else:
                        # 保持原始类型，但确保可以序列化
                        attributes[col] = value
                
                nodes.append((node_id, node_type, attributes))
            
    except Exception as e:
        print(f"处理节点文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return nodes


def process_relation_file_ldbcfin(file_path: str) -> List[Tuple[str, str, str]]:
    """
    处理关系文件，返回边列表
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        边列表，格式为 [(src_id, dst_id, rel_type), ...]
    """
    edges = []
    try:
        # 读取CSV文件，使用 | 分隔符
        df = read_csv(file_path, sep='|')
        
        if df.empty:
            return edges
        
        # 解析文件名获取关系信息
        filename = os.path.basename(file_path)
        src_type, rel_name, dst_type = parse_relation_filename(filename)
        
        if src_type is None or dst_type is None:
            print(f"警告: 无法解析关系文件名 {filename}，跳过")
            return edges
        
        # 关系类型使用完整的关系名
        rel_type = f"{src_type}_{rel_name}_{dst_type}" if rel_name else f"{src_type}_to_{dst_type}"
        
        # 查找源节点ID列和目标节点ID列
        # 常见的列名模式：
        # - fromId, toId
        # - srcId, dstId
        # - sourceId, targetId
        # - 或者特定类型：如 investorId, companyId
        
        src_col = None
        dst_col = None
        
        # 先尝试常见的列名
        for col in df.columns:
            col_lower = str(col).lower()
            if 'from' in col_lower or 'src' in col_lower or 'source' in col_lower:
                src_col = col
            elif 'to' in col_lower or 'dst' in col_lower or 'target' in col_lower:
                dst_col = col
        
        # 如果没找到，尝试根据节点类型查找
        if src_col is None:
            for col in df.columns:
                if src_type.lower() in str(col).lower() and 'id' in str(col).lower():
                    src_col = col
                    break
        
        if dst_col is None:
            for col in df.columns:
                if dst_type.lower() in str(col).lower() and 'id' in str(col).lower():
                    dst_col = col
                    break
        
        # 如果还是没找到，使用前两列
        if src_col is None and len(df.columns) >= 1:
            src_col = df.columns[0]
        if dst_col is None and len(df.columns) >= 2:
            dst_col = df.columns[1]
        
        if src_col is None or dst_col is None:
            print(f"警告: 无法找到源节点或目标节点ID列，文件 {filename}，列: {list(df.columns)}")
            return edges
        
        # 构建边
        src_values = df[src_col].astype(str).values
        dst_values = df[dst_col].astype(str).values
        
        edges = [
            (f"{src_type}:{src}", f"{dst_type}:{dst}", rel_type)
            for src, dst in zip(src_values, dst_values)
        ]
        
    except Exception as e:
        print(f"处理关系文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return edges


def build_graph_from_data_ldbcfin(data_path: str, file_format: str = ".csv") -> nx.MultiDiGraph:
    """
    从 LDBC SNB FinBench 数据构建图
    
    数据格式说明：
    - 驼峰命名法的文件（如 AccountTransferAccount.csv）是关系文件
    - 其他文件（如 Account.csv）是节点文件
    - 所有文件都在同一个目录下
    
    Args:
        data_path: 数据目录路径
        file_format: 文件格式（默认 ".csv"）
        
    Returns:
        nx.MultiDiGraph: 构建好的图
    """
    import time
    overall_start = time.time()
    
    graph = nx.MultiDiGraph()
    print(f"从 {data_path} 加载图数据...")
    
    # 收集所有文件
    all_files = []
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            if file.endswith(file_format):
                file_path = os.path.join(data_path, file)
                all_files.append(file_path)
    else:
        print(f"错误: {data_path} 不是一个有效的目录")
        return graph
    
    total_files = len(all_files)
    print(f"找到 {total_files} 个文件，开始处理...")
    
    # 先处理节点文件，再处理关系文件
    node_files = []
    relation_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if is_camel_case(filename):
            relation_files.append(file_path)
        else:
            node_files.append(file_path)
    
    print(f"节点文件: {len(node_files)} 个，关系文件: {len(relation_files)} 个")
    
    # 处理节点文件
    total_nodes_added = 0
    processed_files = 0
    
    for file_path in node_files:
        try:
            nodes = process_node_file_ldbcfin(file_path)
            for node_id, node_type, attributes in nodes:
                # 添加节点，包含所有属性
                # 将 label 作为单独属性，同时保留所有其他属性
                node_attrs = {'label': node_type}
                node_attrs.update(attributes)
                graph.add_node(node_id, **node_attrs)
            
            nodes_count = len(nodes)
            total_nodes_added += nodes_count
            processed_files += 1
            
            if nodes_count > 0:
                print(f"  {os.path.basename(file_path)}: {nodes_count} 个节点")
                
        except Exception as e:
            print(f"处理节点文件 {os.path.basename(file_path)} 时出错: {e}")
            processed_files += 1
    
    # 处理关系文件
    total_edges_added = 0
    
    for file_path in relation_files:
        try:
            edges = process_relation_file_ldbcfin(file_path)
            for src, dst, rel_type in edges:
                graph.add_edge(src, dst, label=rel_type)
            
            edges_count = len(edges)
            total_edges_added += edges_count
            processed_files += 1
            
            if edges_count > 0:
                print(f"  {os.path.basename(file_path)}: {edges_count} 条边")
                
        except Exception as e:
            print(f"处理关系文件 {os.path.basename(file_path)} 时出错: {e}")
            import traceback
            traceback.print_exc()
            processed_files += 1
    
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("图加载成功！")
    print(f"{'='*60}")
    print(f"总耗时: {overall_elapsed:.2f} 秒")
    print(f"处理文件数: {processed_files}/{total_files}")
    print(f"节点数量: {graph.number_of_nodes():,}")
    print(f"边数量: {graph.number_of_edges():,}")
    print(f"{'='*60}\n")
    
    return graph


