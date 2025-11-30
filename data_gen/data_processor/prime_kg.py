import pandas as pd
import json
import io


file1_path = "../gnd_dataset/PrimeKG/kg.csv"
file2_path = "../gnd_dataset/PrimeKG/drug_features.csv"

def process_knowledge_graph_enhanced(df_list):
    # 1. 合并与去重
    full_df = pd.concat(df_list, ignore_index=True)
    full_df.drop_duplicates(inplace=True)
    
    # 2. 提取节点 (Nodes)
    nodes_x = full_df[['x_id', 'x_type', 'x_name', 'x_source']].rename(columns={
        'x_id': 'id', 'x_type': 'type', 'x_name': 'name', 'x_source': 'source'
    })
    nodes_y = full_df[['y_id', 'y_type', 'y_name', 'y_source']].rename(columns={
        'y_id': 'id', 'y_type': 'type', 'y_name': 'name', 'y_source': 'source'
    })
    all_nodes = pd.concat([nodes_x, nodes_y]).drop_duplicates(subset=['id']).reset_index(drop=True)
    
    # 3. 提取边 (Edges)
    edges = full_df[['x_id', 'y_id', 'relation', 'display_relation']].rename(columns={
        'x_id': 'source',
        'y_id': 'target',
        'relation': 'type',
        'display_relation': 'label'
    })
    
    # --- 新增功能: 统计分析 ---
    
    # 统计节点类型分布
    node_type_stats = all_nodes['type'].value_counts().to_dict()
    
    # 统计关系类型分布 (这里统计的是 display_relation, 你也可以改为 relation)
    edge_type_stats = edges['label'].value_counts().to_dict()
    
    # 统计关系的大类分布 (relation 字段)
    edge_category_stats = edges['type'].value_counts().to_dict()

    # 4. 构建最终结构
    graph_data = {
        "meta_data": {
            "summary": {
                "total_nodes": len(all_nodes),
                "total_edges": len(edges)
            },
            "schema": {
                "node_types": node_type_stats,
                "edge_types": edge_category_stats,
                "edge_labels": edge_type_stats
            }
        },
        "nodes": all_nodes.to_dict(orient='records'),
        "edges": edges.to_dict(orient='records')
    }
    
    return graph_data

# # --- 执行 ---
# df1 = pd.read_csv(file1_path)
# df2 = pd.read_csv(file2_path)

# result = process_knowledge_graph_enhanced([df1, df2])

# # --- 输出结果 ---

# print("Processing Complete.")
# print("-" * 30)
# print(f"Total Nodes: {result['meta_data']['summary']['total_nodes']}")
# print(f"Total Edges: {result['meta_data']['summary']['total_edges']}")

# print("\n[Node Types Found]:")
# for n_type, count in result['meta_data']['schema']['node_types'].items():
#     print(f"  - {n_type}: {count}")

# print("\n[Connection Types Found (Relation)]:dict")
# for e_type, count in result['meta_data']['schema']['edge_types'].items():
#     print(f"  - {e_type}: {count}")

# print("\n[Connection Labels (Display)]: " )
# for label, count in result['meta_data']['schema']['edge_labels'].items():
#     print(f"  - {label}: {count}")

# # 保存文件
# with open('kg_data_analyzed.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, indent=4, ensure_ascii=False)

