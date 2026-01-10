"""
测试 Aggregation_Rank_Template 和 Shared_Neighbor_Template
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from operator_no_neural import load_graph, Scan, Filter, Project, TraverseWithPath
from general_query import Aggregation_Rank_Template, Shared_Neighbor_Template
import pickle
import networkx as nx
from collections import Counter

# 加载图
graph_path = "/home/ylivm/fei_work/NGDB_Benchmark/data_gen/graph_gen/graph_buffer/Primekg_noise_20251231_083857.gpickle"
print("=" * 80)
print("加载图文件...")
G = load_graph(graph_path)

# 分析图结构，获取节点类型和边类型
print("\n" + "=" * 80)
print("分析图结构...")

node_types = Counter()
edge_types = Counter()

for node_id in G.nodes():
    node_data = G.nodes[node_id]
    node_type = node_data.get('type') or node_data.get('label') or node_data.get('node_type')
    if node_type:
        node_types[node_type] += 1

for u, v, edge_data in G.edges(data=True):
    edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
    if edge_type:
        edge_types[edge_type] += 1

print(f"\n节点类型（前10个）:")
for node_type, count in node_types.most_common(10):
    print(f"  {node_type}: {count}")

print(f"\n边类型（前10个）:")
for edge_type, count in edge_types.most_common(10):
    print(f"  {edge_type}: {count}")

# 采样节点和边类型用于测试
def sample_node_by_type(node_type):
    """采样一个指定类型的节点"""
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        ntype = node_data.get('type') or node_data.get('label') or node_data.get('node_type')
        if ntype == node_type:
            return node_id
    return None

def get_node_attr_value(node_id, attr_name):
    """获取节点属性值"""
    node_data = G.nodes[node_id]
    return node_data.get(attr_name)

# 创建参数类
class Params:
    pass

print("\n" + "=" * 80)
print("测试 1: Aggregation_Rank_Template")
print("=" * 80)

# 根据实际图结构，选择合理的路径组合
# 实际路径：gene/protein -> disease_protein -> disease (出边)
#           gene/protein -> drug_protein -> drug (出边)
# 所以我们需要从 gene/protein 开始，或者使用反向遍历
# 这里我们使用：disease -> (IN) disease_protein -> gene/protein -> (OUT) drug_protein -> drug
params1 = Params()
params1.T1 = "disease"  # 起始：疾病
params1.P1 = "name"
params1.E1 = "disease_protein"  # 疾病关联基因（使用IN方向，因为边是 gene/protein -> disease）
params1.T2 = "gene/protein"  # 中间：基因/蛋白质
params1.E2 = "drug_protein"  # 药物靶向蛋白质（使用IN方向，因为边是 drug -> gene/protein）
params1.T3 = "drug"  # 目标：药物
params1.direction1 = "IN"  # 第一跳使用IN方向（从disease反向到gene/protein）
params1.direction2 = "IN"  # 第二跳使用IN方向（从gene/protein反向到drug）

# 采样一个疾病节点，确保它有完整的路径：disease -> gene/protein -> drug
sample_node = None
for node_id in G.nodes():
    node_data = G.nodes[node_id]
    ntype = node_data.get('type') or node_data.get('label') or node_data.get('node_type')
    if ntype == "disease":
        # 检查是否有 disease_protein 入边
        proteins = set()
        for u, v, edge_data in G.in_edges(node_id, data=True):
            edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
            if edge_type == 'disease_protein':
                proteins.add(u)
        
        if proteins:
            # 检查这些蛋白质是否有 drug_protein 入边指向 drug
            for protein_id in list(proteins)[:10]:  # 只检查前10个
                for u, v, edge_data in G.in_edges(protein_id, data=True):
                    edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
                    if edge_type == 'drug_protein':
                        u_data = G.nodes[u]
                        u_type = u_data.get('type') or u_data.get('label') or u_data.get('node_type')
                        if u_type == 'drug':
                            sample_node = node_id
                            break
                if sample_node:
                    break
        if sample_node:
            break

if sample_node:
    name_val = get_node_attr_value(sample_node, "name")
    if name_val:
        params1.V1 = name_val
        params1.P1 = "name"
    else:
        # 尝试其他属性
        for attr in ["id", "identifier", "label"]:
            val = get_node_attr_value(sample_node, attr)
            if val:
                params1.V1 = val
                params1.P1 = attr
                break
        if not hasattr(params1, 'V1') or not params1.V1:
            # 如果都没有，使用节点ID，但需要调整过滤方式
            params1.V1 = str(sample_node)
            params1.P1 = "id"  # 假设有id属性
    
    print(f"\n参数配置:")
    print(f"  T1 (起始类型): {params1.T1}")
    print(f"  P1 (属性名): {params1.P1}")
    print(f"  V1 (属性值): {params1.V1[:100] if len(str(params1.V1)) > 100 else params1.V1}")
    print(f"  E1 (第一跳边): {params1.E1}")
    print(f"  T2 (中间类型): {params1.T2}")
    print(f"  E2 (第二跳边): {params1.E2}")
    print(f"  T3 (目标类型): {params1.T3}")
    
    try:
        # 添加调试信息
        from operator_no_neural import Scan, Filter, Project, TraverseWithPath
        start = Filter(Scan(type=params1.T1), {params1.P1: params1.V1})
        print(f"\n  调试: 起始节点数量: {len(start)}")
        if start:
            direction1 = getattr(params1, 'direction1', 'OUT')
            path_mid = Project(start, edge=params1.E1, direction=direction1)
            print(f"  调试: 第一跳后节点数量: {len(path_mid)} (方向: {direction1})")
            if path_mid:
                direction2 = getattr(params1, 'direction2', 'OUT')
                paths_b_to_c = TraverseWithPath(path_mid, edge=params1.E2, target_type=params1.T3, direction=direction2)
                print(f"  调试: 第二跳后路径数量: {len(paths_b_to_c)} (方向: {direction2})")
        
        result1 = Aggregation_Rank_Template(params1)
        print(f"\n✅ 执行成功!")
        print(f"结果数量: {len(result1)}")
        if result1:
            print(f"\n前10个结果:")
            for i, item in enumerate(result1[:10], 1):
                print(f"  {i}. Key: {item['key']}, Value: {item['value']}")
        else:
            print("  (无结果)")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ 无法找到类型为 {params1.T1} 的节点")

print("\n" + "=" * 80)
print("测试 2: Shared_Neighbor_Template")
print("=" * 80)

# 使用更合理的参数组合
# 实际路径：drug -> drug_protein -> gene/protein (OUT方向)
#           gene/protein -> disease_protein -> disease (OUT方向)
# 查找与某药物靶向相同蛋白质的其他疾病
params2 = Params()
params2.T_A = "drug"  # 起始：药物
params2.P_A = "name"
params2.E1 = "drug_protein"  # 药物靶向蛋白质（OUT方向：drug -> gene/protein）
params2.T_B = "gene/protein"  # 中间：基因/蛋白质
params2.E2 = "disease_protein"  # 疾病关联蛋白质（OUT方向：gene/protein -> disease）
params2.T_C = "disease"  # 目标：疾病
params2.direction1 = "OUT"  # 第一跳使用OUT方向
params2.direction2 = "OUT"  # 第二跳使用OUT方向

# 采样一个药物节点
sample_node_a = sample_node_by_type("drug")
if sample_node_a:
    name_val = get_node_attr_value(sample_node_a, "name")
    if name_val:
        params2.V_A = name_val
    else:
        for attr in ["id", "identifier", "label"]:
            val = get_node_attr_value(sample_node_a, attr)
            if val:
                params2.V_A = val
                params2.P_A = attr
                break
        if not hasattr(params2, 'V_A') or not params2.V_A:
            params2.V_A = str(sample_node_a)
    
    print(f"\n参数配置:")
    print(f"  T_A (起始类型): {params2.T_A}")
    print(f"  P_A (属性名): {params2.P_A}")
    print(f"  V_A (属性值): {params2.V_A[:100] if len(str(params2.V_A)) > 100 else params2.V_A}")
    print(f"  E1 (A->B边): {params2.E1}")
    print(f"  T_B (中间类型): {params2.T_B}")
    print(f"  E2 (B->C边): {params2.E2}")
    print(f"  T_C (目标类型): {params2.T_C}")
    
    try:
        # 添加调试信息
        from operator_no_neural import Scan, Filter, Project, TraverseWithPath
        node_a = Filter(Scan(type=params2.T_A), {params2.P_A: params2.V_A})
        print(f"\n  调试: 起始节点数量: {len(node_a)}")
        if node_a:
            direction1 = getattr(params2, 'direction1', 'OUT')
            nodes_b = Project(node_a, edge=params2.E1, direction=direction1)
            print(f"  调试: 第一跳后节点数量: {len(nodes_b)} (方向: {direction1})")
            if nodes_b:
                direction2 = getattr(params2, 'direction2', 'OUT')
                paths_b_to_c = TraverseWithPath(nodes_b, edge=params2.E2, target_type=params2.T_C, direction=direction2)
                print(f"  调试: 第二跳后路径数量: {len(paths_b_to_c)} (方向: {direction2})")
                filtered_paths = [p for p in paths_b_to_c if p["target_node"] not in node_a]
                print(f"  调试: 过滤后路径数量: {len(filtered_paths)}")
        
        result2 = Shared_Neighbor_Template(params2)
        print(f"\n✅ 执行成功!")
        print(f"结果数量: {len(result2)}")
        if result2:
            print(f"\n前10个结果:")
            for i, item in enumerate(result2[:10], 1):
                print(f"  {i}. Key: {item['key']}, Value: {item['value']}")
        else:
            print("  (无结果)")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ 无法找到类型为 {params2.T_A} 的节点")

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)

