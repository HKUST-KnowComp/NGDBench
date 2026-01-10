"""
测试基本算子和查询模板
"""

import sys
from pathlib import Path
import pickle
import networkx as nx
from collections import Counter
import random

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from operator_no_neural import load_graph, get_graph, Scan, Filter, Project
from general_query import Direct_Relation_Template, Chain_Hop_Template


class Params:
    """简单的参数类，用于存储查询参数"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def sample_graph_statistics():
    """从图中采样统计信息，用于生成测试参数"""
    G = get_graph()
    
    # 统计节点类型
    node_types = Counter()
    node_attrs_samples = {}  # {type: {attr: [values]}}
    
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get('type') or node_data.get('label') or node_data.get('node_type')
        if node_type:
            node_types[node_type] += 1
            if node_type not in node_attrs_samples:
                node_attrs_samples[node_type] = {}
            
            # 采样属性值
            for attr, value in node_data.items():
                if attr not in ['type', 'label', 'node_type']:
                    if node_type not in node_attrs_samples:
                        node_attrs_samples[node_type] = {}
                    if attr not in node_attrs_samples[node_type]:
                        node_attrs_samples[node_type][attr] = []
                    if len(node_attrs_samples[node_type][attr]) < 10:  # 最多保存10个样本
                        node_attrs_samples[node_type][attr].append(value)
    
    # 统计边类型
    edge_types = Counter()
    edge_attrs_samples = {}  # {edge_type: {attr: [values]}}
    
    for source, target, edge_data in G.edges(data=True):
        edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
        if edge_type:
            edge_types[edge_type] += 1
            if edge_type not in edge_attrs_samples:
                edge_attrs_samples[edge_type] = {}
            
            # 采样属性值
            for attr, value in edge_data.items():
                if attr not in ['type', 'label', 'edge_type', 'relation']:
                    if edge_type not in edge_attrs_samples:
                        edge_attrs_samples[edge_type] = {}
                    if attr not in edge_attrs_samples[edge_type]:
                        edge_attrs_samples[edge_type][attr] = []
                    if len(edge_attrs_samples[edge_type][attr]) < 10:
                        edge_attrs_samples[edge_type][attr].append(value)
    
    return {
        'node_types': dict(node_types),
        'node_attrs_samples': node_attrs_samples,
        'edge_types': dict(edge_types),
        'edge_attrs_samples': edge_attrs_samples
    }


def sample_params_for_direct_relation(stats):
    """为 Direct_Relation_Template 采样参数 - 使用实际存在的边连接"""
    G = get_graph()
    
    # 从实际存在的边中采样，确保连接关系真实存在
    # 采样多个边，找到有name属性的起始节点
    candidate_edges = []
    for source, target, edge_data in G.edges(data=True):
        source_data = G.nodes[source]
        target_data = G.nodes[target]
        
        # 确保起始节点有name属性
        if 'name' in source_data:
            edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
            source_type = source_data.get('type') or source_data.get('label') or source_data.get('node_type')
            target_type = target_data.get('type') or target_data.get('label') or target_data.get('node_type')
            
            if edge_type and source_type and target_type:
                candidate_edges.append({
                    'source': source,
                    'target': target,
                    'source_data': source_data,
                    'target_data': target_data,
                    'edge_data': edge_data,
                    'edge_type': edge_type,
                    'source_type': source_type,
                    'target_type': target_type
                })
    
    if not candidate_edges:
        return None
    
    # 随机选择一个边
    selected = random.choice(candidate_edges)
    
    T_start = selected['source_type']
    P_start = 'name'
    V_start = selected['source_data']['name']
    E_type = selected['edge_type']
    T_target = selected['target_type']
    
    # 选择边的属性过滤（如果有数值型属性）
    E_prop_k = None
    E_prop_v = None
    edge_attrs = selected['edge_data']
    for attr, value in edge_attrs.items():
        if attr not in ['type', 'label', 'edge_type', 'relation', 'display_relation', 'version']:
            if isinstance(value, (int, float)):
                E_prop_k = attr
                if isinstance(value, float) and value > 0:
                    E_prop_v = f">{value * 0.5}"
                elif isinstance(value, int) and value > 0:
                    E_prop_v = f">{value // 2}"
                break
    
    return Params(
        T_start=T_start,
        P_start=P_start,
        V_start=V_start,
        E_type=E_type,
        E_prop_k=E_prop_k,
        E_prop_v=E_prop_v,
        T_target=T_target
    )


def sample_params_for_chain_hop(stats):
    """为 Chain_Hop_Template 采样参数 - 使用实际存在的路径"""
    G = get_graph()
    
    # 找一个有name属性的起始节点
    start_node = None
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        if 'name' in node_data:
            start_node = node_id
            start_data = node_data
            break
    
    if not start_node:
        return None
    
    T1 = start_data.get('type') or start_data.get('label') or start_data.get('node_type')
    P1 = 'name'
    V1 = start_data['name']
    
    # 从起始节点开始，找实际存在的路径
    # 第一跳
    E1 = None
    T2 = None
    for source, target, edge_data in G.out_edges(start_node, data=True):
        edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
        target_data = G.nodes[target]
        target_type = target_data.get('type') or target_data.get('label') or target_data.get('node_type')
        if edge_type and target_type:
            E1 = edge_type
            T2 = target_type
            mid_node = target
            break
    
    if not E1:
        return None
    
    # 第二跳
    E2 = None
    T3 = None
    for source, target, edge_data in G.out_edges(mid_node, data=True):
        edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
        target_data = G.nodes[target]
        target_type = target_data.get('type') or target_data.get('label') or target_data.get('node_type')
        if edge_type and target_type:
            E2 = edge_type
            T3 = target_type
            third_node = target
            break
    
    if not E2:
        # 如果第二跳不存在，返回两跳的查询
        return Params(
            T1=T1,
            P1=P1,
            V1=V1,
            E1=E1,
            T2=T2,
            E2=None,
            T3=None,
            E3=None,
            T4=None
        )
    
    # 第三跳（可选）
    E3 = None
    T4 = None
    for source, target, edge_data in G.out_edges(third_node, data=True):
        edge_type = edge_data.get('type') or edge_data.get('label') or edge_data.get('edge_type') or edge_data.get('relation')
        target_data = G.nodes[target]
        target_type = target_data.get('type') or target_data.get('label') or target_data.get('node_type')
        if edge_type and target_type:
            E3 = edge_type
            T4 = target_type
            break
    
    return Params(
        T1=T1,
        P1=P1,
        V1=V1,
        E1=E1,
        T2=T2,
        E2=E2,
        T3=T3,
        E3=E3,
        T4=T4
    )


def test_direct_relation():
    """测试 Direct_Relation_Template"""
    print("\n" + "="*60)
    print("测试 Direct_Relation_Template")
    print("="*60)
    
    # 采样统计信息
    stats = sample_graph_statistics()
    print(f"\n图统计信息:")
    print(f"  - 节点类型数: {len(stats['node_types'])}")
    print(f"  - 边类型数: {len(stats['edge_types'])}")
    print(f"  - 前5个节点类型: {list(stats['node_types'].items())[:5]}")
    print(f"  - 前5个边类型: {list(stats['edge_types'].items())[:5]}")
    
    # 采样参数
    params = sample_params_for_direct_relation(stats)
    if not params:
        print("❌ 无法采样参数（图可能为空）")
        return
    
    print(f"\n采样参数:")
    print(f"  T_start: {params.T_start}")
    print(f"  P_start: {params.P_start}")
    print(f"  V_start: {params.V_start}")
    print(f"  E_type: {params.E_type}")
    print(f"  E_prop_k: {params.E_prop_k}")
    print(f"  E_prop_v: {params.E_prop_v}")
    print(f"  T_target: {params.T_target}")
    
    # 执行查询
    try:
        result = Direct_Relation_Template(params)
        print(f"\n✅ 查询成功!")
        print(f"  结果节点数: {len(result)}")
        if result:
            print(f"  前10个结果节点: {result[:10]}")
        else:
            print("  ⚠️  未找到匹配的节点")
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


def test_chain_hop():
    """测试 Chain_Hop_Template"""
    print("\n" + "="*60)
    print("测试 Chain_Hop_Template")
    print("="*60)
    
    # 采样统计信息
    stats = sample_graph_statistics()
    
    # 采样参数
    params = sample_params_for_chain_hop(stats)
    if not params:
        print("❌ 无法采样参数（图可能为空）")
        return
    
    print(f"\n采样参数:")
    print(f"  T1: {params.T1}, P1: {params.P1}, V1: {params.V1}")
    print(f"  E1: {params.E1}, T2: {params.T2}")
    print(f"  E2: {params.E2}, T3: {params.T3}")
    if params.E3:
        print(f"  E3: {params.E3}, T4: {params.T4}")
    
    # 执行查询
    try:
        result = Chain_Hop_Template(params)
        print(f"\n✅ 查询成功!")
        print(f"  结果节点数: {len(result)}")
        if result:
            print(f"  前10个结果节点: {result[:10]}")
        else:
            print("  ⚠️  未找到匹配的节点")
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    graph_path = Path(__file__).parent.parent / "data_gen" / "graph_gen" / "graph_buffer" / "Primekg_noise_20251231_083857.gpickle"
    
    if not graph_path.exists():
        print(f"❌ 图文件不存在: {graph_path}")
        return
    
    print(f"加载图文件: {graph_path}")
    load_graph(str(graph_path))
    
    # 测试两个模板
    test_direct_relation()
    test_chain_hop()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()

