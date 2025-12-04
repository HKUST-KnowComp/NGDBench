"""
语义扰动生成器使用示例

展示如何使用基于指导文件的 SemanticPerturbationGenerator 来对生物医学知识图谱添加语义噪声
"""

import networkx as nx
from semantic_perturbation import SemanticPerturbationGenerator

# ============ 示例 1: 使用指导文件和默认配置 ============

def example_with_guide_file():
    """使用 paramkg.json 指导文件和默认噪声配置"""
    
    # 创建一个示例知识图谱
    graph = nx.Graph()
    
    # 添加节点（模拟生物医学实体）
    graph.add_node("DB01234", x_name="Aspirin", x_type="drug", x_source="DrugBank", x_id="DB01234")
    graph.add_node("9606", x_name="TP53", x_type="gene", x_source="NCBI", x_id="9606")
    graph.add_node("5594", y_name="MAPK1", y_type="protein", y_source="UniProt", y_id="5594")
    graph.add_node("GO:0008283", x_name="cell proliferation", x_type="biological_process", 
                   x_source="GO", x_id="GO:0008283")
    
    # 添加边（模拟生物医学关系）
    graph.add_edge("DB01234", "9606", relation="inhibits", display_relation="inhibits", 
                   relation_type="drug_gene")
    graph.add_edge("9606", "5594", relation="ppi", display_relation="protein-protein interaction",
                   relation_type="protein_protein")
    graph.add_edge("5594", "GO:0008283", relation="positively_regulates", 
                   display_relation="positively regulates", relation_type="bioprocess_protein")
    
    # 配置生成器
    config = {
        'guide_file': './semantic_perturb_guide/paramkg.json',  # 指导文件路径
        # noise_profile 不指定，将使用指导文件中的 default_profile
    }
    
    # 创建生成器实例
    generator = SemanticPerturbationGenerator(config)
    
    # 应用语义噪声
    perturbed_graph, perturbation_info = generator.apply_perturbation(graph, 'noise')
    
    # 查看扰动信息
    print("扰动操作数量:", len(perturbation_info['operations']))
    print("\n扰动详情:")
    for op in perturbation_info['operations'][:5]:  # 显示前5个操作
        print(f"- {op['operation']}: {op}")
    
    return perturbed_graph, perturbation_info


# ============ 示例 2: 自定义噪声配置 ============

def example_with_custom_profile():
    """使用自定义的噪声比例配置"""
    
    # 创建示例图谱
    graph = nx.Graph()
    graph.add_node("DB00001", x_name="Lepirudin", x_type="drug", x_id="DB00001")
    graph.add_node("7157", x_name="TP53", x_type="gene", x_id="7157")
    graph.add_edge("DB00001", "7157", relation="targets", display_relation="targets")
    
    # 自定义噪声配置（更高的拼写错误比例，更低的缺失边比例）
    custom_noise_profile = {
        "false_edges": 0.10,        # 10% 虚假边
        "missing_edges": 0.02,      # 2% 缺失边（降低）
        "name_typos": 0.20,         # 20% 名称拼写错误（提高）
        "relation_type_noise": 0.05,
        "source_conflicts": 0.03,
        "node_type_noise": 0.02,
        "id_corruption": 0.05,
        "duplicate_edges": 0.05,
        "path_level_noise": 0.02
    }
    
    config = {
        'guide_file': './semantic_perturb_guide/paramkg.json',
        'noise_profile': custom_noise_profile  # 使用自定义配置
    }
    
    generator = SemanticPerturbationGenerator(config)
    perturbed_graph, perturbation_info = generator.apply_perturbation(graph, 'noise')
    
    return perturbed_graph, perturbation_info


# ============ 示例 3: 仅应用特定类型的噪声 ============

def example_specific_noise_types():
    """仅应用特定类型的噪声"""
    
    graph = nx.Graph()
    # ... 添加节点和边 ...
    
    # 只应用名称拼写错误和ID损坏
    specific_noise_profile = {
        "false_edges": 0.0,
        "missing_edges": 0.0,
        "name_typos": 0.3,          # 只启用名称拼写错误
        "relation_type_noise": 0.0,
        "source_conflicts": 0.0,
        "node_type_noise": 0.0,
        "id_corruption": 0.2,       # 只启用ID损坏
        "duplicate_edges": 0.0,
        "path_level_noise": 0.0
    }
    
    config = {
        'guide_file': './semantic_perturb_guide/paramkg.json',
        'noise_profile': specific_noise_profile
    }
    
    generator = SemanticPerturbationGenerator(config)
    perturbed_graph, perturbation_info = generator.apply_perturbation(graph, 'noise')
    
    return perturbed_graph, perturbation_info


# ============ 示例 4: 分析扰动效果 ============

def analyze_perturbation_effects():
    """分析扰动对图谱的影响"""
    
    # 创建原始图谱
    original_graph = nx.Graph()
    # ... 添加节点和边 ...
    
    config = {
        'guide_file': './semantic_perturb_guide/paramkg.json',
    }
    
    generator = SemanticPerturbationGenerator(config)
    perturbed_graph, perturbation_info = generator.apply_perturbation(
        original_graph.copy(), 'noise'
    )
    
    # 统计各类噪声的数量
    noise_counts = {}
    for op in perturbation_info['operations']:
        noise_type = op['operation']
        noise_counts[noise_type] = noise_counts.get(noise_type, 0) + 1
    
    print("噪声类型统计:")
    for noise_type, count in noise_counts.items():
        print(f"  {noise_type}: {count}")
    
    # 对比图谱变化
    print(f"\n图谱变化:")
    print(f"  原始节点数: {original_graph.number_of_nodes()}")
    print(f"  扰动后节点数: {perturbed_graph.number_of_nodes()}")
    print(f"  原始边数: {original_graph.number_of_edges()}")
    print(f"  扰动后边数: {perturbed_graph.number_of_edges()}")
    
    return noise_counts


# ============ 示例 5: 批量处理多个数据集 ============

def batch_process_datasets():
    """批量处理多个数据集"""
    
    datasets = [
        ("dataset1.graphml", "./semantic_perturb_guide/paramkg.json"),
        ("dataset2.graphml", "./semantic_perturb_guide/other_kg.json"),
    ]
    
    for dataset_path, guide_file in datasets:
        # 加载数据集
        # graph = nx.read_graphml(dataset_path)
        
        # 配置生成器
        config = {
            'guide_file': guide_file,
        }
        
        generator = SemanticPerturbationGenerator(config)
        # perturbed_graph, info = generator.apply_perturbation(graph, 'noise')
        
        # 保存扰动后的图谱
        # output_path = dataset_path.replace('.graphml', '_noisy.graphml')
        # nx.write_graphml(perturbed_graph, output_path)
        
        print(f"已处理: {dataset_path}")


# ============ 运行示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("示例 1: 使用指导文件和默认配置")
    print("=" * 60)
    # example_with_guide_file()
    
    print("\n" + "=" * 60)
    print("示例 2: 自定义噪声配置")
    print("=" * 60)
    # example_with_custom_profile()
    
    print("\n" + "=" * 60)
    print("示例 4: 分析扰动效果")
    print("=" * 60)
    # analyze_perturbation_effects()
    
    print("\n使用方法:")
    print("1. 准备指导文件（如 paramkg.json）")
    print("2. 配置 guide_file 路径")
    print("3. 可选：自定义 noise_profile")
    print("4. 调用 apply_perturbation(graph, 'noise')")

