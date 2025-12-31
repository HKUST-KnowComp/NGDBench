"""
图生成器 - 用于加载图数据并应用扰动

使用示例：
    python graph_generator.py --input graph_buffer/Primekg.gpickle --output graph_buffer/Primekg_perturbed.gpickle
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from perturbation_generator.graph_perturbation import (
    GraphPerturbation,
    load_graph_from_gpickle
)


def main():
    """主函数 - 演示图扰动的完整流程"""
    
    # 默认路径配置
    default_graph_dir = Path(__file__).parent / "graph_gen" / "graph_buffer"
    default_records_dir = Path(__file__).parent / "perturbation_generator" / "perturb_record"
    default_guide_file = Path(__file__).parent / "perturbation_generator" / "perturb_guide" / "general_guid.json"
    
    parser = argparse.ArgumentParser(description="图扰动生成器")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(default_graph_dir / "Primekg.gpickle"),
        help="输入图文件路径 (.gpickle格式)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(default_graph_dir),
        help="图文件输出目录路径 (默认: graph_buffer/)"
    )
    parser.add_argument(
        "--records-dir", "-r",
        type=str,
        default=str(default_records_dir),
        help="记录文件输出目录路径 (默认: perturb_record/)"
    )
    parser.add_argument(
        "--guide", "-g",
        type=str,
        default=str(default_guide_file),
        help="扰动指导文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--save-records",
        action="store_true",
        default=True,
        help="是否保存扰动记录"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="数据集名称 (默认: 从输入文件名推断)"
    )
    
    args = parser.parse_args()
    
    # 从输入文件名推断数据集名称
    input_path = Path(args.input)
    dataset_name = args.dataset_name or input_path.stem
    
    print(f"\n{'='*70}")
    print("图扰动生成器")
    print(f"{'='*70}")
    print(f"输入文件: {args.input}")
    print(f"图输出目录: {args.output_dir}")
    print(f"记录输出目录: {args.records_dir}")
    print(f"数据集名称: {dataset_name}")
    print(f"指导文件: {args.guide}")
    print(f"{'='*70}\n")
    
    # Step 1: 加载图
    print("Step 1: 加载图数据...")
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    graph = load_graph_from_gpickle(args.input)
    
    # Step 2: 创建扰动器
    print("\nStep 2: 初始化图扰动器...")
    if not Path(args.guide).exists():
        print(f"警告: 指导文件不存在: {args.guide}")
        print("   将使用默认扰动配置")
        perturbation = GraphPerturbation(graph)
        # 设置默认配置
        perturbation.set_noise_profile({
            "false_edges": 0.03,
            "relation_type_noise": 0.02,
            "name_typos": 0.2,
            "node_type_noise": 0.05
        })
    else:
        perturbation = GraphPerturbation(graph, guide_file=args.guide)
    
    # Step 3: 应用扰动
    print("\nStep 3: 应用扰动...")
    perturbed_graph = perturbation.apply_all_perturbations()
    
    # Step 4: 显示扰动摘要
    print("\nStep 4: 扰动摘要")
    summary = perturbation.get_perturbation_summary()
    print(f"   原始图: {summary['original_graph']['nodes']:,} 节点, {summary['original_graph']['edges']:,} 边")
    print(f"   扰动图: {summary['perturbed_graph']['nodes']:,} 节点, {summary['perturbed_graph']['edges']:,} 边")
    print(f"   噪声节点数: {summary['noise_statistics']['noisy_nodes_count']:,}")
    print(f"   噪声边数: {summary['noise_statistics']['noisy_edges_count']:,}")
    print(f"   - 删除边数: {summary['noise_statistics']['deleted_edges_count']:,}")
    print(f"   - 添加边数: {summary['noise_statistics']['added_edges_count']:,}")
    print(f"   - 修改边数: {summary['noise_statistics']['modified_edges_count']:,}")
    print(f"   - 修改节点数: {summary['noise_statistics']['modified_nodes_count']:,}")
    
    # Step 5: 保存结果（使用时间戳命名）
    print("\nStep 5: 保存结果...")
    saved_files = perturbation.save_all_with_timestamp(
        dataset_name=dataset_name,
        output_dir=args.output_dir,
        records_dir=args.records_dir,
        save_records=args.save_records
    )
    
    print(f"\n{'='*70}")
    print("扰动完成!")
    print(f"{'='*70}")
    print("保存的文件:")
    for file_type, file_path in saved_files.items():
        print(f"   - {file_type}: {file_path}")
    print(f"{'='*70}\n")


def demo_basic_usage():
    """
    基本使用演示 - 展示如何在代码中使用 GraphPerturbation 类
    """
    print("\n" + "="*70)
    print("📚 GraphPerturbation 基本使用演示")
    print("="*70 + "\n")
    
    # 路径配置
    graph_dir = Path(__file__).parent / "graph_gen" / "graph_buffer"
    guide_file = Path(__file__).parent / "perturbation_generator" / "perturb_guide" / "general_guid.json"
    
    # 选择一个图文件
    graph_file = graph_dir / "Primekg.gpickle"
    
    if not graph_file.exists():
        print(f"图文件不存在: {graph_file}")
        print("请确保 graph_buffer 目录下有 .gpickle 文件")
        return
    
    # 1. 加载图
    print("1.加载图...")
    graph = load_graph_from_gpickle(str(graph_file))
    
    # 2. 创建扰动器
    print("\n2.创建扰动器...")
    perturbation = GraphPerturbation(graph, guide_file=str(guide_file))
    
    # 3. 可选：自定义扰动配置
    # perturbation.set_noise_profile({
    #     "incomplete_edges": 0.01,  # 删除1%的边
    #     "false_edges": 0.02,       # 添加2%的假边
    #     "relation_type_noise": 0.01,
    #     "node_type_noise": 0.03,
    #     "name_typos": 0.1
    # })
    
    # 4. 应用扰动
    print("\n3.应用扰动...")
    perturbed_graph = perturbation.apply_all_perturbations()
    
    # 5. 获取扰动信息
    print("\n4.获取扰动信息...")
    
    # 获取噪声节点列表
    noisy_nodes = perturbation.get_noisy_nodes()
    print(f"   噪声节点数量: {len(noisy_nodes)}")
    if noisy_nodes:
        sample_nodes = list(noisy_nodes)[:5]
        print(f"   示例节点: {sample_nodes}")
    
    # 获取噪声边列表
    noisy_edges = perturbation.get_noisy_edges()
    print(f"\n   噪声边数量: {len(noisy_edges)}")
    if noisy_edges:
        sample_edges = list(noisy_edges)[:5]
        print(f"   示例边: {sample_edges}")
    
    # 获取详细记录
    detailed = perturbation.get_detailed_records()
    print(f"\n   删除边记录数: {len(detailed['deleted_edges'])}")
    print(f"   添加边记录数: {len(detailed['added_edges'])}")
    print(f"   修改边记录数: {len(detailed['modified_edges'])}")
    print(f"   修改节点记录数: {len(detailed['modified_nodes'])}")
    
    # 6. 展示部分详细记录
    print("\n5.详细记录示例...")
    
    if detailed['deleted_edges']:
        print("\n   删除边示例:")
        for record in detailed['deleted_edges'][:2]:
            print(f"      - 边: {record['source']} -> {record['target']}")
            print(f"        原始属性: {record['original_attrs']}")
    
    if detailed['added_edges']:
        print("\n   添加边示例:")
        for record in detailed['added_edges'][:2]:
            print(f"      - 边: {record['source']} -> {record['target']}")
            print(f"        新属性: {record['new_attrs']}")
    
    if detailed['modified_nodes']:
        print("\n   修改节点示例:")
        for record in detailed['modified_nodes'][:2]:
            print(f"      - 节点: {record['node_id']}")
            print(f"        修改字段: {record['field_changed']}")
            if 'name' in record['original_attrs']:
                print(f"        原始名称: {record['original_attrs'].get('name', 'N/A')}")
            if record['new_attrs'] and 'name' in record['new_attrs']:
                print(f"        新名称: {record['new_attrs'].get('name', 'N/A')}")
    
    # 7. 保存结果（使用时间戳命名）
    print("\n6.保存结果...")
    records_dir = Path(__file__).parent / "perturbation_generator" / "perturb_record"
    saved_files = perturbation.save_all_with_timestamp(
        dataset_name="Primekg",
        output_dir=str(graph_dir),
        records_dir=str(records_dir),
        save_records=True
    )
    
    print("\n" + "="*70)
    print("演示完成!")
    print("保存的文件:")
    for file_type, file_path in saved_files.items():
        print(f"   - {file_type}: {file_path}")
    print("="*70 + "\n")


def demo_selective_perturbation():
    """
    选择性扰动演示 - 展示如何只应用部分扰动类型
    """
    print("\n" + "="*70)
    print("📚 选择性扰动演示")
    print("="*70 + "\n")
    
    graph_dir = Path(__file__).parent / "graph_gen" / "graph_buffer"
    graph_file = graph_dir / "Primekg.gpickle"
    
    if not graph_file.exists():
        print(f"图文件不存在: {graph_file}")
        return
    
    # 加载图
    graph = load_graph_from_gpickle(str(graph_file))
    
    # 创建扰动器（不使用指导文件）
    perturbation = GraphPerturbation(graph)
    
    # 只应用特定的扰动类型
    perturbation.set_noise_profile({
        "name_typos": 0.05,  # 只对5%的节点注入拼写错误
        "false_edges": 0.01  # 添加1%的假边
    })
    
    # 应用扰动
    perturbed_graph = perturbation.apply_all_perturbations()
    
    # 显示结果
    summary = perturbation.get_perturbation_summary()
    print(f"\n扰动摘要:")
    print(f"  - 噪声节点: {summary['noise_statistics']['noisy_nodes_count']}")
    print(f"  - 噪声边: {summary['noise_statistics']['noisy_edges_count']}")
    
    print("\n选择性扰动演示完成!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # 运行演示
        demo_basic_usage()
    elif len(sys.argv) > 1 and sys.argv[1] == "--selective":
        # 运行选择性扰动演示
        demo_selective_perturbation()
    else:
        # 运行主程序
        main()



