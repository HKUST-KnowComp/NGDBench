import json
import argparse
import logging
from generator.noise_query_gen import NoiseQueryGenerator

logger = logging.getLogger(__name__)

# 默认连接配置（可被命令行参数覆盖）
DEFAULT_NEO4J_URI = "bolt://localhost:7692"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "fei123456"
DEFAULT_DATASET = "ldbcfin"
DEFAULT_PERTURB_RECORD_DIR = "../../data_gen/perturbation_generator/perturb_record"


def test_noise_query_generator(uri: str, user: str, password: str, dataset: str, 
                                perturb_record_dir: str, noise_prefix: str = None):
    """测试噪声查询生成器"""
    
    # 初始化输出文件变量，避免在异常时未定义
    output_file_step1 = f"noise_query_results_step1_{dataset}.json"
    output_file_step2 = f"noise_judge_query_results_step2_{dataset}.json"
    
    # ========== 第一步：根据 template.json 生成 2000 个查询 ==========
    print("\n" + "="*80)
    print("第一步：根据 template.json 生成 2000 个查询")
    print("="*80)
    
    generator_step1 = NoiseQueryGenerator(
        uri=uri,
        user=user,
        password=password,
        template_path="query_template/template.json",
        perturb_record_dir=perturb_record_dir,
        noise_prefix=noise_prefix,
        dataset=dataset,
    )
    
    try:
        generator_step1.connect()
        generator_step1.initialize()
        
        print(f"\n=== Schema 信息 ===")
        print(f"Labels: {list(generator_step1.schema.labels.keys())}")
        print(f"Relationships: {list(generator_step1.schema.relationships.keys())}")
        print(f"Total nodes: {generator_step1.schema.total_nodes}")
        print(f"Total edges: {generator_step1.schema.total_edges}")
        print(f"Target sample count: {generator_step1.get_target_sample_count()}")
        
        print(f"\n=== 噪声数据信息 ===")
        print(f"噪声节点数量: {len(generator_step1.noise_loader.noisy_node_set)}")
        print(f"噪声边数量: {len(generator_step1.noise_loader.noisy_edges)}")
        print(f"噪声节点标签: {list(generator_step1.noise_loader.noisy_nodes.keys())}")
        if generator_step1.noise_loader.noisy_nodes:
            for label, node_ids in list(generator_step1.noise_loader.noisy_nodes.items())[:5]:
                print(f"  {label}: {len(node_ids)} 个噪声节点 (示例: {node_ids[:3]})")
        
        print(f"\n=== 开始生成噪声查询（第一步）===")
        output_file_step1 = f"noise_query_results_step1_{dataset}.json"
        # 使用实时输出，一边生成一边写入文件
        results_step1 = generator_step1.generate_samples(
            target_count=200,
            realtime_output_path=output_file_step1
        )
        
        print(f"\n=== 第一步生成结果 ===")
        print(f"共生成 {len(results_step1)} 个查询")
        print(f"所有结果已实时写入文件: {output_file_step1}")
        
        # 统计第一步的结果
        noise_query_count_step1 = 0
        clean_query_count_step1 = 0
        
        for r in results_step1:
            verification_result = _verify_query_contains_noise(r, generator_step1.noise_loader)
            if verification_result['contains_noise']:
                noise_query_count_step1 += 1
            else:
                clean_query_count_step1 += 1
        
        print(f"噪声查询数量: {noise_query_count_step1}/{len(results_step1)}")
        print(f"干净查询数量: {clean_query_count_step1}/{len(results_step1)}")
        
    except Exception as e:
        logger.error(f"第一步测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator_step1.close()
    
    # ========== 第二步：根据 template_judge1.json 生成 200 组查询 ==========
    print("\n" + "="*80)
    print("第二步：根据 template_judge1.json 生成 200 组查询")
    print("="*80)
    
    generator_step2 = NoiseQueryGenerator(
        uri=uri,
        user=user,
        password=password,
        template_path="query_template/template.json",  # 这个路径在 generate_judge_queries 中会被覆盖
        perturb_record_dir=perturb_record_dir,
        noise_prefix=noise_prefix,
        dataset=dataset,
    )
    
    try:
        generator_step2.connect()
        generator_step2.initialize()
        
        print(f"\n=== 开始生成判断查询（第二步）===")
        output_file_step2 = f"noise_judge_query_results_step2_{dataset}.json"
        
        # 目标：生成 200 组查询
        target_groups = 50
        all_results_step2 = []
        seen_queries = set()  # 用于去重，基于查询字符串
        
        # 循环生成查询，直到达到目标数量
        max_iterations = 100  # 最多尝试100次，避免无限循环
        iteration = 0
        
        while len(all_results_step2) < target_groups and iteration < max_iterations:
            iteration += 1
            print(f"\n--- 第 {iteration} 轮生成（当前已有 {len(all_results_step2)} 组，目标 {target_groups} 组）---")
            
            # 使用 generate_judge_queries 方法生成查询
            results_batch = generator_step2.generate_judge_queries(
                template_file_path="query_template/template_judge1.json",
                max_unique_answers=20
            )
            
            # 去重：只添加新的查询（基于 template_query 和 anti_template_query 的组合）
            new_count = 0
            for result in results_batch:
                # 使用查询字符串组合作为唯一标识
                query_key = (result["template_query"], result["anti_template_query"])
                if query_key not in seen_queries:
                    seen_queries.add(query_key)
                    all_results_step2.append(result)
                    new_count += 1
            
            print(f"本轮生成 {len(results_batch)} 组查询，其中 {new_count} 组为新查询，累计 {len(all_results_step2)} 组")
            
            # 如果本轮没有生成任何新结果，可能已经穷尽所有可能性，退出循环
            if new_count == 0:
                print("本轮未生成新查询，停止生成")
                break
            
            # 如果已达到目标数量，退出循环
            if len(all_results_step2) >= target_groups:
                break
        
        # 如果超过目标数量，截取前 target_groups 个
        if len(all_results_step2) > target_groups:
            all_results_step2 = all_results_step2[:target_groups]
            print(f"\n已生成 {len(all_results_step2)} 组查询（达到目标数量）")
        else:
            print(f"\n共生成 {len(all_results_step2)} 组查询（未达到目标数量 {target_groups}，可能模板数量不足）")
        
        print(f"\n=== 第二步生成结果 ===")
        print(f"共处理 {len(all_results_step2)} 组查询")
        
        # 保存结果到文件
        output_data = []
        for result in all_results_step2:
            output_data.append({
                "template_id": result["template_id"],
                "template_type": result["template_type"],
                "template_query": result["template_query"],
                "anti_template_query": result["anti_template_query"],
                "parameters_used": result["parameters_used"],
                "contains_noise": result.get("contains_noise", True),
                "template_results_count": result["template_results_count"],
                "anti_template_results_count": result["anti_template_results_count"],
                "unique_in_template_count": result["unique_in_template_count"],
                "unique_in_anti_template_count": result["unique_in_anti_template_count"],
                "unique_in_template_answers": result["unique_in_template_answers"],
                "unique_in_anti_template_answers": result["unique_in_anti_template_answers"]
            })
        
        with open(output_file_step2, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"所有结果已写入文件: {output_file_step2}")
        
        # 统计第二步的结果
        total_unique_template = sum(r["unique_in_template_count"] for r in all_results_step2)
        total_unique_anti_template = sum(r["unique_in_anti_template_count"] for r in all_results_step2)
        
        # 按模板ID统计
        template_counts = {}
        for r in all_results_step2:
            template_key = f"{r['template_type']}_{r['template_id']}"
            template_counts[template_key] = template_counts.get(template_key, 0) + 1
        
        print(f"\n=== 第二步统计信息 ===")
        print(f"生成的查询组数: {len(all_results_step2)}")
        print(f"Template 独特答案总数: {total_unique_template}")
        print(f"Anti-template 独特答案总数: {total_unique_anti_template}")
        print(f"\n各模板生成数量:")
        for template_key, count in sorted(template_counts.items()):
            print(f"  {template_key}: {count} 组")
        
        # 显示前几个查询组的结果摘要
        for i, r in enumerate(all_results_step2[:5]):
            print(f"\n[{i+1}] Template: {r['template_id']} ({r['template_type']})")
            print(f"    Template 查询结果数: {r['template_results_count']}")
            print(f"    Anti-template 查询结果数: {r['anti_template_results_count']}")
            print(f"    Template 独特答案数: {r['unique_in_template_count']}")
            print(f"    Anti-template 独特答案数: {r['unique_in_anti_template_count']}")
        
        if len(all_results_step2) > 5:
            print(f"\n... 还有 {len(all_results_step2) - 5} 组查询结果")
        
    except Exception as e:
        logger.error(f"第二步测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator_step2.close()
    
    print("\n" + "="*80)
    print("所有步骤完成！")
    print("="*80)
    print(f"第一步输出文件: {output_file_step1}")
    print(f"第二步输出文件: {output_file_step2}")


def _verify_query_contains_noise(result, noise_loader) -> dict:
    """
    验证查询结果是否包含噪声节点或边
    
    Returns:
        包含以下字段的字典：
        - contains_noise: 是否包含噪声
        - uses_noisy_node_id: 是否使用了噪声节点ID
        - uses_noisy_edge: 是否使用了噪声边
        - uses_noisy_label_only: 是否仅使用了噪声标签（但未使用噪声节点ID）
    """
    import re
    
    query = result.query
    params = result.parameters_used
    
    result_dict = {
        'contains_noise': False,
        'uses_noisy_node_id': False,
        'uses_noisy_edge': False,
        'uses_noisy_label_only': False
    }
    
    # 检查参数中使用的标签和值是否对应噪声节点
    used_labels = set()
    noisy_labels_in_query = []
    
    for param_name, value in params.items():
        # 检查LABEL参数
        if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
            label = str(value)
            used_labels.add(label)
            # 检查该标签是否有噪声节点
            if label in noise_loader.noisy_nodes:
                noisy_labels_in_query.append(label)
                result_dict['uses_noisy_label_only'] = True
                
                # 检查VALUE参数是否对应噪声节点的ID
                for value_key in ['VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'END_VALUE']:
                    if value_key in params:
                        value_str = str(params[value_key])
                        # 检查这个值是否在噪声节点ID列表中
                        if value_str in noise_loader.noisy_nodes[label]:
                            result_dict['uses_noisy_node_id'] = True
                            result_dict['contains_noise'] = True
                            return result_dict  # 找到明确的噪声节点ID使用，直接返回
    
    # 检查查询字符串中是否包含噪声节点的ID（作为字符串值）
    quoted_strings = re.findall(r"'([^']+)'", query)
    for quoted_str in quoted_strings:
        # 检查这个字符串是否是某个噪声节点的ID
        for label, node_ids in noise_loader.noisy_nodes.items():
            if quoted_str in node_ids:
                result_dict['uses_noisy_node_id'] = True
                result_dict['contains_noise'] = True
                return result_dict  # 找到明确的噪声节点ID使用，直接返回
    
    # 检查噪声边：如果查询中使用了两个噪声节点的标签，可能包含噪声边
    if len(noisy_labels_in_query) >= 2:
        # 进一步检查是否真的使用了噪声边
        # 检查START_VALUE和END_VALUE是否对应噪声节点
        start_value = params.get('START_VALUE')
        end_value = params.get('END_VALUE')
        
        if start_value and end_value:
            # 检查这两个值是否在噪声节点列表中
            start_is_noisy = False
            end_is_noisy = False
            
            for label in noisy_labels_in_query:
                if str(start_value) in noise_loader.noisy_nodes.get(label, []):
                    start_is_noisy = True
                if str(end_value) in noise_loader.noisy_nodes.get(label, []):
                    end_is_noisy = True
            
            if start_is_noisy and end_is_noisy:
                # 检查这两个节点是否构成噪声边
                start_node_str = None
                end_node_str = None
                for label in noisy_labels_in_query:
                    if str(start_value) in noise_loader.noisy_nodes.get(label, []):
                        start_node_str = f"{label}:{start_value}"
                    if str(end_value) in noise_loader.noisy_nodes.get(label, []):
                        end_node_str = f"{label}:{end_value}"
                
                if start_node_str and end_node_str:
                    if (start_node_str, end_node_str) in noise_loader.noisy_edge_set:
                        result_dict['uses_noisy_edge'] = True
                        result_dict['contains_noise'] = True
                        return result_dict
        
        # 如果只是使用了多个噪声标签，也认为可能包含噪声
        result_dict['contains_noise'] = True
    
    # 如果只使用了噪声标签但没有使用噪声节点ID，标记为仅使用标签
    if result_dict['uses_noisy_label_only'] and not result_dict['uses_noisy_node_id']:
        result_dict['contains_noise'] = True
    
    return result_dict


def _extract_noise_info(result, noise_loader) -> dict:
    """提取查询中使用的噪声信息"""
    params = result.parameters_used
    noise_info = {}
    
    for param_name, value in params.items():
        if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
            label = value
            if label in noise_loader.noisy_nodes:
                noise_info[param_name] = {
                    'label': label,
                    'noisy_node_count': len(noise_loader.noisy_nodes[label])
                }
                # 检查对应的VALUE
                for value_key in ['VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE']:
                    if value_key in params:
                        value_str = str(params[value_key])
                        if value_str in noise_loader.noisy_nodes[label]:
                            noise_info[param_name]['noisy_node_id'] = value_str
                            break
    
    return noise_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NoiseQueryGenerator with Neo4j.")
    parser.add_argument(
        "--uri",
        type=str,
        default=DEFAULT_NEO4J_URI,
        help=f"Neo4j URI, 默认: {DEFAULT_NEO4J_URI}",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=DEFAULT_NEO4J_USER,
        help=f"Neo4j 用户名, 默认: {DEFAULT_NEO4J_USER}",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=DEFAULT_NEO4J_PASSWORD, 
        help="Neo4j 密码（命令行传入会出现在历史记录中，注意安全）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"数据集, 默认: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--perturb-record-dir",
        type=str,
        default=DEFAULT_PERTURB_RECORD_DIR,
        help=f"扰动记录文件夹路径, 默认: {DEFAULT_PERTURB_RECORD_DIR}",
    )
    parser.add_argument(
        "--noise-prefix",
        type=str,
        default="ldbc_snb_finbench_noise_20260118_220611",
        help="噪声文件前缀，如果为None则自动查找最新的文件",
    )
    args = parser.parse_args()
    test_noise_query_generator(
        uri=args.uri,
        user=args.user,
        password=args.password,
        dataset=args.dataset,
        perturb_record_dir=args.perturb_record_dir,
        noise_prefix=args.noise_prefix,
    )
