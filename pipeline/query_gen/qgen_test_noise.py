import json
import os
import sys
import argparse
import logging

# 添加 pipeline 目录到路径，以便导入 handler 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from generator.noise_query_gen import NoiseQueryGenerator
from handler.cleaner import extract_and_clean_answers, clean_judge_query_answers

logger = logging.getLogger(__name__)

# 默认连接配置（可被命令行参数覆盖）
DEFAULT_NEO4J_URI = "bolt://localhost:7692"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "fei123456"
DEFAULT_DATASET = "ldbcfin"
DEFAULT_PERTURB_RECORD_DIR = "../../data_gen/perturbation_generator/perturb_record"


def generate_step1_queries(uri: str, user: str, password: str, dataset: str,
                           perturb_record_dir: str, noise_prefix: str = None,
                           template_path: str = "query_template/template.json",
                           target_count: int = 100,
                           output_file: str = None,
                           show_schema: bool = True,
                           show_noise_info: bool = True) -> dict:
    """
    第一步：根据 template.json 生成噪声查询
    
    Args:
        uri: Neo4j URI
        user: Neo4j 用户名
        password: Neo4j 密码
        dataset: 数据集名称
        perturb_record_dir: 扰动记录文件夹路径
        noise_prefix: 噪声文件前缀
        template_path: 模板文件路径
        target_count: 目标生成数量
        output_file: 输出文件路径（如果为None，则使用默认路径）
        show_schema: 是否显示Schema信息
        show_noise_info: 是否显示噪声数据信息
    
    Returns:
        包含以下字段的字典：
        - output_file: 输出文件路径
        - results: 生成的结果列表
        - noise_query_count: 噪声查询数量
        - clean_query_count: 干净查询数量
        - generator: NoiseQueryGenerator实例（已关闭连接）
    """
    if output_file is None:
        output_file = f"noise_query_results_step1_{dataset}.json"
    
    print("\n" + "="*80)
    print("第一步：根据 template.json 生成噪声查询")
    print("="*80)
    
    generator = NoiseQueryGenerator(
        uri=uri,
        user=user,
        password=password,
        template_path=template_path,
        perturb_record_dir=perturb_record_dir,
        noise_prefix=noise_prefix,
        dataset=dataset,
    )
    
    try:
        generator.connect()
        generator.initialize()
        
        if show_schema:
            print(f"\n=== Schema 信息 ===")
            print(f"Labels: {list(generator.schema.labels.keys())}")
            print(f"Relationships: {list(generator.schema.relationships.keys())}")
            print(f"Total nodes: {generator.schema.total_nodes}")
            print(f"Total edges: {generator.schema.total_edges}")
            print(f"Target sample count: {generator.get_target_sample_count()}")
        
        if show_noise_info:
            print(f"\n=== 噪声数据信息 ===")
            print(f"噪声节点数量: {len(generator.noise_loader.noisy_node_set)}")
            print(f"噪声边数量: {len(generator.noise_loader.noisy_edges)}")
            print(f"噪声节点标签: {list(generator.noise_loader.noisy_nodes.keys())}")
            if generator.noise_loader.noisy_nodes:
                for label, node_ids in list(generator.noise_loader.noisy_nodes.items())[:5]:
                    print(f"  {label}: {len(node_ids)} 个噪声节点 (示例: {node_ids[:3]})")
        
        print(f"\n=== 开始生成噪声查询（第一步）===")
        # 使用实时输出，一边生成一边写入文件
        results = generator.generate_samples(
            target_count=target_count,
            realtime_output_path=output_file
        )
        
        print(f"\n=== 第一步生成结果 ===")
        print(f"共生成 {len(results)} 个查询")
        print(f"所有结果已实时写入文件: {output_file}")
        
        # 统计第一步的结果
        noise_query_count = 0
        clean_query_count = 0
        
        for r in results:
            verification_result = _verify_query_contains_noise(r, generator.noise_loader)
            if verification_result['contains_noise']:
                noise_query_count += 1
            else:
                clean_query_count += 1
        
        print(f"噪声查询数量: {noise_query_count}/{len(results)}")
        print(f"干净查询数量: {clean_query_count}/{len(results)}")
        
        return {
            "output_file": output_file,
            "results": results,
            "noise_query_count": noise_query_count,
            "clean_query_count": clean_query_count,
            "generator": generator
        }
        
    except Exception as e:
        logger.error(f"第一步测试失败: {e}")
        import traceback
        traceback.print_exc()
        generator.close()
        raise
    finally:
        # 注意：这里不关闭连接，让调用者决定何时关闭
        pass


def generate_step2_queries(uri: str, user: str, password: str, dataset: str,
                           perturb_record_dir: str, noise_prefix: str = None,
                           template_file_path: str = "query_template/template_judge1.json",
                           target_groups: int = 50,
                           max_iterations: int = 100,
                           max_unique_answers: int = 20,
                           output_file: str = None) -> dict:
    """
    第二步：根据 template_judge1.json 生成判断查询
    
    Args:
        uri: Neo4j URI
        user: Neo4j 用户名
        password: Neo4j 密码
        dataset: 数据集名称
        perturb_record_dir: 扰动记录文件夹路径
        noise_prefix: 噪声文件前缀
        template_file_path: 判断查询模板文件路径
        target_groups: 目标生成组数
        max_iterations: 最大迭代次数
        max_unique_answers: 最大独特答案数
        output_file: 输出文件路径（如果为None，则使用默认路径）
    
    Returns:
        包含以下字段的字典：
        - output_file: 输出文件路径
        - results: 生成的结果列表
        - total_unique_template: Template独特答案总数
        - total_unique_anti_template: Anti-template独特答案总数
        - template_counts: 各模板生成数量统计
        - generator: NoiseQueryGenerator实例（已关闭连接）
    """
    if output_file is None:
        output_file = f"noise_judge_query_results_step2_{dataset}.json"
    
    print("\n" + "="*80)
    print("第二步：根据 template_judge1.json 生成判断查询")
    print("="*80)
    
    generator = NoiseQueryGenerator(
        uri=uri,
        user=user,
        password=password,
        template_path="query_template/template.json",  # 这个路径在 generate_judge_queries 中会被覆盖
        perturb_record_dir=perturb_record_dir,
        noise_prefix=noise_prefix,
        dataset=dataset,
    )
    
    try:
        generator.connect()
        generator.initialize()
        
        print(f"\n=== 开始生成判断查询（第二步）===")
        
        all_results = []
        seen_queries = set()  # 用于去重，基于查询字符串
        
        # 循环生成查询，直到达到目标数量
        iteration = 0
        
        while len(all_results) < target_groups and iteration < max_iterations:
            iteration += 1
            print(f"\n--- 第 {iteration} 轮生成（当前已有 {len(all_results)} 组，目标 {target_groups} 组）---")
            
            # 使用 generate_judge_queries 方法生成查询
            results_batch = generator.generate_judge_queries(
                template_file_path=template_file_path,
                max_unique_answers=max_unique_answers
            )
            
            # 去重：只添加新的查询（基于 template_query 和 anti_template_query 的组合）
            new_count = 0
            for result in results_batch:
                # 使用查询字符串组合作为唯一标识
                query_key = (result["template_query"], result["anti_template_query"])
                if query_key not in seen_queries:
                    seen_queries.add(query_key)
                    all_results.append(result)
                    new_count += 1
            
            print(f"本轮生成 {len(results_batch)} 组查询，其中 {new_count} 组为新查询，累计 {len(all_results)} 组")
            
            # 如果本轮没有生成任何新结果，可能已经穷尽所有可能性，退出循环
            if new_count == 0:
                print("本轮未生成新查询，停止生成")
                break
            
            # 如果已达到目标数量，退出循环
            if len(all_results) >= target_groups:
                break
        
        # 如果超过目标数量，截取前 target_groups 个
        if len(all_results) > target_groups:
            all_results = all_results[:target_groups]
            print(f"\n已生成 {len(all_results)} 组查询（达到目标数量）")
        else:
            print(f"\n共生成 {len(all_results)} 组查询（未达到目标数量 {target_groups}，可能模板数量不足）")
        
        print(f"\n=== 第二步生成结果 ===")
        print(f"共处理 {len(all_results)} 组查询")
        
        # 保存结果到文件
        output_data = []
        for result in all_results:
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
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"所有结果已写入文件: {output_file}")
        
        # 统计第二步的结果
        total_unique_template = sum(r["unique_in_template_count"] for r in all_results)
        total_unique_anti_template = sum(r["unique_in_anti_template_count"] for r in all_results)
        
        # 按模板ID统计
        template_counts = {}
        for r in all_results:
            template_key = f"{r['template_type']}_{r['template_id']}"
            template_counts[template_key] = template_counts.get(template_key, 0) + 1
        
        print(f"\n=== 第二步统计信息 ===")
        print(f"生成的查询组数: {len(all_results)}")
        print(f"Template 独特答案总数: {total_unique_template}")
        print(f"Anti-template 独特答案总数: {total_unique_anti_template}")
        print(f"\n各模板生成数量:")
        for template_key, count in sorted(template_counts.items()):
            print(f"  {template_key}: {count} 组")
        
        # 显示前几个查询组的结果摘要
        for i, r in enumerate(all_results[:5]):
            print(f"\n[{i+1}] Template: {r['template_id']} ({r['template_type']})")
            print(f"    Template 查询结果数: {r['template_results_count']}")
            print(f"    Anti-template 查询结果数: {r['anti_template_results_count']}")
            print(f"    Template 独特答案数: {r['unique_in_template_count']}")
            print(f"    Anti-template 独特答案数: {r['unique_in_anti_template_count']}")
        
        if len(all_results) > 5:
            print(f"\n... 还有 {len(all_results) - 5} 组查询结果")
        
        return {
            "output_file": output_file,
            "results": all_results,
            "total_unique_template": total_unique_template,
            "total_unique_anti_template": total_unique_anti_template,
            "template_counts": template_counts,
            "generator": generator
        }
        
    except Exception as e:
        logger.error(f"第二步测试失败: {e}")
        import traceback
        traceback.print_exc()
        generator.close()
        raise
    finally:
        # 注意：这里不关闭连接，让调用者决定何时关闭
        pass


def clean_step1_results(input_file: str, output_file: str = None) -> str:
    """
    清洗第一步的结果文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为None，则自动生成）
    
    Returns:
        清洗后的文件路径
    """
    if output_file is None:
        output_file = input_file.replace(".json", "_cleaned.json")
    
    print(f"\n=== 开始清洗第一步结果文件 ===")
    try:
        extract_and_clean_answers(
            input_file=input_file,
            output_file=output_file,
            fields=["query", "answer", "is_noise_query"],
            fields_to_clean=["answer"],
            clean_method="normal"
        )
        print(f"第一步结果清洗完成，清洗后的文件: {output_file}")
        return output_file
    except Exception as e:
        logger.warning(f"清洗第一步结果文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def clean_step2_results(input_file: str, output_file: str = None) -> str:
    """
    清洗第二步的结果文件
    
    使用 clean_judge_query_answers 函数进行清洗，该函数会：
    1. 剔除 unique_in_template_answers 或 unique_in_anti_template_answers 里面只有一项且某个字段值为空列表的数据
    2. 清洗 unique_in_template_answers 和 unique_in_anti_template_answers 中的答案
    3. 将这两个字段合并为 CandidateSet 结构
    4. 只保留以下字段：template_query, anti_template_query, contains_noise, CandidateSet
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为None，则自动生成）
    
    Returns:
        清洗后的文件路径
    """
    if output_file is None:
        output_file = input_file.replace(".json", "_cleaned.json")
    
    print(f"\n=== 开始清洗第二步结果文件 ===")
    try:
        # 先使用 clean_judge_query_answers 进行清洗
        import tempfile
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            tmp_output = tmp_file.name
        
        clean_judge_query_answers(
            input_file=input_file,
            output_file=tmp_output
        )
        
        # 只保留必要字段
        with open(tmp_output, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)
        
        # 只保留必要字段
        essential_fields = ["template_query", "anti_template_query", "contains_noise", "CandidateSet"]
        filtered_data = []
        for item in cleaned_data:
            filtered_item = {}
            for field in essential_fields:
                if field in item:
                    filtered_item[field] = item[field]
            filtered_data.append(filtered_item)
        
        # 保存到最终输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        # 删除临时文件
        os.unlink(tmp_output)
        
        print(f"已过滤字段，只保留: {', '.join(essential_fields)}")
        print(f"第二步结果清洗完成，清洗后的文件: {output_file}")
        return output_file
    except Exception as e:
        logger.warning(f"清洗第二步结果文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise




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
        "--perturb_record_dir",
        type=str,
        default=DEFAULT_PERTURB_RECORD_DIR,
        help=f"扰动记录文件夹路径, 默认: {DEFAULT_PERTURB_RECORD_DIR}",
    )
    parser.add_argument(
        "--noise_prefix",
        type=str,
        default="ldbc_snb_finbench_noise_20260118_220611",
        help="噪声文件前缀，如果为None则自动查找最新的文件",
    )
    parser.add_argument(
        "--step1",
        action="store_true",
        help="执行第一步：生成噪声查询",
    )
    parser.add_argument(
        "--step2",
        action="store_true",
        help="执行第二步：生成判断查询",
    )
    parser.add_argument(
        "--clean_step1",
        type=str,
        default=None,
        help="清洗第一步结果文件（传入文件路径）",
    )
    parser.add_argument(
        "--clean_step2",
        type=str,
        default=None,
        help="清洗第二步结果文件（传入文件路径）",
    )
    parser.add_argument(
        "--step1_target_count",
        type=int,
        default=2000,
        help="第一步目标生成数量, 默认: 2000",
    )
    parser.add_argument(
        "--step2_target_groups",
        type=int,
        default=200,
        help="第二步目标生成组数, 默认: 200",
    )
    args = parser.parse_args()
    
    # 如果没有指定任何步骤，默认执行所有步骤
    if not (args.step1 or args.step2 or args.clean_step1 or args.clean_step2):
        args.step1 = True
        args.step2 = True
    
    generator_step1 = None
    generator_step2 = None
    
    try:
        # ========== 第一步：根据 template.json 生成查询 ==========
        if args.step1:
            print("\n" + "="*80)
            print("执行第一步：生成噪声查询")
            print("="*80)
            
            step1_result = generate_step1_queries(
                uri=args.uri,
                user=args.user,
                password=args.password,
                dataset=args.dataset,
                perturb_record_dir=args.perturb_record_dir,
                noise_prefix=args.noise_prefix,
                target_count=args.step1_target_count
            )
            generator_step1 = step1_result["generator"]
            output_file_step1 = step1_result["output_file"]
            
            # 自动清洗第一步的结果文件
            try:
                clean_step1_results(output_file_step1)
            except Exception as e:
                logger.warning(f"清洗第一步结果失败: {e}")
        
        # ========== 第二步：根据 template_judge1.json 生成判断查询 ==========
        if args.step2:
            print("\n" + "="*80)
            print("执行第二步：生成判断查询")
            print("="*80)
            
            step2_result = generate_step2_queries(
                uri=args.uri,
                user=args.user,
                password=args.password,
                dataset=args.dataset,
                perturb_record_dir=args.perturb_record_dir,
                noise_prefix=args.noise_prefix,
                target_groups=args.step2_target_groups
            )
            generator_step2 = step2_result["generator"]
            output_file_step2 = step2_result["output_file"]
            
            # 自动清洗第二步的结果文件
            try:
                clean_step2_results(output_file_step2)
            except Exception as e:
                logger.warning(f"清洗第二步结果失败: {e}")
        
        # ========== 独立清洗步骤 ==========
        if args.clean_step1:
            print("\n" + "="*80)
            print("清洗第一步结果文件")
            print("="*80)
            clean_step1_results(args.clean_step1)
        
        if args.clean_step2:
            print("\n" + "="*80)
            print("清洗第二步结果文件")
            print("="*80)
            clean_step2_results(args.clean_step2)
        
        # ========== 汇总输出 ==========
        print("\n" + "="*80)
        print("所有步骤完成！")
        print("="*80)
        if args.step1:
            output_file_step1 = f"noise_query_results_step1_{args.dataset}.json"
            print(f"第一步输出文件: {output_file_step1}")
            cleaned_file_step1 = output_file_step1.replace(".json", "_cleaned.json")
            if os.path.exists(cleaned_file_step1):
                print(f"第一步清洗后文件: {cleaned_file_step1}")
        if args.step2:
            output_file_step2 = f"noise_judge_query_results_step2_{args.dataset}.json"
            print(f"第二步输出文件: {output_file_step2}")
            cleaned_file_step2 = output_file_step2.replace(".json", "_cleaned.json")
            if os.path.exists(cleaned_file_step2):
                print(f"第二步清洗后文件: {cleaned_file_step2}")
    
    finally:
        # 关闭连接
        if generator_step1:
            generator_step1.close()
        if generator_step2:
            generator_step2.close()
