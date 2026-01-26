#!/usr/bin/env python3
"""
脚本功能：
1. 将 noise_judge_query_results_step2_ldbcfin_cleaned_with_noise_candidates.json 文件中
   NoiseCandidateSet 的 valid_answer 和 invalid_answer 只保留前20项
2. 给所有 CandidateSet 下面的 valid_answer 每项加一个 "judge"，值设置为 true
3. 给所有 CandidateSet 下面的 invalid_answer 每项加一个 "judge"，值设置为 false
4. 给所有 NoiseCandidateSet 下面的 valid_answer 每项加一个 "judge"，值设置为 true
5. 给所有 NoiseCandidateSet 下面的 invalid_answer 每项加一个 "judge"，值设置为 false
6. 清洗数据：将 {"_node_id": "Medium:17592186044627"} 这样的结构简化为 "Medium:17592186044627"
7. 将 "CandidateSet" 重命名为 "clean_answer"
8. 将 "NoiseCandidateSet" 重命名为 "noise_answer"
"""

import json
import sys
from pathlib import Path


def clean_node_id_structure(obj):
    """
    递归清洗数据，将 {"_node_id": "value"} 结构简化为 "value"
    
    Args:
        obj: 要处理的对象（可以是 dict, list, 或其他类型）
    
    Returns:
        清洗后的对象
    """
    if isinstance(obj, dict):
        # 如果是一个只包含 "_node_id" 键的字典，直接返回其值
        if len(obj) == 1 and "_node_id" in obj:
            return obj["_node_id"]
        
        # 否则递归处理所有值
        return {key: clean_node_id_structure(value) for key, value in obj.items()}
    
    elif isinstance(obj, list):
        # 递归处理列表中的每个元素
        return [clean_node_id_structure(item) for item in obj]
    
    else:
        # 其他类型直接返回
        return obj


def process_candidate_set(data):
    """
    处理 JSON 数据，执行以下操作：
    1. 将 NoiseCandidateSet 的 valid_answer 和 invalid_answer 只保留前20项
    2. 给所有 CandidateSet 下面的 valid_answer 每项加一个 "judge"，值设置为 true
    3. 给所有 CandidateSet 下面的 invalid_answer 每项加一个 "judge"，值设置为 false
    4. 给所有 NoiseCandidateSet 下面的 valid_answer 每项加一个 "judge"，值设置为 true
    5. 给所有 NoiseCandidateSet 下面的 invalid_answer 每项加一个 "judge"，值设置为 false
    6. 清洗数据：将 {"_node_id": "Medium:17592186044627"} 这样的结构简化为 "Medium:17592186044627"
    7. 将 "CandidateSet" 重命名为 "clean_answer"
    8. 将 "NoiseCandidateSet" 重命名为 "noise_answer"
    
    Args:
        data: JSON 数据（应该是列表）
    """
    if not isinstance(data, list):
        print("错误：JSON 文件应该是一个数组")
        sys.exit(1)
    
    print(f"找到 {len(data)} 条记录")
    
    processed_noise_count = 0
    processed_candidate_count = 0
    judge_added_valid_count = 0
    judge_added_invalid_count = 0
    cleaned_count = 0
    
    # 处理每条记录
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        
        # 处理 NoiseCandidateSet
        if 'NoiseCandidateSet' in item and isinstance(item['NoiseCandidateSet'], dict):
            noise_set = item['NoiseCandidateSet']
            
            # 处理 valid_answer：截断、添加 judge 字段并清洗数据
            if 'valid_answer' in noise_set and isinstance(noise_set['valid_answer'], list):
                original_len = len(noise_set['valid_answer'])
                noise_set['valid_answer'] = noise_set['valid_answer'][:20]
                if original_len > 20:
                    processed_noise_count += 1
                    print(f"  记录 {i+1}: NoiseCandidateSet.valid_answer 从 {original_len} 项缩减到 {len(noise_set['valid_answer'])} 项")
                
                # 添加 judge 字段并清洗数据
                for idx, answer_item in enumerate(noise_set['valid_answer']):
                    if isinstance(answer_item, dict):
                        # 添加 judge 字段
                        if 'judge' not in answer_item:
                            answer_item['judge'] = True
                            judge_added_valid_count += 1
                        
                        # 清洗 _node_id 结构（递归处理整个对象）
                        cleaned_item = clean_node_id_structure(answer_item)
                        if isinstance(cleaned_item, dict):
                            # 更新原对象
                            answer_item.clear()
                            answer_item.update(cleaned_item)
                            cleaned_count += 1
                        else:
                            # 如果清洗后不是字典（理论上不应该发生），保持原样
                            noise_set['valid_answer'][idx] = cleaned_item
                            cleaned_count += 1
            
            # 处理 invalid_answer：截断、添加 judge 字段并清洗数据
            if 'invalid_answer' in noise_set and isinstance(noise_set['invalid_answer'], list):
                original_len = len(noise_set['invalid_answer'])
                noise_set['invalid_answer'] = noise_set['invalid_answer'][:20]
                if original_len > 20:
                    processed_noise_count += 1
                    print(f"  记录 {i+1}: NoiseCandidateSet.invalid_answer 从 {original_len} 项缩减到 {len(noise_set['invalid_answer'])} 项")
                
                # 添加 judge 字段并清洗数据
                for idx, answer_item in enumerate(noise_set['invalid_answer']):
                    if isinstance(answer_item, dict):
                        # 添加 judge 字段
                        if 'judge' not in answer_item:
                            answer_item['judge'] = False
                            judge_added_invalid_count += 1
                        
                        # 清洗 _node_id 结构（递归处理整个对象）
                        cleaned_item = clean_node_id_structure(answer_item)
                        if isinstance(cleaned_item, dict):
                            # 更新原对象
                            answer_item.clear()
                            answer_item.update(cleaned_item)
                            cleaned_count += 1
                        else:
                            # 如果清洗后不是字典（理论上不应该发生），保持原样
                            noise_set['invalid_answer'][idx] = cleaned_item
                            cleaned_count += 1
        
        # 处理 CandidateSet
        if 'CandidateSet' in item and isinstance(item['CandidateSet'], dict):
            candidate_set = item['CandidateSet']
            processed_candidate_count += 1
            
            # 处理 valid_answer：添加 judge 字段并清洗数据
            if 'valid_answer' in candidate_set and isinstance(candidate_set['valid_answer'], list):
                for idx, answer_item in enumerate(candidate_set['valid_answer']):
                    if isinstance(answer_item, dict):
                        # 添加 judge 字段
                        if 'judge' not in answer_item:
                            answer_item['judge'] = True
                            judge_added_valid_count += 1
                        
                        # 清洗 _node_id 结构（递归处理整个对象）
                        cleaned_item = clean_node_id_structure(answer_item)
                        if isinstance(cleaned_item, dict):
                            # 更新原对象
                            answer_item.clear()
                            answer_item.update(cleaned_item)
                            cleaned_count += 1
                        else:
                            # 如果清洗后不是字典（理论上不应该发生），保持原样
                            candidate_set['valid_answer'][idx] = cleaned_item
                            cleaned_count += 1
            
            # 处理 invalid_answer：添加 judge 字段并清洗数据
            if 'invalid_answer' in candidate_set and isinstance(candidate_set['invalid_answer'], list):
                for idx, answer_item in enumerate(candidate_set['invalid_answer']):
                    if isinstance(answer_item, dict):
                        # 添加 judge 字段
                        if 'judge' not in answer_item:
                            answer_item['judge'] = False
                            judge_added_invalid_count += 1
                        
                        # 清洗 _node_id 结构（递归处理整个对象）
                        cleaned_item = clean_node_id_structure(answer_item)
                        if isinstance(cleaned_item, dict):
                            # 更新原对象
                            answer_item.clear()
                            answer_item.update(cleaned_item)
                            cleaned_count += 1
                        else:
                            # 如果清洗后不是字典（理论上不应该发生），保持原样
                            candidate_set['invalid_answer'][idx] = cleaned_item
                            cleaned_count += 1
    
    # 重命名键名：CandidateSet -> clean_answer, NoiseCandidateSet -> noise_answer
    renamed_candidate_count = 0
    renamed_noise_count = 0
    for item in data:
        if isinstance(item, dict):
            if 'CandidateSet' in item:
                item['clean_answer'] = item.pop('CandidateSet')
                renamed_candidate_count += 1
            if 'NoiseCandidateSet' in item:
                item['noise_answer'] = item.pop('NoiseCandidateSet')
                renamed_noise_count += 1
    
    print(f"\n处理统计：")
    print(f"  - 处理了 {processed_candidate_count} 个 CandidateSet")
    print(f"  - 添加了 {judge_added_valid_count} 个 valid_answer 的 judge 字段 (true)")
    print(f"  - 添加了 {judge_added_invalid_count} 个 invalid_answer 的 judge 字段 (false)")
    print(f"  - 清洗了 {cleaned_count} 个数据项")
    print(f"  - 处理了 {processed_noise_count} 个 NoiseCandidateSet 字段的截断")
    print(f"  - 重命名了 {renamed_candidate_count} 个 CandidateSet -> clean_answer")
    print(f"  - 重命名了 {renamed_noise_count} 个 NoiseCandidateSet -> noise_answer")


def filter_noise_candidates(input_file, output_file=None):
    """
    处理 JSON 文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为 None 则覆盖原文件
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"错误：文件 {input_file} 不存在")
        sys.exit(1)
    
    print(f"正在读取文件: {input_file}")
    
    # 读取 JSON 文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取文件失败: {e}")
        sys.exit(1)
    
    # 处理数据
    process_candidate_set(data)
    
    # 确定输出文件路径
    if output_file is None:
        output_path = input_path
        print(f"\n将覆盖原文件: {output_path}")
    else:
        output_path = Path(output_file)
        print(f"\n将保存到: {output_path}")
    
    # 保存处理后的数据
    print("正在保存文件...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"保存成功: {output_path}")
    except Exception as e:
        print(f"错误：保存文件失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # 默认输入文件
    default_input = 'noise_judge_query_results_step2_ldbcfin_cleaned_with_noise_candidates.json'
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 确定输入文件路径
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = script_dir / default_input
    
    # 确定输出文件路径
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None  # 默认覆盖原文件
    
    filter_noise_candidates(input_file, output_file)
