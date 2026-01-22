import json
import os
from typing import List, Dict, Any, Union


def clean_answer(answer: List[Dict[str, Any]]) -> Union[str, List]:
    """
    清洗答案数据（从 handler.py 复制）
    
    处理规则：
    1. 如果只有一个简单答案（如 {"min_value": "Aachen"}），提取值为字符串
    2. 如果答案包含数组值（如 {"vals": [...]}），提取为列表 [...]
    3. 如果答案包含多个元素，且每个元素有"a"和"b"键，提取它们的"_node_id"
    
    Args:
        answer: 原始答案列表
        
    Returns:
        清洗后的答案（字符串或列表）
    """
    if not answer:
        return []
    
    # 情况1：只有一个元素，且该元素只有一个键值对，提取值
    if len(answer) == 1:
        first_item = answer[0]
        if isinstance(first_item, dict):
            keys = list(first_item.keys())
            # 如果只有一个键，提取该键对应的值
            if len(keys) == 1:
                value = first_item[keys[0]]
                # 如果是简单类型（字符串、数字、布尔值），直接返回
                if isinstance(value, (str, int, float, bool)):
                    return value
                # 如果是列表，提取列表值（如 {"vals": [...]} → [...]）
                elif isinstance(value, list):
                    return value
    
    # 情况2：检查是否每个元素都有"a"和"b"键（pair类型）
    # 且"a"和"b"都是字典，包含"_node_id"字段
    is_pair_type = True
    for item in answer:
        if not isinstance(item, dict):
            is_pair_type = False
            break
        if "a" not in item or "b" not in item:
            is_pair_type = False
            break
        # 检查a和b是否是字典且包含_node_id
        a_val = item.get("a", {})
        b_val = item.get("b", {})
        if not isinstance(a_val, dict) or not isinstance(b_val, dict):
            is_pair_type = False
            break
        if "_node_id" not in a_val or "_node_id" not in b_val:
            is_pair_type = False
            break
    
    if is_pair_type:
        # 提取每个pair的_node_id
        cleaned_pairs = []
        for item in answer:
            a_node_id = item.get("a", {}).get("_node_id", "")
            b_node_id = item.get("b", {}).get("_node_id", "")
            cleaned_pairs.append({
                "a": a_node_id,
                "b": b_node_id
            })
        return cleaned_pairs
    
    # 情况3：其他情况，保持原样或返回列表
    return answer


def process_noise_query_results(
    input_file: str,
    output_file: str,
    nlp_source_file: str
) -> None:
    """
    处理噪声查询执行结果
    
    从 noise_query_execution_results.json 中提取：
    - query
    - original_answer (重命名为 clean_answer，并清洗)
    - execution_result (重命名为 noise_answer，并清洗)
    - same_as_cleangraph
    并从 nlp_source_file 中提取对应的 nlp 字段
    
    Args:
        input_file: 输入的 noise_query_execution_results.json 文件路径
        output_file: 输出的 JSON 文件路径
        nlp_source_file: 包含 nlp 字段的源文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 读取 nlp 源文件并建立 query -> nlp 的映射
    with open(nlp_source_file, 'r', encoding='utf-8') as f:
        nlp_data = json.load(f)
    
    # 建立 query -> nlp 的映射
    query_to_nlp = {}
    for item in nlp_data:
        query = item.get("query", "")
        nlp = item.get("nlp", "")
        if query:
            query_to_nlp[query] = nlp
    
    # 处理每一项
    processed_data = []
    for item in data:
        query = item.get("query", "")
        original_answer = item.get("original_answer", [])
        execution_result = item.get("execution_result", [])
        same_as_cleangraph = item.get("same_as_cleangraph", False)
        
        # 清洗答案
        clean_answer_value = clean_answer(original_answer)
        noise_answer_value = clean_answer(execution_result)
        
        # 获取对应的 nlp
        nlp_value = query_to_nlp.get(query, "")
        
        # 构建处理后的项
        processed_item = {
            "query": query,
            "clean_answer": clean_answer_value,
            "noise_answer": noise_answer_value,
            "nlp": nlp_value,
            "same_as_cleangraph": same_as_cleangraph
        }
        
        processed_data.append(processed_item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共处理 {len(processed_data)} 条记录")
    print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    # 设置文件路径
    input_file = os.path.join(
        os.path.dirname(__file__),
        "noise_query_execution_results.json"
    )
    output_file = os.path.join(
        os.path.dirname(__file__),
        "noise_query_execution_results_cleaned.json"
    )
    nlp_source_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "query_gen",
        "query",
        "ldbc_snb_finbench",
        "noise_query_results_ldbcfin_cleaned_with_nlp_fixed.json"
    )
    
    # 执行处理
    process_noise_query_results(input_file, output_file, nlp_source_file)
