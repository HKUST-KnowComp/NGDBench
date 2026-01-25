import json
import re
import os
from typing import List, Dict, Any, Union, Optional
# NOTE: 该模块的大部分清洗函数不依赖 OpenAI。
# 为了让离线环境也能使用清洗能力，这里将 openai 依赖变为可选。
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

def extract_and_clean_answers(
    input_file: str, 
    output_file: str,
    fields: Optional[List[str]] = None,
    fields_to_clean: Optional[List[str]] = None,
    default_values: Optional[Dict[str, Any]] = None,
    node_id_key: str = "_node_id",
    clean_method: str = "normal"
) -> None:
    """
    函数1：提取答案并清洗
    
    从JSON文件中提取指定的字段，并对指定字段进行清洗处理。
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        fields: 要提取的字段列表，默认为 ["query", "answer", "is_noise_query"]
        fields_to_clean: 需要清洗的字段列表，默认为 ["answer"]
        default_values: 字段的默认值字典，例如 {"query": "", "is_noise_query": False}
        node_id_key: 用于提取节点ID的键名，默认为 "_node_id"
        clean_method: 清洗方法，可选 "normal" 或 "judge"，默认为 "normal"
            - "normal": 使用 clean_normal_answer 进行普通清洗
            - "judge": 使用 _clean_judge_answers 进行 judge 查询结果清洗
    
    Examples:
        # 使用普通清洗方法
        extract_and_clean_answers(
            "input.json",
            "output.json",
            clean_method="normal"
        )
        
        # 使用 judge 清洗方法
        extract_and_clean_answers(
            "input.json",
            "output.json",
            clean_method="judge"
        )
    """
    # 设置默认值
    if fields is None:
        fields = ["query", "answer", "is_noise_query"]
    if fields_to_clean is None:
        fields_to_clean = ["answer"]
    if default_values is None:
        default_values = {"query": ""}
    
    # 验证 clean_method 参数
    if clean_method not in ["normal", "judge"]:
        raise ValueError(f"clean_method 必须是 'normal' 或 'judge'，当前值为: {clean_method}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    
    for item in data:
        cleaned_item = {}
        for field in fields:
            # 获取字段值，使用默认值或 None
            field_value = item.get(field, default_values.get(field, None))
            
            # 如果字段需要清洗，根据 clean_method 选择清洗方法
            if field in fields_to_clean and isinstance(field_value, list):
                if clean_method == "normal":
                    field_value = clean_normal_answer(field_value, node_id_key=node_id_key)
                elif clean_method == "judge":
                    field_value = _clean_judge_answers(field_value)
            
            cleaned_item[field] = field_value
        
        cleaned_data.append(cleaned_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)


def clean_normal_answer(
    answer: List[Dict[str, Any]], 
    node_id_key: str = "_node_id"
) -> Union[str, List]:
    """
    清洗答案数据
    
    处理规则：
    1. 如果只有一个简单答案（如 {"min_value": "Aachen"}），提取值为字符串
    2. 如果答案包含数组值（如 {"vals": [...]}），提取为列表 [...]
    3. 如果答案中每个元素都是字典，且每个字典只有一个键，该键对应的值也是字典且包含node_id_key字段，则只保留node_id_key对应的值
    4. 如果答案包含多个元素，且每个元素有"a"和"b"键，提取它们的node_id_key字段
    
    Args:
        answer: 原始答案列表
        node_id_key: 用于提取节点ID的键名，默认为 "_node_id"
        
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
                # 如果是字典且包含node_id_key，提取node_id_key对应的值
                elif isinstance(value, dict) and node_id_key in value:
                    return value.get(node_id_key, "")
    
    # 情况2：检查是否每个元素都是字典，且每个字典只有一个键，该键对应的值也是字典且包含node_id_key字段
    is_nested_dict_type = True
    for item in answer:
        if not isinstance(item, dict):
            is_nested_dict_type = False
            break
        keys = list(item.keys())
        # 每个元素必须只有一个键
        if len(keys) != 1:
            is_nested_dict_type = False
            break
        # 该键对应的值必须是字典
        inner_dict = item[keys[0]]
        if not isinstance(inner_dict, dict):
            is_nested_dict_type = False
            break
        # 该字典必须包含node_id_key字段
        if node_id_key not in inner_dict:
            is_nested_dict_type = False
            break
    
    if is_nested_dict_type:
        # 提取每个元素中node_id_key对应的值
        cleaned_values = []
        for item in answer:
            keys = list(item.keys())
            if keys:
                inner_dict = item[keys[0]]
                node_id_value = inner_dict.get(node_id_key, "")
                cleaned_values.append(node_id_value)
        return cleaned_values
    
    # 情况4：检查是否每个元素都有"a"和"b"键（pair类型）
    # 且"a"和"b"都是字典，包含node_id_key字段
    is_pair_type = True
    for item in answer:
        if not isinstance(item, dict):
            is_pair_type = False
            break
        if "a" not in item or "b" not in item:
            is_pair_type = False
            break
        # 检查a和b是否是字典且包含node_id_key
        a_val = item.get("a", {})
        b_val = item.get("b", {})
        if not isinstance(a_val, dict) or not isinstance(b_val, dict):
            is_pair_type = False
            break
        if node_id_key not in a_val or node_id_key not in b_val:
            is_pair_type = False
            break
    
    if is_pair_type:
        # 提取每个pair的node_id_key
        cleaned_pairs = []
        for item in answer:
            a_node_id = item.get("a", {}).get(node_id_key, "")
            b_node_id = item.get("b", {}).get(node_id_key, "")
            cleaned_pairs.append({
                "a": a_node_id,
                "b": b_node_id
            })
        return cleaned_pairs
    
    # 情况5：检查是否是 [{"key": "value"}, {"key": "value"}] 格式，其中值是简单类型（字符串、数字、布尔值、None）
    # 这种情况需要提取为 ["value", "value"] 或 [value, value]
    is_single_key_simple_type = True
    if len(answer) > 0:
        first_item = answer[0]
        if isinstance(first_item, dict) and len(first_item) == 1:
            first_key = list(first_item.keys())[0]
            first_value = first_item[first_key]
            # 第一个值是简单类型（字符串、数字、布尔值、None）
            if isinstance(first_value, (str, int, float, bool)) or first_value is None:
                # 检查所有元素是否都是这种格式
                for item in answer:
                    if not isinstance(item, dict) or len(item) != 1:
                        is_single_key_simple_type = False
                        break
                    key = list(item.keys())[0]
                    value = item[key]
                    # 值必须是简单类型（包括 None）
                    if not (isinstance(value, (str, int, float, bool)) or value is None):
                        is_single_key_simple_type = False
                        break
                if is_single_key_simple_type:
                    # 提取所有值组成列表
                    cleaned_values = [item[list(item.keys())[0]] for item in answer]
                    return cleaned_values
    
    # 情况6：其他情况，保持原样或返回列表
    return answer


def remove_empty_answers(
    input_file: str,
    output_file: str,
    answer_field: str = "answer"
) -> None:
    """
    函数5：移除answer字段为0或空列表的项
    
    从JSON文件中移除answer字段为0或[]的item，保留其他所有项。
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        answer_field: answer字段名，默认为 "answer"
    
    Examples:
        # 移除answer为0或[]的项
        remove_empty_answers(
            "query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned_with_nlp_fixed.json",
            "query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned_with_nlp_fixed_filtered.json"
        )
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤掉answer为0或[]的项
    filtered_items = []
    removed_count = 0
    
    for item in data:
        answer = item.get(answer_field)
        
        # 检查answer是否为0（数字0）或空列表[]
        if answer == 0 or answer == []:
            removed_count += 1
            continue
        
        filtered_items.append(item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_items, f, ensure_ascii=False, indent=2)
    
    print(f"移除了 {removed_count} 个answer为0或[]的项（共 {len(data)} 个）")
    print(f"保留了 {len(filtered_items)} 个项")
    print(f"结果已保存到 {output_file}")


def filter_queries_by_pattern(
    input_file: str,
    output_file: str,
    pattern: str,
    case_sensitive: bool = False,
    query_field: str = "query"
) -> None:
    """
    函数4：根据特定模式过滤查询
    
    从JSON文件中找出包含特定模式的查询对应的item，并保存到新文件中。
    支持正则表达式模式匹配。
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        pattern: 要搜索的模式（支持正则表达式），例如 "order by" 或 "ORDER BY"
        case_sensitive: 是否区分大小写，默认为False（不区分大小写）
        query_field: 查询字段名，默认为 "query"
    
    Examples:
        # 查找包含 "order by" 的查询（不区分大小写）
        filter_queries_by_pattern(
            "input.json",
            "output.json",
            "order by"
        )
        
        # 查找包含 "LIMIT" 的查询（区分大小写）
        filter_queries_by_pattern(
            "input.json",
            "output.json",
            "LIMIT",
            case_sensitive=True
        )
        
        # 使用正则表达式查找以 "MATCH" 开头且包含 "WHERE" 的查询
        filter_queries_by_pattern(
            "input.json",
            "output.json",
            r"^MATCH.*WHERE"
        )
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 编译正则表达式
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"无效的正则表达式模式: {pattern}. 错误: {e}")
    
    # 过滤匹配的items
    matched_items = []
    for item in data:
        query = item.get(query_field, "")
        if isinstance(query, str) and regex.search(query):
            matched_items.append(item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_items, f, ensure_ascii=False, indent=2)
    
    print(f"找到 {len(matched_items)} 个匹配的查询（共 {len(data)} 个）")
    print(f"结果已保存到 {output_file}")


def clean_judge_query_answers(
    input_file: str,
    output_file: str
) -> None:
    """
    清洗 judge 查询结果文件
    
    功能：
    1. 剔除 unique_in_template_answers 和 unique_in_anti_template_answers 都为空列表的数据
    2. 剔除 unique_in_template_answers 或 unique_in_anti_template_answers 里面只有一项且某个字段值为空列表的数据
    3. 对于 unique_in_template_answers 和 unique_in_anti_template_answers 里面包含的答案：
       - 如果包含 "a" 和 "b" 字段（pair类型）："a" 和 "b" 字段都只保留 "_node_id"
       - 如果包含 "a" 和 "bs" 字段："a" 字段只保留 "_node_id"，"bs" 字段去重（保留去重后的唯一值或前5个结果）
    4. 将 unique_in_template_answers 和 unique_in_anti_template_answers 合并为 CandidateSet 字段：
       - unique_in_template_answers → CandidateSet.valid_answer
       - unique_in_anti_template_answers → CandidateSet.invalid_answer
       - 原字段会被移除
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
    
    Examples:
        clean_judge_query_results(
            "noise_judge_query_results_step2_ldbcfin.json",
            "noise_judge_query_results_step2_ldbcfin_cleaned.json"
        )
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    removed_count = 0
    cleaned_answers_count = 0
    
    for item in data:
        # 检查 unique_in_template_answers 或 unique_in_anti_template_answers 是否需要剔除
        unique_template_answers = item.get("unique_in_template_answers", [])
        unique_anti_template_answers = item.get("unique_in_anti_template_answers", [])
        
        # 规则0: 剔除 unique_in_template_answers 和 unique_in_anti_template_answers 都为空列表的数据
        should_remove = False
        if len(unique_template_answers) == 0 and len(unique_anti_template_answers) == 0:
            should_remove = True
            removed_count += 1
            continue
        
        # 规则1: 剔除 unique_in_template_answers 里面只有一项且某个字段值为空列表的数据
        if len(unique_template_answers) == 1:
            first_answer = unique_template_answers[0]
            if isinstance(first_answer, dict):
                # 检查是否只有一个键且该键对应的值是空列表
                keys = list(first_answer.keys())
                if len(keys) == 1:
                    key = keys[0]
                    value = first_answer[key]
                    if isinstance(value, list) and len(value) == 0:
                        should_remove = True
                        removed_count += 1
                        continue
        
        # 规则2: 剔除 unique_in_anti_template_answers 里面只有一项且某个字段值为空列表的数据
        if not should_remove and len(unique_anti_template_answers) == 1:
            first_anti_answer = unique_anti_template_answers[0]
            if isinstance(first_anti_answer, dict):
                # 检查是否只有一个键且该键对应的值是空列表
                keys = list(first_anti_answer.keys())
                if len(keys) == 1:
                    key = keys[0]
                    value = first_anti_answer[key]
                    if isinstance(value, list) and len(value) == 0:
                        should_remove = True
                        removed_count += 1
                        continue
        
        # 如果不需要剔除，则进行清洗
        cleaned_item = item.copy()
        
        # 清洗 unique_in_template_answers
        cleaned_template_answers = []
        if "unique_in_template_answers" in cleaned_item:
            cleaned_template_answers = _clean_judge_answers(
                cleaned_item["unique_in_template_answers"]
            )
            if cleaned_template_answers != cleaned_item["unique_in_template_answers"]:
                cleaned_answers_count += 1
        
        # 清洗 unique_in_anti_template_answers
        cleaned_anti_template_answers = []
        if "unique_in_anti_template_answers" in cleaned_item:
            cleaned_anti_template_answers = _clean_judge_answers(
                cleaned_item["unique_in_anti_template_answers"]
            )
            if cleaned_anti_template_answers != cleaned_item["unique_in_anti_template_answers"]:
                cleaned_answers_count += 1
        
        # 合并为 CandidateSet 结构
        cleaned_item["CandidateSet"] = {
            "valid_answer": cleaned_template_answers,
            "invalid_answer": cleaned_anti_template_answers
        }
        
        # 移除原来的字段
        cleaned_item.pop("unique_in_template_answers", None)
        cleaned_item.pop("unique_in_anti_template_answers", None)
        
        cleaned_data.append(cleaned_item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"移除了 {removed_count} 个 unique_in_template_answers 或 unique_in_anti_template_answers 为空的数据项")
    print(f"清洗了 {cleaned_answers_count} 个答案字段")
    print(f"保留了 {len(cleaned_data)} 个数据项（共 {len(data)} 个）")
    print(f"结果已保存到 {output_file}")



def _clean_judge_answers(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清洗 judge 查询结果中的答案列表
    
    对于包含 "a" 和 "bs" 字段的答案：
    - "a" 字段只保留 "_node_id"
    - "bs" 字段去重（保留去重后的唯一值或前5个结果）
    
    对于包含 "a" 和 "b" 字段的答案（pair类型）：
    - "a" 字段只保留 "_node_id"
    - "b" 字段只保留 "_node_id"
    
    Args:
        answers: 答案列表
    
    Returns:
        清洗后的答案列表
    """
    cleaned_answers = []
    
    for answer in answers:
        if not isinstance(answer, dict):
            cleaned_answers.append(answer)
            continue
        
        cleaned_answer = answer.copy()
        
        # 处理 "a" 字段：只保留 "_node_id"
        if "a" in cleaned_answer and isinstance(cleaned_answer["a"], dict):
            a_dict = cleaned_answer["a"]
            if "_node_id" in a_dict:
                cleaned_answer["a"] = {"_node_id": a_dict["_node_id"]}
            else:
                # 如果没有 _node_id，保留空字典或移除该字段
                cleaned_answer["a"] = {}
        
        # 处理 "b" 字段：只保留 "_node_id"（用于pair类型的答案）
        if "b" in cleaned_answer and isinstance(cleaned_answer["b"], dict):
            b_dict = cleaned_answer["b"]
            if "_node_id" in b_dict:
                cleaned_answer["b"] = {"_node_id": b_dict["_node_id"]}
            else:
                # 如果没有 _node_id，保留空字典或移除该字段
                cleaned_answer["b"] = {}
        
        # 处理 "bs" 字段：去重并保留前5个
        if "bs" in cleaned_answer and isinstance(cleaned_answer["bs"], list):
            bs_list = cleaned_answer["bs"]
            # 去重：使用列表保持顺序，但只保留第一次出现的值
            seen = set()
            unique_bs = []
            for item in bs_list:
                # 将 item 转换为可哈希的类型
                # 对于可哈希类型（str, int, float, bool, None），直接使用
                # 对于不可哈希类型（list, dict），转换为 JSON 字符串
                if isinstance(item, (str, int, float, bool, type(None))):
                    item_key = item
                else:
                    item_key = json.dumps(item, sort_keys=True)
                
                if item_key not in seen:
                    seen.add(item_key)
                    unique_bs.append(item)
            
            # 保留前5个结果
            cleaned_answer["bs"] = unique_bs[:5]
        
        cleaned_answers.append(cleaned_answer)
    
    return cleaned_answers


def clean_validation_fields(
    data: List[Dict[str, Any]],
    node_id_key: str = "_node_id",
) -> List[Dict[str, Any]]:
    """
    清洗验证字段：只保留指定字段
    
    对于每个项，只保留：
    - pre_validation 的 query 和 answer
    - template 的 query
    - post_validation 的 query 和 answer
    
    Args:
        data: 输入的数据列表
    
    Returns:
        清洗后的数据列表
    
    Examples:
        with open("input.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        cleaned_data = clean_validation_fields(data, node_id_key="_node_id")
    """
    cleaned_data = []
    
    for item in data:
        pre_validation = item.get("pre_validation", {})
        post_validation = item.get("post_validation", {})
        template = item.get("template", {})

        # 在“保留字段”之后，对 answer 做一次标准清洗
        pre_answer = pre_validation.get("answer")
        if isinstance(pre_answer, list):
            pre_answer = clean_normal_answer(pre_answer, node_id_key=node_id_key)

        post_answer = post_validation.get("answer")
        if isinstance(post_answer, list):
            post_answer = clean_normal_answer(post_answer, node_id_key=node_id_key)
        
        # 构建新的项，只保留指定字段
        cleaned_item = {
            "pre_validation": {
                "query": pre_validation.get("query"),
                "answer": pre_answer,
            },
            "template": {
                "query": template.get("query")
            },
            "post_validation": {
                "query": post_validation.get("query"),
                "answer": post_answer,
            }
        }
        cleaned_data.append(cleaned_item)
    
    return cleaned_data


def filter_validation_differences(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    筛选验证结果不同的项
    
    筛选出 pre_validation 和 post_validation 的 answer 不一样的项。
    
    Args:
        data: 输入的数据列表
    
    Returns:
        筛选后的数据列表（只包含 answer 不同的项）
    
    Examples:
        with open("input.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        filtered_data = filter_validation_differences(data)
    """
    filtered_data = []
    
    for item in data:
        pre_validation = item.get("pre_validation", {})
        post_validation = item.get("post_validation", {})
        
        # 获取 pre_validation 和 post_validation 的 answer
        pre_answer = pre_validation.get("answer")
        post_answer = post_validation.get("answer")
        
        # 比较两个 answer 是否不同（使用 JSON 序列化进行深度比较）
        pre_answer_str = json.dumps(pre_answer, sort_keys=True) if pre_answer is not None else None
        post_answer_str = json.dumps(post_answer, sort_keys=True) if post_answer is not None else None
        
        # 如果 answer 不同，则保留该项
        if pre_answer_str != post_answer_str:
            filtered_data.append(item)
    
    return filtered_data


def clean_validation_fields_from_file(
    input_file: str,
    output_file: str
) -> None:
    """
    从文件读取数据并清洗验证字段（只清洗，不筛选）
    
    对于每个项，只保留：
    - pre_validation 的 query 和 answer
    - template 的 query
    - post_validation 的 query 和 answer
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
    
    Examples:
        clean_validation_fields_from_file(
            "management_query_ldbc_fin.json",
            "management_query_ldbc_fin_cleaned.json"
        )
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = clean_validation_fields(data)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"清洗了 {len(cleaned_data)} 个数据项（共 {len(data)} 个）")
    print(f"结果已保存到 {output_file}")


def filter_and_clean_validation_differences(
    input_file: str,
    output_file: str
) -> None:
    """
    筛选并清洗验证结果不同的项（组合函数）
    
    功能：
    1. 筛选出 pre_validation 和 post_validation 的 answer 不一样的项
    2. 对于筛选出的项，只保留：
       - pre_validation 的 query 和 answer
       - template 的 query
       - post_validation 的 query 和 answer
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
    
    Examples:
        filter_and_clean_validation_differences(
            "management_query_ldbc_fin.json",
            "management_query_ldbc_fin_filtered.json"
        )
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    
    # 先筛选，再清洗
    filtered_data = filter_validation_differences(data)
    cleaned_data = clean_validation_fields(filtered_data)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"筛选出 {len(cleaned_data)} 个 pre_validation 和 post_validation answer 不同的项（共 {total_count} 个）")
    print(f"结果已保存到 {output_file}")

def process_noise_query_results(
    input_file: str,
    output_file: str,
) -> None:
    """
    处理噪声查询执行结果
    
    从 noise_query_execution_results.json 中提取并清洗字段：
    - query
    - original_answer (重命名为 clean_answer，并清洗)
    - execution_result (重命名为 noise_answer，并清洗)
    - same_as_cleangraph
    
    Args:
        input_file: 输入的 noise_query_execution_results.json 文件路径
        output_file: 输出的 JSON 文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
        
        # 构建处理后的项（只保留 query / clean_answer / noise_answer / same_as_cleangraph）
        processed_item = {
            "query": query,
            "clean_answer": clean_answer_value,
            "noise_answer": noise_answer_value,
            "same_as_cleangraph": same_as_cleangraph
        }
        
        processed_data.append(processed_item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共处理 {len(processed_data)} 条记录")
    print(f"结果已保存到 {output_file}")

