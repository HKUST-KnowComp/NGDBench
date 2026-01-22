import json
import re
import os
from typing import List, Dict, Any, Union, Optional
from openai import OpenAI


def extract_and_clean_answers(
    input_file: str, 
    output_file: str,
    fields: Optional[List[str]] = None,
    fields_to_clean: Optional[List[str]] = None,
    default_values: Optional[Dict[str, Any]] = None,
    node_id_key: str = "_node_id"
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
    """
    # 设置默认值
    if fields is None:
        fields = ["query", "answer", "is_noise_query"]
    if fields_to_clean is None:
        fields_to_clean = ["answer"]
    if default_values is None:
        default_values = {"query": ""}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    
    for item in data:
        cleaned_item = {}
        for field in fields:
            # 获取字段值，使用默认值或 None
            field_value = item.get(field, default_values.get(field, None))
            
            # 如果字段需要清洗，则调用 clean_answer
            if field in fields_to_clean and isinstance(field_value, list):
                field_value = clean_answer(field_value, node_id_key=node_id_key)
            
            cleaned_item[field] = field_value
        
        cleaned_data.append(cleaned_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)


def clean_answer(
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
    
    # 情况5：其他情况，保持原样或返回列表
    return answer


def add_nlp_descriptions(input_file: str, output_file: str, 
                        api_key: str = None, 
                        base_url: str = None,
                        model: str = "qwen2.5-7b-instruct",
                        chunk_size: int = 50) -> None:
    """
    函数2：提取所有查询，分块传递给LLM获取自然语言描述
    
    从JSON文件中提取所有查询，分块调用LLM获取自然语言描述，
    然后在每个查询对象中新增"nlp"字段。
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        api_key: OpenAI API密钥（如果为None，将从环境变量OPENAI_API_KEY获取）
        base_url: API基础URL（可选，用于自定义API端点）
        model: 使用的模型名称，默认为"qwen2.5-7b-instruct"
        chunk_size: 每次传递给LLM的查询数量，默认为10
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化OpenAI客户端
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    if base_url:
        client_kwargs["base_url"] = base_url
    
    try:
        client = OpenAI(**client_kwargs)
    except Exception as e:
        raise ValueError(f"无法初始化OpenAI客户端: {e}。请确保提供了api_key或设置了OPENAI_API_KEY环境变量。")
    
    # 提取所有查询
    queries = [item.get("query", "") for item in data]
    
    # 分块处理
    nlp_descriptions = []
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i + chunk_size]
        descriptions = get_nlp_descriptions_batch(client, chunk, model)
        nlp_descriptions.extend(descriptions)
        print(f"已处理 {min(i + chunk_size, len(queries))}/{len(queries)} 个查询")
    
    # 为每个查询添加nlp字段
    for i, item in enumerate(data):
        item["nlp"] = nlp_descriptions[i] if i < len(nlp_descriptions) else ""
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_file}")


def get_nlp_descriptions_batch(client: OpenAI, queries: List[str], model: str) -> List[str]:
    """
    批量获取查询的自然语言描述
    
    Args:
        client: OpenAI客户端实例
        queries: 查询列表
        model: 模型名称
        
    Returns:
        自然语言描述列表
    """
    if not queries:
        return []
    
    # 构建提示词
    prompt = "Please provide concise natural language descriptions for the following Cypher queries. Use one line per query to describe the query's intent and basic process, without going into details.\n\n"
    prompt += "Important: When min() or max() is applied to string values, it refers to lexicographic (alphabetical) ordering, not numerical ordering. For example, min(city) means the city name that comes first alphabetically.\n\n"
    
    for i, query in enumerate(queries, 1):
        prompt += f"Query {i}:\n{query}\n\n"
    
    prompt += "Please provide one natural language description per query in order, separated by newlines, without adding numbers:"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional database query analysis expert, skilled at converting Cypher queries into concise natural language descriptions. When min() or max() is applied to strings, it means lexicographic (alphabetical) ordering. Please respond in English."},
                {"role": "user", "content": prompt}
            ]
        )
        
        descriptions_text = response.choices[0].message.content.strip()
        # 按行分割描述，去除可能的编号前缀（如"1. ", "查询1: "等）
        lines = descriptions_text.split('\n')
        descriptions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去除可能的编号前缀
            line = re.sub(r'^(\d+[\.\)、]?\s*|查询\d+[:：]?\s*)', '', line)
            if line:
                descriptions.append(line)
        
        # 确保返回的描述数量与查询数量一致
        while len(descriptions) < len(queries):
            descriptions.append("")
        descriptions = descriptions[:len(queries)]
        
        return descriptions
    
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        # 返回空描述列表
        return [""] * len(queries)


def check_and_fix_nlp_descriptions(input_file: str, output_file: str,
                                   api_key: str = None,
                                   base_url: str = None,
                                   model: str = "qwen2.5-7b-instruct",
                                   chunk_size: int = 50) -> None:
    """
    函数3：检查并修复查询对应的nlp描述
    
    检查JSON文件中每个查询的nlp字段是否有翻译错误、为空或不自然，
    使用LLM修复这些问题，生成更准确、自然、完整的描述。
    
    Args:
        input_file: 输入的JSON文件路径（应包含query和nlp字段）
        output_file: 输出的JSON文件路径
        api_key: OpenAI API密钥（如果为None，将从环境变量OPENAI_API_KEY获取）
        base_url: API基础URL（可选，用于自定义API端点）
        model: 使用的模型名称，默认为"qwen2.5-7b-instruct"
        chunk_size: 每次传递给LLM的查询数量，默认为50
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化OpenAI客户端
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    if base_url:
        client_kwargs["base_url"] = base_url
    
    try:
        client = OpenAI(**client_kwargs)
    except Exception as e:
        raise ValueError(f"无法初始化OpenAI客户端: {e}。请确保提供了api_key或设置了OPENAI_API_KEY环境变量。")
    
    # 提取所有查询和对应的nlp
    query_nlp_pairs = []
    for item in data:
        query = item.get("query", "")
        nlp = item.get("nlp", "")
        query_nlp_pairs.append((query, nlp))
    
    # 分块处理
    fixed_nlp_descriptions = []
    for i in range(0, len(query_nlp_pairs), chunk_size):
        chunk = query_nlp_pairs[i:i + chunk_size]
        fixed_descriptions = check_and_fix_nlp_batch(client, chunk, model)
        fixed_nlp_descriptions.extend(fixed_descriptions)
        print(f"已处理 {min(i + chunk_size, len(query_nlp_pairs))}/{len(query_nlp_pairs)} 个查询")
    
    # 更新每个查询的nlp字段
    fixed_count = 0
    for i, item in enumerate(data):
        old_nlp = item.get("nlp", "")
        new_nlp = fixed_nlp_descriptions[i] if i < len(fixed_nlp_descriptions) else old_nlp
        
        if old_nlp != new_nlp:
            item["nlp"] = new_nlp
            fixed_count += 1
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，修复了 {fixed_count} 个nlp字段，结果已保存到 {output_file}")


def check_and_fix_nlp_batch(client: OpenAI, query_nlp_pairs: List[tuple], model: str) -> List[str]:
    """
    批量检查和修复nlp描述
    
    Args:
        client: OpenAI客户端实例
        query_nlp_pairs: (query, nlp) 元组列表
        model: 模型名称
        
    Returns:
        修复后的nlp描述列表
    """
    if not query_nlp_pairs:
        return []
    
    # 构建提示词
    prompt = "Please check and fix the natural language descriptions (NLP) for the following Cypher queries. "
    prompt += "For each query, review its corresponding NLP description and:\n"
    prompt += "1. If the NLP is empty or missing, generate a complete and natural description.\n"
    prompt += "2. If the NLP has translation errors or is inaccurate, correct it.\n"
    prompt += "3. If the NLP is unnatural or lacks information, improve it to be more natural and complete.\n"
    prompt += "4. If the NLP is already good, keep it as is.\n\n"
    prompt += "Important guidelines:\n"
    prompt += "- When min() or max() is applied to string values, it refers to lexicographic (alphabetical) ordering.\n"
    prompt += "- Descriptions should be concise, clear, and natural in English.\n"
    prompt += "- Ensure all important information from the query is captured.\n\n"
    
    for i, (query, nlp) in enumerate(query_nlp_pairs, 1):
        prompt += f"Query {i}:\n{query}\n"
        prompt += f"Current NLP: {nlp if nlp else '(empty)'}\n\n"
    
    prompt += "Please provide one improved NLP description per query in order, separated by newlines, without adding numbers. "
    prompt += "If a description is already good, return it unchanged."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional database query analysis expert, skilled at reviewing and improving natural language descriptions of Cypher queries. You check for accuracy, completeness, and naturalness. Please respond in English."},
                {"role": "user", "content": prompt}
            ]
        )
        
        descriptions_text = response.choices[0].message.content.strip()
        # 按行分割描述，去除可能的编号前缀
        lines = descriptions_text.split('\n')
        descriptions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去除可能的编号前缀
            line = re.sub(r'^(\d+[\.\)、]?\s*|查询\d+[:：]?\s*|Query\s*\d+[:：]?\s*)', '', line, flags=re.IGNORECASE)
            if line:
                descriptions.append(line)
        
        # 确保返回的描述数量与查询数量一致
        while len(descriptions) < len(query_nlp_pairs):
            # 如果描述数量不足，保留原来的nlp
            idx = len(descriptions)
            descriptions.append(query_nlp_pairs[idx][1] if idx < len(query_nlp_pairs) else "")
        descriptions = descriptions[:len(query_nlp_pairs)]
        
        return descriptions
    
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        # 返回原来的nlp列表
        return [pair[1] for pair in query_nlp_pairs]


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


if __name__ == "__main__":
    # 示例用法
    # input_file = "query/ldbc_snb_finbench/noise_query_results_ldbcfin.json"
    # input_file = "query/mcp_tool/query_results_mcp1.json"
    # # 函数1：提取和清洗答案
    # output_file_1 = input_file.replace(".json", "_cleaned.json")
    # extract_and_clean_answers(input_file, output_file_1, node_id_key="id")
    # print(f"答案清洗完成，结果保存到: {output_file_1}")
    
    # 函数2：添加自然语言描述（需要设置API密钥）
    # output_file_2 = input_file.replace(".json", "_with_nlp.json")
    # model_name = "qwen2.5-14b-instruct"
    # # client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="sk-edc6c171ed524d059e3053b33ea96705")

    # # add_nlp_descriptions(input_file, output_file_2, model=model_name, api_key="sk-edc6c171ed524d059e3053b33ea96705", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", chunk_size=60)
    # # print(f"自然语言描述添加完成，结果保存到: {output_file_2}")
    
    # # 函数3：检查并修复nlp描述
    # input_file_with_nlp = "query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned_with_nlp.json"
    # output_file_fixed = input_file_with_nlp.replace(".json", "_fixed.json")
    # check_and_fix_nlp_descriptions(input_file_with_nlp, output_file_fixed, model=model_name, api_key="sk-edc6c171ed524d059e3053b33ea96705", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", chunk_size=10)
    # print(f"nlp描述检查修复完成，结果保存到: {output_file_fixed}")
    # 查找包含 "order by" 的查询（不区分大小写）
    # 移除answer为0或[]的项
    # remove_empty_answers(
    # "query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned_with_nlp_fixed.json",
    # "query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned_with_nlp_fixed_filtered.json"
    # )
    filter_queries_by_pattern(
        "query_results_ldbcfin.json",
        "query/ldbc_snb_finbench/query_results_ldbcfin_with_orderby.json",
        "order by"
    )