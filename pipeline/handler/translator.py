import json
import re
import os
from typing import List, Dict, Any, Union, Optional
from openai import OpenAI

# 可复用的 CRITICAL REQUIREMENTS 常量
_CRITICAL_REQUIREMENTS_TEMPLATE = """***CRITICAL REQUIREMENTS:***
1. **ONLY return the {content_type} of each {item_type}**
2. **DO NOT include any explanations, comments, or additional text**
3. **DO NOT include {item_type} numbers, prefixes, or any other metadata**
4. **DO NOT include phrases like 'This {item_type}...', 'The {item_type}...', or any introductory text**
5. **Each line should contain ONLY the pure {content_type} of the corresponding {item_type}**
6. **Return one {content_type} per {item_type} in order, separated by newlines**
7. **NO numbers, NO prefixes, NO extra words - ONLY the {content_name} itself**
"""

# System message 中的强调部分模板
_SYSTEM_CRITICAL_TEMPLATE = "**CRITICAL: You MUST ONLY return the pure {content_type} of {item_type_plural}. DO NOT include any explanations, comments, numbers, prefixes, or any other text. Each response line should contain ONLY the {content_name} itself.**"


def _get_critical_requirements(content_type: str, item_type: str = "query", content_name: str = None) -> str:
    """
    生成 CRITICAL REQUIREMENTS 文本
    
    Args:
        content_type: 返回内容的类型，如 "natural language description/translation", "question stem/translation"
        item_type: 项目类型，如 "query", "query pair"
        content_name: 内容名称，用于第7条，如 "translation", "question stem", "description"
    
    Returns:
        CRITICAL REQUIREMENTS 文本
    """
    if content_name is None:
        content_name = content_type.split('/')[0] if '/' in content_type else content_type
    return _CRITICAL_REQUIREMENTS_TEMPLATE.format(
        content_type=content_type,
        item_type=item_type,
        content_name=content_name
    )


def _get_system_critical_emphasis(content_type: str, item_type_plural: str = "queries", content_name: str = None) -> str:
    """
    生成 system message 中的强调部分
    
    Args:
        content_type: 返回内容的类型
        item_type_plural: 项目类型的复数形式
        content_name: 内容名称
    
    Returns:
        System message 强调文本
    """
    if content_name is None:
        content_name = content_type.split('/')[0] if '/' in content_type else content_type
    return _SYSTEM_CRITICAL_TEMPLATE.format(
        content_type=content_type,
        item_type_plural=item_type_plural,
        content_name=content_name
    )

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


def get_nlp_descriptions_batch(client: OpenAI, queries: Union[List[str], List[Dict[str, str]]], model: str, mode: str = "normal") -> Union[List[str], List[Dict[str, str]]]:
    """
    批量获取查询的自然语言描述
    
    Args:
        client: OpenAI客户端实例
        queries: 查询列表。当 mode="normal" 时，为字符串列表；当 mode="judge" 时，为包含 "template_query" 和 "anti_template_query" 的字典列表；当 mode="manage" 时，为包含 "operate_query" 和 "valid_query" 的字典列表
        model: 模型名称
        mode: 模式，可选 "normal"、"judge" 或 "manage"，默认为 "normal"
            - "normal": 普通查询描述模式
            - "judge": 判断题模式，将查询对转化为判断题题干
            - "manage": 管理查询模式，将操作查询和验证查询分别转化为自然语言描述
        
    Returns:
        自然语言描述列表。当 mode="manage" 时，返回包含 "operate_nlp" 和 "valid_nlp" 的字典列表；否则返回字符串列表
    """
    if not queries:
        return []
    
    if mode == "judge":
        return _get_judge_descriptions_batch(client, queries, model)
    elif mode == "manage":
        return _get_manage_descriptions_batch(client, queries, model)
    else:
        return _get_normal_descriptions_batch(client, queries, model)


def _get_normal_descriptions_batch(client: OpenAI, queries: List[str], model: str) -> List[str]:
    """普通查询描述模式"""
    # 构建提示词
    prompt = "Task: Write ONE concise English natural-language description for each Cypher query.\n"
    prompt += "Key rules:\n"
    prompt += "- Preserve ALL numeric/date/time literals and quantitative constraints from the Cypher in the description (e.g., numbers, decimals, negative values, ranges, LIMIT/SKIP, slices, counts, offsets, thresholds, timestamps). Do NOT omit or generalize them.\n"
    prompt += "- When min() or max() is applied to string values, interpret it as lexicographic (alphabetical) ordering.\n\n"
    prompt += "Example:\n"
    prompt += "\"query\": \"MATCH (n:Account) WHERE n.createTime = '2022-10-15 14:34:56.729' OR (n)-[:Account_Repay_Loan]->(:Company) RETURN n.email\"\n"
    prompt += "\"nlp\": Find the email of accounts created on a 2022-10-15 14:34:56.729 or accounts repaying a loan\n\n"
    
    for i, query in enumerate(queries, 1):
        prompt += f"Query {i}:\n{query}\n\n"
    
    prompt += _get_critical_requirements(
        content_type="natural language description/translation",
        item_type="query",
        content_name="translation"
    )
    prompt += "\nReturn the translations in order, one per line:"
    
    try:
        system_content = (
            "You are a professional database query analysis expert. "
            "Please respond in English."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
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


def _get_judge_descriptions_batch(client: OpenAI, query_pairs: List[Dict[str, str]], model: str) -> List[str]:
    """判断题模式：将查询转化为判断题题干"""
    # 构建提示词
    prompt = "Please convert each Cypher query into a judgment question (true/false question) stem. "
    prompt += "You will be given a template_query that represents a positive condition (what should be true). "
    prompt += "Your task is to create a question stem that asks whether the query result matches the intended condition.\n\n"
    
    prompt += "The template_query typically checks for a specific relationship or condition. "
    prompt += "You need to formulate a question that asks: 'Does the query result correctly represent this condition?'\n\n"
    
    prompt += "Example1:\n"
    prompt += "Template Query: MATCH (a:Medium), (b:Medium) WHERE a.mediumType = b.mediumType AND a <> b WITH a, b RETURN a, collect(b.mediumType) AS bs\n"
    prompt += "Question Stem: For Medium node a, is the collect b.mediumType belongs to Medium nodes b that are not the same node and have the same mediumType as a?\n\n"
    prompt += "Example2:\n"
    prompt += "Template Query: MATCH (a:Person) WHERE (a)-[:Person_Guarantee_Person]->(:Person) WITH a RETURN collect(a.personName)[0..5] AS values\n"
    prompt += "Question Stem: Are the five Person nodes acting as guarantors for other Person nodes?\n\n"

    prompt += "Guidelines:\n"
    prompt += "- The question should be clear and concise, asking about whether the query result correctly represents the condition checked by the template_query.\n"
    prompt += "- Analyze what the template_query is checking (relationships, properties, conditions, etc.) and formulate a question about that.\n"
    prompt += "- Use natural language that clearly describes the semantic meaning of what the query is verifying.\n"
    prompt += "- Format as a judgment question that can be answered with true/false or yes/no.\n"
    prompt += "- Focus on the key condition or relationship that the query is designed to check.\n\n"
    
    for i, query_pair in enumerate(query_pairs, 1):
        template_query = query_pair.get("template_query", "")
        prompt += f"Query {i}:\n{template_query}\n\n"
    
    prompt += _get_critical_requirements(
        content_type="question stem/translation",
        item_type="query",
        content_name="question stem"
    )
    prompt += "\nPlease provide one question stem per query in order, separated by newlines, without adding numbers:"
    
    try:
        system_content = (
            "You are a professional database query analysis expert, skilled at converting Cypher queries into clear judgment questions. "
            "You understand the semantic meaning of queries and can formulate precise question stems that ask whether the query result correctly represents the intended condition. "
            f"{_get_system_critical_emphasis('question stem/translation', 'queries', 'question stem')} "
            "Please respond in English."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
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
            line = re.sub(r'^(\d+[\.\)、]?\s*|查询\d+[:：]?\s*|Pair\s*\d+[:：]?\s*)', '', line, flags=re.IGNORECASE)
            if line:
                descriptions.append(line)
        
        # 确保返回的描述数量与查询对数量一致
        while len(descriptions) < len(query_pairs):
            descriptions.append("")
        descriptions = descriptions[:len(query_pairs)]
        
        return descriptions
    
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        # 返回空描述列表
        return [""] * len(query_pairs)


def _get_manage_descriptions_batch(client: OpenAI, query_pairs: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
    """管理查询模式：将操作查询和验证查询分别转化为自然语言描述"""
    # 构建提示词
    prompt = "You are given pairs of Cypher queries:\n\n"
    prompt += "1. An operate query that creates or modifies graph data (e.g., CREATE, DELETE, SET, MERGE operations).\n"
    prompt += "2. A validation query that reads from the graph and computes a result.\n\n"
    prompt += "Your task is to generate TWO separate natural language descriptions for each pair:\n"
    prompt += "- operate_nlp: A concise description of what the operate query does (e.g., 'Insert a Loan node with interestRate 0.008').\n"
    prompt += "- valid_nlp: A concise question describing what the validation query asks (e.g., 'What is the average interestRate of all Loan nodes?').\n\n"
    prompt += "Guidelines:\n"
    prompt += "- Keep descriptions concise and clear.\n"
    prompt += "- Do not mention Cypher, queries, or database operations explicitly.\n"
    prompt += "- Use natural language that clearly describes the operation and the question.\n"
    prompt += "- For operate_nlp, focus on what data is being created/modified/deleted and key properties.\n"
    prompt += "- For valid_nlp, focus on what result is being asked for.\n\n"
    
    for i, query_pair in enumerate(query_pairs, 1):
        operate_query = query_pair.get("operate_query", "")
        valid_query = query_pair.get("valid_query", "")
        prompt += f"Pair {i}:\n"
        prompt += f"Operate query: {operate_query}\n"
        prompt += f"Validation query: {valid_query}\n\n"
    
    prompt += "***CRITICAL REQUIREMENTS:***\n"
    prompt += "1. **Return TWO lines per pair: first line is operate_nlp, second line is valid_nlp**\n"
    prompt += "2. **DO NOT include any explanations, comments, or additional text**\n"
    prompt += "3. **DO NOT include pair numbers, prefixes, or any other metadata**\n"
    prompt += "4. **Each line should contain ONLY the pure natural language description**\n"
    prompt += "5. **Format: one pair per two lines, separated by newlines**\n"
    prompt += "6. **NO numbers, NO prefixes, NO extra words - ONLY the descriptions themselves**\n\n"
    prompt += "Please provide the descriptions in order, with two lines per pair (operate_nlp, then valid_nlp), separated by newlines:"
    
    try:
        system_content = (
            "You are a professional database query analysis expert, skilled at converting Cypher queries into natural language descriptions. "
            "You understand data operations (create, modify, delete) and can formulate clear descriptions for both operations and validation queries. "
            "**CRITICAL: You MUST return TWO lines per pair - first line is operate_nlp, second line is valid_nlp. DO NOT include any explanations, comments, numbers, prefixes, or any other text.** "
            "Please respond in English."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        )
        
        descriptions_text = response.choices[0].message.content.strip()
        # 按行分割描述，去除可能的编号前缀
        lines = descriptions_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去除可能的编号前缀
            line = re.sub(r'^(\d+[\.\)、]?\s*|查询\d+[:：]?\s*|Pair\s*\d+[:：]?\s*|operate_nlp[:：]?\s*|valid_nlp[:：]?\s*)', '', line, flags=re.IGNORECASE)
            if line:
                cleaned_lines.append(line)
        
        # 每两行组成一个 pair (operate_nlp, valid_nlp)
        descriptions = []
        for i in range(0, len(cleaned_lines), 2):
            operate_nlp = cleaned_lines[i] if i < len(cleaned_lines) else ""
            valid_nlp = cleaned_lines[i + 1] if i + 1 < len(cleaned_lines) else ""
            descriptions.append({
                "operate_nlp": operate_nlp,
                "valid_nlp": valid_nlp
            })
        
        # 确保返回的描述数量与查询对数量一致
        while len(descriptions) < len(query_pairs):
            descriptions.append({"operate_nlp": "", "valid_nlp": ""})
        descriptions = descriptions[:len(query_pairs)]
        
        return descriptions
    
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        # 返回空描述列表
        return [{"operate_nlp": "", "valid_nlp": ""} for _ in query_pairs]


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
    
    critical_req = _get_critical_requirements(
        content_type="improved/fixed natural language description/translation",
        item_type="query",
        content_name="description"
    )
    # 为 check_and_fix 模式添加第8条特殊要求
    critical_req += "8. **If a description is already good, return it unchanged (but still ONLY the description, nothing else)**\n\n"
    prompt += critical_req
    prompt += "Please provide one improved NLP description per query in order, separated by newlines, without adding numbers. "
    prompt += "If a description is already good, return it unchanged."
    
    try:
        system_content = (
            "You are a professional database query analysis expert, skilled at reviewing and improving natural language descriptions of Cypher queries. "
            "You check for accuracy, completeness, and naturalness. "
            f"{_get_system_critical_emphasis('improved description/translation', 'queries', 'description')} "
            "Please respond in English."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
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

