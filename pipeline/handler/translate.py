import json
import os
import sys
from typing import Optional
from openai import OpenAI

# 导入 translator.py 中的函数
# 支持相对导入和绝对导入两种方式
try:
    from .translator import get_nlp_descriptions_batch
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行脚本时）
    from translator import get_nlp_descriptions_batch


def translate_normal(input_file: str, output_file: str,
                     api_key: str = None,
                     base_url: str = None,
                     model: str = "qwen2.5-7b-instruct",
                     chunk_size: int = 10) -> None:
    """
    翻译 normal 类型文件：为包含 query 字段的 JSON 文件添加 nlp 描述
    
    Args:
        input_file: 输入的JSON文件路径（应包含 query 字段）
        output_file: 输出的JSON文件路径
        api_key: OpenAI API密钥（如果为None，将从环境变量OPENAI_API_KEY获取）
        base_url: API基础URL（可选，用于自定义API端点）
        model: 使用的模型名称，默认为"qwen2.5-7b-instruct"
        chunk_size: 每次传递给LLM的查询数量，默认为50
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("文件为空，无需处理")
        return
    
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
    
    # 流式输出模式：边处理边写入
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
        first_item = True
        
        # 分块处理
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            descriptions = get_nlp_descriptions_batch(client, chunk, model, mode="normal")
            
            # 为当前 chunk 的条目添加 nlp 并写入
            for j, item in enumerate(data[i:i + chunk_size]):
                item["nlp"] = descriptions[j] if j < len(descriptions) else ""
                
                if not first_item:
                    f.write(',\n')
                json.dump(item, f, ensure_ascii=False, indent=2)
                f.flush()
                first_item = False
            
            print(f"已处理 {min(i + chunk_size, len(queries))}/{len(queries)} 个查询")
        
        f.write('\n]')
    
    print(f"处理完成，结果已保存到 {output_file}")


def translate_judge(input_file: str, output_file: str,
                    api_key: str = None,
                    base_url: str = None,
                    model: str = "qwen2.5-7b-instruct",
                    chunk_size: int = 10) -> None:
    """
    翻译 judge 类型文件：为包含 template_query 字段的 JSON 文件添加 nlp 描述
    
    Args:
        input_file: 输入的JSON文件路径（应包含 template_query 字段）
        output_file: 输出的JSON文件路径
        api_key: OpenAI API密钥（如果为None，将从环境变量OPENAI_API_KEY获取）
        base_url: API基础URL（可选，用于自定义API端点）
        model: 使用的模型名称，默认为"qwen2.5-7b-instruct"
        chunk_size: 每次传递给LLM的查询数量，默认为50
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("文件为空，无需处理")
        return
    
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
    
    # 提取所有 template_query，转换为字典列表格式
    query_pairs = [{"template_query": item.get("template_query", "")} for item in data]
    
    # 流式输出模式：边处理边写入
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
        first_item = True
        
        # 分块处理
        for i in range(0, len(query_pairs), chunk_size):
            chunk = query_pairs[i:i + chunk_size]
            descriptions = get_nlp_descriptions_batch(client, chunk, model, mode="judge")
            
            # 为当前 chunk 的条目添加 nlp 并写入
            for j, item in enumerate(data[i:i + chunk_size]):
                item["nlp"] = descriptions[j] if j < len(descriptions) else ""
                
                if not first_item:
                    f.write(',\n')
                json.dump(item, f, ensure_ascii=False, indent=2)
                f.flush()
                first_item = False
            
            print(f"已处理 {min(i + chunk_size, len(query_pairs))}/{len(query_pairs)} 个查询对")
        
        f.write('\n]')
    
    print(f"处理完成，结果已保存到 {output_file}")


def translate_manage(input_file: str, output_file: str,
                     api_key: str = None,
                     base_url: str = None,
                     model: str = "qwen2.5-7b-instruct",
                     chunk_size: int = 10,
                     stream_output: bool = True) -> None:
    """
    翻译 manage 类型文件：为包含 template 和 post_validation 字段的 JSON 文件添加 nlp 描述
    
    Args:
        input_file: 输入的JSON文件路径（应包含 template.query 和 post_validation.query 字段）
        output_file: 输出的JSON文件路径
        api_key: OpenAI API密钥（如果为None，将从环境变量OPENAI_API_KEY获取）
        base_url: API基础URL（可选，用于自定义API端点）
        model: 使用的模型名称，默认为"qwen2.5-7b-instruct"
        chunk_size: 每次传递给LLM的查询数量，默认为10
        stream_output: 是否流式输出（边处理边写入文件），默认为True
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("文件为空，无需处理")
        return
    
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
    
    # 提取 template.query 和 post_validation.query
    query_pairs = []
    for item in data:
        template_query = ""
        post_validation_query = ""
        
        # 处理 template 字段（可能是对象或字典）
        if "template" in item:
            template = item["template"]
            if isinstance(template, dict):
                template_query = template.get("query", "")
            elif isinstance(template, str):
                template_query = template
        
        # 处理 post_validation 字段（可能是对象或字典）
        if "post_validation" in item:
            post_validation = item["post_validation"]
            if isinstance(post_validation, dict):
                post_validation_query = post_validation.get("query", "")
            elif isinstance(post_validation, str):
                post_validation_query = post_validation
        
        query_pairs.append({
            "template_query": template_query,
            "post_validation_query": post_validation_query
        })
    
    if stream_output:
        # 流式输出模式：边处理边写入
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            first_item = True
            
            # 分块处理
            for i in range(0, len(query_pairs), chunk_size):
                chunk = query_pairs[i:i + chunk_size]
                descriptions = get_nlp_descriptions_batch(client, chunk, model, mode="manage")
                
                # 为当前 chunk 的条目添加 nlp 并写入
                for j, item in enumerate(data[i:i + chunk_size]):
                    item["nlp"] = descriptions[j] if j < len(descriptions) else ""
                    
                    if not first_item:
                        f.write(',\n')
                    json.dump(item, f, ensure_ascii=False, indent=2)
                    f.flush()
                    first_item = False
                
                print(f"已处理 {min(i + chunk_size, len(query_pairs))}/{len(query_pairs)} 个查询对")
            
            f.write('\n]')
    else:
        # 传统模式：全部处理完再写入
        nlp_descriptions = []
        for i in range(0, len(query_pairs), chunk_size):
            chunk = query_pairs[i:i + chunk_size]
            descriptions = get_nlp_descriptions_batch(client, chunk, model, mode="manage")
            nlp_descriptions.extend(descriptions)
            print(f"已处理 {min(i + chunk_size, len(query_pairs))}/{len(query_pairs)} 个查询对")
        
        # 为每个条目添加nlp字段
        for i, item in enumerate(data):
            item["nlp"] = nlp_descriptions[i] if i < len(nlp_descriptions) else ""
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_file}")


def translate_all(normal_file: str = None, judge_file: str = None, manage_file: str = None,
                  normal_output: str = None, judge_output: str = None, manage_output: str = None,
                  chunk_size: int = 10,
                  api_key: str = None, base_url: str = None, model: str = "qwen2.5-7b-instruct") -> None:
    """
    批量翻译所有类型的文件
    
    Args:
        normal_file: normal 类型输入文件路径
        judge_file: judge 类型输入文件路径
        manage_file: manage 类型输入文件路径
        normal_output: normal 类型输出文件路径
        judge_output: judge 类型输出文件路径
        manage_output: manage 类型输出文件路径
        chunk_size: 所有模式的批次大小，默认为 10
        api_key: OpenAI API密钥
        base_url: API基础URL
        model: 模型名称
    """
    if normal_file:
        print(f"\n{'='*60}")
        print(f"开始翻译 Normal 类型文件: {normal_file}")
        print(f"{'='*60}")
        translate_normal(normal_file, normal_output or normal_file.replace('.json', '_translated.json'),
                        api_key=api_key, base_url=base_url, model=model, chunk_size=chunk_size)
    
    if judge_file:
        print(f"\n{'='*60}")
        print(f"开始翻译 Judge 类型文件: {judge_file}")
        print(f"{'='*60}")
        translate_judge(judge_file, judge_output or judge_file.replace('.json', '_translated.json'),
                       api_key=api_key, base_url=base_url, model=model, chunk_size=chunk_size)
    
    if manage_file:
        print(f"\n{'='*60}")
        print(f"开始翻译 Manage 类型文件: {manage_file}")
        print(f"{'='*60}")
        translate_manage(manage_file, manage_output or manage_file.replace('.json', '_translated.json'),
                        api_key=api_key, base_url=base_url, model=model, chunk_size=chunk_size)
    
    print(f"\n{'='*60}")
    print("所有翻译任务完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    """
    使用示例：
    
    # 方式1：单独翻译
    python translate.py normal input.json output.json 50
    
    # 方式2：批量翻译（通过代码调用）
    translate_all(
        normal_file="query/ldbc_snb_finbench/noise_query_execution_step1_ldbcfin.json",
        judge_file="query/ldbc_snb_finbench/noise_judge_query_results_step2_ldbcfin.json",
        manage_file="query/ldbc_snb_finbench/management_query_ldbc_fin.json"
    )
    """
    import sys
    
    # 解析命令行参数，支持 --api-key 和 --base-url
    api_key = "sk-edc6c171ed524d059e3053b33ea96705"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen2.5-7b-instruct"
    
    # 检查是否有 --api-key 参数
    if "--api-key" in sys.argv:
        idx = sys.argv.index("--api-key")
        if idx + 1 < len(sys.argv):
            api_key = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)
    
    # 检查是否有 --base-url 参数
    if "--base-url" in sys.argv:
        idx = sys.argv.index("--base-url")
        if idx + 1 < len(sys.argv):
            base_url = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)
    
    # 检查是否有 --model 参数
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)
    
    # 如果参数数量少于3个，尝试批量处理模式
    if len(sys.argv) < 3:
        # 批量处理模式：使用默认文件路径
        print("使用批量处理模式，翻译所有三个文件...")
        translate_all(
            normal_file="../query_gen/query/ldbc_snb_finbench/noise_query_execution_step1_ldbcfin.json",
            judge_file="../query_gen/query/ldbc_snb_finbench/noise_judge_query_results_step2_ldbcfin.json",
            manage_file="../query_gen/query/ldbc_snb_finbench/management_query_ldbc_fin.json",
            chunk_size=10,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
    elif len(sys.argv) < 4:
        print("用法: python translate.py [选项] [<mode> <input_file> <output_file> [chunk_size]]")
        print("  或: python translate.py [选项]  (批量处理默认文件)")
        print("\n选项:")
        print("  --api-key <key>      OpenAI API密钥")
        print("  --base-url <url>      API基础URL（可选）")
        print("  --model <model>      模型名称（默认: qwen2.5-7b-instruct）")
        print("\n参数:")
        print("  mode: normal/judge/manage")
        print("  chunk_size: 每次处理的查询数量 (默认: 10)")
        print("\n示例:")
        print("  python translate.py --api-key sk-xxx")
        print("  python translate.py --api-key sk-xxx normal input.json output.json 10")
        print("  python translate.py --api-key sk-xxx --base-url http://localhost:8000/v1 judge input.json output.json")
        sys.exit(1)
    else:
        # 单独处理模式
        mode = sys.argv[1]
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        print(f"开始翻译: {input_file} -> {output_file}")
        print(f"模式: {mode}, 批次大小: {chunk_size}")
        
        if mode == "normal":
            translate_normal(input_file, output_file, api_key=api_key, base_url=base_url, model=model, chunk_size=chunk_size)
        elif mode == "judge":
            translate_judge(input_file, output_file, api_key=api_key, base_url=base_url, model=model, chunk_size=chunk_size)
        elif mode == "manage":
            translate_manage(input_file, output_file, api_key=api_key, base_url=base_url, model=model, chunk_size=chunk_size)
        else:
            print(f"错误: 未知的模式 '{mode}'，请使用 'normal'、'judge' 或 'manage'")
            sys.exit(1)
        
        print("翻译完成！")
