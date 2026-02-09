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
    翻译 manage 类型文件：为包含 template.query 和 validation 字段的 JSON 文件添加 nlp 描述
    
    处理逻辑：
    - 从 template.query 数组的索引1开始提取查询
    - 从 validation 数组中 index=1 开始的元素提取验证查询
    - 将两者配对：query[1] 与 validation[index=1], query[2] 与 validation[index=2], 等等
    - 为每个配对生成 operate_nlp 和 valid_nlp 描述
    
    输出格式：
    - step: 步骤编号（从1开始）
    - operate_query: 操作查询语句
    - operate_nlp: 操作查询的自然语言描述
    - valid_query: 验证查询语句
    - valid_nlp: 验证查询的自然语言描述
    - answer: 验证查询的答案
    
    Args:
        input_file: 输入的JSON文件路径（应包含 template.query 数组和 validation 数组）
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
    
    # 提取 template.query 和 validation 查询对
    # 从 template.query 数组索引1开始，与 validation 数组中 index=1 开始的元素配对
    # 每个batch的所有步骤放在一个对象中
    batch_query_pairs = []  # 存储每个batch的查询对列表
    batch_output_data = []  # 存储每个batch的输出数据
    
    for item in data:
        # 获取 template.query 数组（从索引1开始）
        template_queries = []
        if "template" in item:
            template = item["template"]
            if isinstance(template, dict):
                query_list = template.get("query", [])
                if isinstance(query_list, list):
                    # 从索引1开始提取（跳过索引0）
                    template_queries = query_list[1:] if len(query_list) > 1 else []
                elif isinstance(query_list, str):
                    # 如果只有一个查询，跳过（因为需要从索引1开始）
                    template_queries = []
        
        # 获取 validation 数组（从 index=1 开始）
        validation_items = []
        if "validation" in item:
            validation_list = item["validation"]
            if isinstance(validation_list, list):
                # 过滤出 index >= 1 的项
                validation_items = [v for v in validation_list if isinstance(v, dict) and v.get("index", -1) >= 1]
                # 按 index 排序
                validation_items.sort(key=lambda x: x.get("index", 0))
        
        # 创建一个 index 到 validation_item 的映射
        validation_map = {v.get("index", -1): v for v in validation_items}
        
        # 当前batch的查询对和步骤数据
        batch_pairs = []
        batch_steps = []
        
        # 配对逻辑：从query索引1开始，配对validation的index=1到index=5
        # query数组有5个元素（索引0-4），从索引1开始取得到4个（索引1-4，即template_queries[0-3]）
        # validation有index=1到index=5，共5个
        # 配对规则：
        #   - validation[index=1] 对应 query[1] (即template_queries[0])
        #   - validation[index=2] 对应 query[2] (即template_queries[1])
        #   - validation[index=3] 对应 query[3] (即template_queries[2])
        #   - validation[index=4] 对应 query[4] (即template_queries[3])
        #   - validation[index=5] 对应 query[4] (最后一个操作执行完后的状态，即template_queries[3])
        
        # 遍历所有可用的validation项（index=1到index=5）
        # 确保按index排序处理
        for validation_item in validation_items:
            validation_index = validation_item.get("index", -1)
            if validation_index < 1:
                continue
            
            # validation[index] 对应 template_queries[index-1]
            # 例如：validation[index=1] -> template_queries[0]
            #      validation[index=2] -> template_queries[1]
            #      ...
            #      validation[index=4] -> template_queries[3]
            #      validation[index=5] -> template_queries[3] (最后一个)
            
            query_idx_in_sliced = validation_index - 1  # validation[index] 对应 template_queries[index-1]
            
            # 如果索引超出范围，使用最后一个query（对应最后一个操作执行完后的状态）
            if query_idx_in_sliced >= len(template_queries):
                if len(template_queries) > 0:
                    operate_query = template_queries[-1]
                else:
                    # 如果没有template_queries，跳过
                    continue
            else:
                operate_query = template_queries[query_idx_in_sliced]
            
            valid_query = validation_item.get("query", "")
            answer = validation_item.get("answer", None)
            
            batch_pairs.append({
                "operate_query": operate_query,
                "valid_query": valid_query
            })
            
            # 保存配对信息用于后续输出
            batch_steps.append({
                "step": validation_index,  # step 对应validation的index
                "operate_query": operate_query,
                "valid_query": valid_query,
                "answer": answer
                # operate_nlp 和 valid_nlp 将在后续处理中添加
            })
        
        # 调试信息：打印当前batch的步骤数量
        if batch_steps:
            print(f"Batch {len(batch_output_data) + 1}: 找到 {len(batch_steps)} 个步骤 (step {batch_steps[0]['step']} 到 {batch_steps[-1]['step']})")
        
        # 将当前batch的数据添加到列表中
        if batch_pairs:
            batch_query_pairs.append(batch_pairs)
            batch_output_data.append(batch_steps)
    
    # 流式输出模式：边翻译边写入文件
    if stream_output:
        print(f"\n开始流式翻译和输出：共有 {len(batch_output_data)} 个batch")
        
        # 打开输出文件，准备写入JSON数组
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            first_batch = True
            
            # 遍历每个batch，逐个处理并写入
            desc_index = 0  # 全局描述索引
            
            for batch_idx, batch_steps in enumerate(batch_output_data):
                batch_pairs = batch_query_pairs[batch_idx]
                
                # 翻译当前batch的查询对
                print(f"\n处理 Batch {batch_idx + 1}/{len(batch_output_data)}: {len(batch_pairs)} 个查询对")
                batch_nlp_descriptions = []
                
                # 分块翻译当前batch的查询对
                for i in range(0, len(batch_pairs), chunk_size):
                    chunk = batch_pairs[i:i + chunk_size]
                    descriptions = get_nlp_descriptions_batch(client, chunk, model, mode="manage")
                    batch_nlp_descriptions.extend(descriptions)
                    print(f"  已翻译 {min(i + chunk_size, len(batch_pairs))}/{len(batch_pairs)} 个查询对")
                
                # 构建当前batch的输出对象
                batch_obj = {
                    "steps": []
                }
                
                # 将翻译结果分配到步骤
                for step_idx, step_data in enumerate(batch_steps):
                    if step_idx < len(batch_nlp_descriptions):
                        desc = batch_nlp_descriptions[step_idx]
                        if isinstance(desc, dict):
                            step_data["operate_nlp"] = desc.get("operate_nlp", "")
                            step_data["valid_nlp"] = desc.get("valid_nlp", "")
                        else:
                            step_data["operate_nlp"] = ""
                            step_data["valid_nlp"] = desc if isinstance(desc, str) else ""
                    else:
                        step_data["operate_nlp"] = ""
                        step_data["valid_nlp"] = ""
                        print(f"  警告：步骤 {step_data.get('step', '?')} 没有对应的翻译结果")
                    
                    batch_obj["steps"].append(step_data)
                
                # 写入当前batch到文件
                if not first_batch:
                    f.write(',\n')
                json.dump(batch_obj, f, ensure_ascii=False, indent=2)
                f.flush()  # 立即刷新到磁盘
                first_batch = False
                
                print(f"  Batch {batch_idx + 1} 完成：包含 {len(batch_obj['steps'])} 个步骤，已写入文件")
            
            f.write('\n]')
        
        print(f"\n处理完成，结果已保存到 {output_file}")
        print(f"输出格式：每个batch一个对象，包含steps数组")
    
    else:
        # 传统模式：全部处理完再写入
        # 展平所有batch的查询对，用于批量翻译
        query_pairs = []
        for batch_pairs in batch_query_pairs:
            query_pairs.extend(batch_pairs)
        
        # 批量翻译所有查询对
        nlp_descriptions = []
        for i in range(0, len(query_pairs), chunk_size):
            chunk = query_pairs[i:i + chunk_size]
            descriptions = get_nlp_descriptions_batch(client, chunk, model, mode="manage")
            nlp_descriptions.extend(descriptions)
            print(f"已处理 {min(i + chunk_size, len(query_pairs))}/{len(query_pairs)} 个查询对")
        
        # 将翻译结果分配回对应的batch和步骤
        desc_index = 0
        final_output_data = []
        
        print(f"\n开始分配翻译结果：共有 {len(batch_output_data)} 个batch，{len(nlp_descriptions)} 个翻译结果")
        
        for batch_idx, batch_steps in enumerate(batch_output_data):
            batch_obj = {
                "steps": []
            }
            
            print(f"处理 Batch {batch_idx + 1}: 有 {len(batch_steps)} 个步骤")
            
            for step_idx, step_data in enumerate(batch_steps):
                if desc_index < len(nlp_descriptions):
                    desc = nlp_descriptions[desc_index]
                    if isinstance(desc, dict):
                        step_data["operate_nlp"] = desc.get("operate_nlp", "")
                        step_data["valid_nlp"] = desc.get("valid_nlp", "")
                    else:
                        step_data["operate_nlp"] = ""
                        step_data["valid_nlp"] = desc if isinstance(desc, str) else ""
                else:
                    step_data["operate_nlp"] = ""
                    step_data["valid_nlp"] = ""
                    print(f"警告：Batch {batch_idx + 1} 的步骤 {step_data.get('step', '?')} 没有对应的翻译结果")
                
                batch_obj["steps"].append(step_data)
                desc_index += 1
            
            final_output_data.append(batch_obj)
            print(f"Batch {batch_idx + 1} 完成：包含 {len(batch_obj['steps'])} 个步骤")
        
        # 验证输出格式
        print(f"\n验证输出格式：")
        print(f"  - 共有 {len(final_output_data)} 个batch")
        for i, batch_obj in enumerate(final_output_data):
            if isinstance(batch_obj, dict) and "steps" in batch_obj:
                print(f"  - Batch {i+1}: 包含 {len(batch_obj['steps'])} 个步骤")
            else:
                print(f"  - 警告：Batch {i+1} 格式不正确！")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成，结果已保存到 {output_file}")
        print(f"输出格式：每个batch一个对象，包含steps数组")


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
    api_key = ""
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
        chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        
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
