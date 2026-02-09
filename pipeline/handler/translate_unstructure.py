"""
将 Cypher 查询翻译为流畅的自然英语描述（不包含数据库/图模式信息）。
一条一条翻译，覆盖输入 JSON 中每条记录的 nlp 字段。
"""
import json
import os
from typing import Optional
from openai import OpenAI

# 用户指定的翻译 prompt：Cypher → 自然英语语义描述
TRANSLATION_SYSTEM = (
    "You are a professional expert at explaining database query intent in plain English. "
    "You never mention database-specific terms (labels, relationship types, property names, query syntax). "
    "You paraphrase identifiers into natural concepts (events, steps, functions)."
)

TRANSLATION_TASK = """Task:
Translate a given Cypher query into a fluent and natural English description that clearly conveys the query's intent.

Requirements:

Describe the semantic intent of the query in plain English.

Do not mention any database-specific information, including:

node labels (e.g., entity)

relationship types

property names or values (e.g., id)

query syntax (e.g., MATCH, WHERE, RETURN)

Identifiers in the query represent events, steps, or functions and should be paraphrased into natural concepts.

The output should sound natural and be understandable without any knowledge of databases or graph schemas.

Do not copy literal strings or identifiers from the query into the output.

Example

Input Cypher Query:

MATCH (n:entity)
WHERE (n)-[:before]->(:entity {id: 'check-available-resources'})
RETURN n.id


Expected Output:

Retrieve all steps that occur before checking whether resources are available.
"""


def translate_one_query(
    client: OpenAI,
    query: str,
    model: str,
) -> str:
    """
    对单条 Cypher 查询调用 LLM，返回自然英语描述。

    Args:
        client: OpenAI 客户端
        query: Cypher 查询字符串
        model: 模型名

    Returns:
        自然语言描述，失败时返回空字符串
    """
    if not (query or "").strip():
        return ""

    user_content = TRANSLATION_TASK.strip() + "\n\nInput Cypher Query:\n\n" + query.strip() + "\n\nExpected Output (one line, no prefix):"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        # 去掉可能的前缀（如 "Output:", "Expected Output:", 编号等）
        for prefix in ("Expected Output:", "Output:", "Expected output:", "output:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
                break
        return text
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return ""


def translate_unstructured(
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "qwen2.5-7b-instruct",
) -> None:
    """
    翻译 unstructured 类型文件：为包含 query 的每条记录添加/覆盖 nlp 字段。

    - 读取 JSON 数组，每个元素需有 "query" 字段
    - 一条一条调用 LLM，将 Cypher 译为自然英语（不包含数据库/图模式信息）
    - 将结果写入该元素的 "nlp" 字段
    - 采用流式写入，每处理一条就写入一条，避免中断丢失进度
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("文件为空，无需处理")
        return

    if not isinstance(data, list):
        raise ValueError("输入文件的顶层结构必须是 JSON 数组（list）。")

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
        raise ValueError(
            f"无法初始化 OpenAI 客户端: {e}。请提供 api_key 或设置 OPENAI_API_KEY。"
        )

    total = len(data)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for i, item in enumerate(data):
            query = item.get("query", "")
            nlp = translate_one_query(client, query, model)
            item["nlp"] = nlp

            if not first:
                f.write(",\n")
            json.dump(item, f, ensure_ascii=False, indent=2)
            f.flush()
            first = False

            print(f"已处理 {i + 1}/{total} 条")

        f.write("\n]")

    print(f"处理完成，结果已保存到 {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="将 Cypher 译为自然英语并写入 nlp 字段（一条一条）")
    parser.add_argument("--input", "-i", default="mcp_query_gnd_gpt5.json", help="输入 JSON 文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 文件路径（默认与输入同目录，文件名加 _nlp 后缀）")
    parser.add_argument("--model", "-m", default="qwen2.5-7b-instruct", help="使用的模型")
    parser.add_argument("--api-key", default="sk-edc6c171ed524d059e3053b33ea96705", help="OpenAI API Key（也可用环境变量 OPENAI_API_KEY）")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API 基础 URL（可选）")
    args = parser.parse_args()

    input_path = os.path.join(os.path.dirname(__file__), args.input) if not os.path.isabs(args.input) else args.input
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_nlp" + ext

    translate_unstructured(
        input_file=input_path,
        output_file=output_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )


if __name__ == "__main__":
    main()
