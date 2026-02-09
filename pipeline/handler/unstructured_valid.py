import json
import os
import sys
from typing import List, Dict, Any, Optional

from openai import OpenAI


def _init_client(api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> OpenAI:
    """
    初始化 OpenAI 客户端，逻辑与 translate.py 保持一致。
    """
    client_kwargs: Dict[str, Any] = {}
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
            f"无法初始化OpenAI客户端: {e}。请确保提供了api_key或设置了OPENAI_API_KEY环境变量。"
        )
    return client


def _build_system_prompt(dataset: str = "mcp") -> str:
    """
    v1：使用 Cypher query + 上下文进行推理，并同时生成 query_nlp。
    （保留以兼容旧版本，不在本次 mcp v2 流程中使用。）
    """
    if dataset == "multi_fin":
        dataset_desc = (
            "The data comes from a financial-activity event dataset. "
            "Queries and context describe entities and events in financial scenarios "
            "such as accounts, loans, transactions, companies and their relationships.\n\n"
        )
    else:
        dataset_desc = (
            "The data comes from an MCP (Model Context Protocol) tools and event-graph dataset "
            "named 'mcp', where queries and context are about tools, functions, user requirement, "
            "tool trajectories etc.\n\n"
        )

    return (
        "You are an expert assistant that answers questions using Cypher queries and "
        "unstructured context from tools, documentation and event graphs.\n\n"
        + dataset_desc +
        "You will receive, for each task:\n"
        "- A Cypher query over an event/knowledge graph (field: 'query').\n"
        "- A list of unstructured context strings, usually tool/function descriptions, "
        "documentation, or other related texts.\n\n"
        "Your job is to carefully read BOTH the query and the context, and then:\n"
        "1. First understand the Cypher query semantics, and write one concise, logically "
        "correct and natural-language description of what the Cypher query is doing "
        "(output it in the JSON field 'query_nlp').\n"
        "2. Then analyze the provided unstructured context and search for evidence that can "
        "support an answer implied by the query.\n"
        "3. Decide whether the context contains enough information to infer a correct answer.\n"
        "4. If yes, produce an answer grounded ONLY in the provided context, and explicitly "
        "mark which snippets of the context are your evidence.\n"
        "5. If the context is insufficient, conflicting, or does not allow you to infer a "
        "reliable answer, you MUST return 'null' and the reason.\n\n"
        "Output format requirements (CRITICAL):\n"
        "- You MUST return a single JSON object.\n"
        "- The JSON schema is:\n"
        "  {\n"
        "    \"answer\": <any JSON value or null>,\n"
        "    \"evidence\": [<string>, ...],\n"
        "    \"query_nlp\": <string>\n"
        "  }\n"
        "- 'answer' must capture your best grounded answer. If you cannot answer, set it to null.\n"
        "- 'evidence' must be an array of strings; each string should be an exact or nearly-exact "
        "snippet from the context that supports the answer.\n"
        "- 'query_nlp' must be one short, fluent natural-language description of the Cypher query, "
        "logically faithful to the query semantics.\n"
        "- DO NOT include any extra keys.\n"
        "- DO NOT include explanations outside the JSON.\n"
        "- DO NOT wrap the JSON in backticks or any additional text.\n"
    )


def _build_user_message(item: Dict[str, Any]) -> str:
    """
    构造传给大模型的 user message，把 query + context 打包给模型阅读。
    这里假设结构类似 mcp_answer_mentions.json：
        - query: Cypher 查询
        - answer: （可选）标准答案，仅用于调试 / 评估，不要求模型看到
        - mention_in_nodes: List[str]，来自 MCP 工具文档等的上下文
    """
    query = item.get("query", "")
    # mcp_answer_mentions.json 使用的是 mention_in_nodes
    context_nodes: List[str] = item.get("mention_in_nodes") or item.get("context") or []
    if not isinstance(context_nodes, list):
        context_nodes = [str(context_nodes)]

    prompt_parts: List[str] = []
    prompt_parts.append("Below is one QA task.\n")
    prompt_parts.append("Cypher query (from event graph):\n")
    prompt_parts.append(query or "(empty)")
    prompt_parts.append("\n\nUnstructured context from MCP tools and related sources:\n")

    for idx, ctx in enumerate(context_nodes, 1):
        prompt_parts.append(f"[CONTEXT #{idx}]\n{ctx}\n")

    prompt_parts.append(
        "\nYour task:\n"
        "- Use ONLY the above context to infer the answer that this query is effectively asking for.\n"
        "- If you can infer an answer, return it as JSON field 'answer' and list the supporting "
        "context snippets in 'evidence'.\n"
        "- If you cannot infer an answer, set 'answer' to null and 'evidence' to an empty array.\n"
        "- Remember: only output the JSON object, nothing else."
    )

    return "".join(prompt_parts)
def _build_system_prompt_v2(dataset: str = "mcp") -> str:
    """
    v2：使用已经生成好的自然语言问题（nlp）+ 非结构化上下文来回答问题。
    """
    if dataset == "multi_fin":
        dataset_desc = (
            "The data comes from a financial-activity event dataset. "
            "Questions and context describe financial entities, events and their relationships.\n\n"
        )
    else:
        dataset_desc = (
            "The data comes from an MCP (Model Context Protocol) tools and event-graph dataset "
            "named 'mcp', where questions and context are about tools, functions, user requests "
            "and tool trajectories.\n\n"
        )

    return (
        "You are an expert assistant that answers questions using ONLY the provided unstructured context.\n\n"
        + dataset_desc +
        "For each task you will receive:\n"
        "- One natural-language question that describes the semantics of a Cypher query (field: 'nlp').\n"
        "- A list of unstructured context snippets (field: 'mention_in_nodes'), usually tool/function "
        "descriptions, documentation, or other related texts.\n\n"
        "Your job:\n"
        "1. Carefully read the question and ALL context snippets.\n"
        "2. Decide whether the context contains enough information to answer the question.\n"
        "3. If yes, answer the question grounded ONLY in the provided context, and list the supporting "
        "context snippets as evidence.\n"
        "4. If the context is insufficient, conflicting, or does not allow you to infer a reliable "
        "answer, you MUST set 'answer' to null.\n\n"
        "Output format (CRITICAL):\n"
        "- You MUST return a single JSON object.\n"
        "- The JSON schema is:\n"
        "  {\n"
        "    \"answer\": <any JSON value or null>,\n"
        "    \"evidence\": [<string>, ...]\n"
        "  }\n"
        "- 'evidence' must be an array of strings; each string should be an exact or nearly-exact "
        "snippet from the context that supports the answer.\n"
        "- DO NOT include any extra keys.\n"
        "- DO NOT include explanations outside the JSON.\n"
        "- DO NOT wrap the JSON in backticks or any additional text.\n"
    )


def _build_user_message_v2(item: Dict[str, Any]) -> str:
    """
    v2：使用 nlp 作为自然语言问题，mention_in_nodes 作为上下文。

    适配 mcp_answer_mentions_nlp.json 的结构：
        - nlp: 由 translate_unstructure 生成的自然语言问题
        - mention_in_nodes: List[str]，来自 MCP 工具文档等的上下文
    """
    nlp_question = item.get("nlp", "") or ""
    context_nodes: List[str] = item.get("mention_in_nodes") or item.get("context") or []
    if not isinstance(context_nodes, list):
        context_nodes = [str(context_nodes)]

    parts: List[str] = []
    parts.append("Below is one QA task.\n\n")
    parts.append("Question (natural-language description of the query semantics):\n")
    parts.append(nlp_question or "(empty)")
    parts.append("\n\nUnstructured context from MCP tools and related sources:\n")

    for idx, ctx in enumerate(context_nodes, 1):
        parts.append(f"[CONTEXT #{idx}]\n{ctx}\n")

    parts.append(
        "\nYour task:\n"
        "- Use ONLY the above context snippets to answer the Question.\n"
        "- If you can infer an answer, return it as JSON field 'answer' and list the supporting "
        "context snippets in 'evidence'.\n"
        "- If you cannot infer an answer, set 'answer' to null and 'evidence' to an empty array.\n"
        "- Remember: only output the JSON object, nothing else."
    )

    return "".join(parts)
def run_unstructured_validation(
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "qwen2.5-7b-instruct",
    dataset: str = "mcp",
) -> None:
    """
    对非结构化上下文（例如 MCP 工具描述 + 事件图查询）进行验证：
    - 读取 input_file（JSON 数组），每个元素包含 query 和上下文字段；
    - 调用大模型，根据 system_prompt 的要求返回 answer + evidence；
    - 将结果写入 output_file 中（在原对象上新增 'llm_answer' 字段）。

    建议用于测试的文件：query_module/mcp_answer_mentions.json
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        print("输入文件为空，无需处理")
        return

    client = _init_client(api_key=api_key, base_url=base_url)
    # 根据数据集类型构造不同的 system prompt 语境
    system_prompt = _build_system_prompt(dataset=dataset or "mcp")

    # 实时流式写出结果，避免等全部处理完才落盘
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        first_item = True

        for idx, item in enumerate(data, 1):
            user_msg = _build_user_message(item)

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
                content = resp.choices[0].message.content.strip()

                # 尝试解析为 JSON；如果失败，就认为模型输出不符合规范，标为 null
                llm_answer: Dict[str, Any]
                try:
                    llm_answer = json.loads(content)
                except Exception:
                    llm_answer = {"answer": None, "evidence": []}

            except Exception as e:
                print(f"[{idx}/{len(data)}] 调用LLM失败: {e}")
                llm_answer = {"answer": None, "evidence": []}

            item_with_llm = dict(item)
            # 若已有 llm_answer 则覆盖，没有则追加
            item_with_llm["llm_answer"] = llm_answer

            # 追加写入当前条目
            if not first_item:
                f.write(",\n")
            json.dump(item_with_llm, f, ensure_ascii=False, indent=2)
            f.flush()
            first_item = False

            print(f"[{idx}/{len(data)}] 已处理并写入一条样本")

        f.write("\n]\n")

    print(f"处理完成，结果已实时写入 {output_file}")


def run_unstructured_validation_v2(
    input_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "qwen2.5-7b-instruct",
    dataset: str = "mcp",
) -> None:
    """
    v2：使用已经生成好的 nlp（自然语言问题）+ mention_in_nodes 作为上下文，
    让大模型直接回答问题。

    - 读取 input_file（JSON 数组），每个元素至少包含 nlp 和 mention_in_nodes；
    - 调用大模型，根据 system_prompt_v2 的要求返回 answer + evidence；
    - 将结果写入 output_file 中（在原对象上新增 'llm_answer' 字段，其 query_nlp 直接复用 nlp）。

    推荐用于测试的文件：query_module/mcp_answer_mentions_nlp.json
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        print("输入文件为空，无需处理")
        return

    client = _init_client(api_key=api_key, base_url=base_url)
    system_prompt = _build_system_prompt_v2(dataset=dataset or "mcp")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        first_item = True

        for idx, item in enumerate(data, 1):
            user_msg = _build_user_message_v2(item)

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
                content = resp.choices[0].message.content.strip()

                # 期望只有 answer 和 evidence 两个字段
                try:
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise ValueError("模型未返回 JSON 对象")
                    answer = parsed.get("answer", None)
                    evidence = parsed.get("evidence", [])
                    if not isinstance(evidence, list):
                        evidence = [str(evidence)]
                except Exception:
                    answer = None
                    evidence = []

                llm_answer: Dict[str, Any] = {
                    "answer": answer,
                    "evidence": evidence,
                }

            except Exception as e:
                print(f"[{idx}/{len(data)}] 调用LLM失败: {e}")
                llm_answer = {"answer": None, "evidence": []}

            item_with_llm = dict(item)
            # 若已有 llm_answer 则覆盖，没有则追加
            item_with_llm["llm_answer"] = llm_answer

            if not first_item:
                f.write(",\n")
            json.dump(item_with_llm, f, ensure_ascii=False, indent=2)
            f.flush()
            first_item = False

            print(f"[{idx}/{len(data)}] v2 已处理并写入一条样本")

        f.write("\n]\n")

    print(f"v2 处理完成，结果已实时写入 {output_file}")


if __name__ == "__main__":
    """
    简单命令行用法：

    python unstructured_valid.py input.json output.json \\
        --api-key sk-xxx --base-url https://xxx/v1 --model qwen2.5-7b-instruct

    推荐测试：
        input.json  = ../query_module/mcp_answer_mentions.json
        output.json = ../query_module/mcp_answer_mentions_llm_valid.json
    """
    if len(sys.argv) < 3:
        print(
            "用法: python unstructured_valid.py <input_file> <output_file> "
            "[--api-key KEY] [--base-url URL] [--model MODEL] [--dataset DATASET] [--mode v1|v2]"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    
    # api_key_arg: str = ""
    # base_url_arg: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # model_arg: str = "qwen3-max-2026-01-23"   #"deepseek-v3.2"        #"qwen3-max-2026-01-23"
    # dataset_arg: str = "mcp"

    api_key_arg: str = ""
    base_url_arg: str = "https://api.poe.com/v1"
    model_arg: str = "gpt-5"   # "gemini-3-pro"
    dataset_arg: str = "mcp"
    mode_arg: str = "v2"       # 默认走 v2：使用 nlp + mention_in_nodes
    # 简单解析可选参数
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--api-key" and i + 1 < len(args):
            api_key_arg = args[i + 1]
            i += 2
        elif args[i] == "--base-url" and i + 1 < len(args):
            base_url_arg = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model_arg = args[i + 1]
            i += 2
        elif args[i] == "--dataset" and i + 1 < len(args):
            # mcp 或 multi_fin
            dataset_arg = args[i + 1]
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            # v1: 使用 query + context
            # v2: 使用 nlp + mention_in_nodes（本次需求）
            mode_arg = args[i + 1]
            i += 2
        else:
            i += 1

    if mode_arg == "v1":
        run_unstructured_validation(
            input_file=input_path,
            output_file=output_path,
            api_key=api_key_arg,
            base_url=base_url_arg,
            model=model_arg,
            dataset=dataset_arg,
        )
    else:
        # 默认使用 v2：读取 nlp + mention_in_nodes 来回答问题
        run_unstructured_validation_v2(
            input_file=input_path,
            output_file=output_path,
            api_key=api_key_arg,
            base_url=base_url_arg,
            model=model_arg,
            dataset=dataset_arg,
        )

