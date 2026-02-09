#!/usr/bin/env python3
"""
导出 Neo4j 中 50 个节点的 id、type、concept 到文件。
使用方式: conda activate autokg311 && python export_nodes_sample.py
"""
import json
import argparse
from pathlib import Path

# 与 qgen_test.py 保持一致
DEFAULT_NEO4J_URI = "bolt://localhost:7690"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "fei123456"
DEFAULT_OUTPUT = "nodes_sample_50.json"
DEFAULT_LIMIT = 50


def main():
    parser = argparse.ArgumentParser(description="导出节点 id/type/concept 到文件")
    parser.add_argument("--uri", default=DEFAULT_NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--user", default=DEFAULT_NEO4J_USER, help="Neo4j 用户")
    parser.add_argument("--password", default=DEFAULT_NEO4J_PASSWORD, help="Neo4j 密码")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="输出文件路径")
    parser.add_argument("--limit", "-n", type=int, default=DEFAULT_LIMIT, help="节点数量")
    parser.add_argument("--type", "-t", type=str, default=None, help="只采样指定 type 的节点，如 event、entity、passage")
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("请先安装 neo4j: pip install neo4j")
        print("或使用 conda 环境: conda activate autokg311")
        return 1

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )
    # type: 优先用 entity/event/passage 等业务 label，排除 NGDBNode；可加 --type 只采样某类节点
    if args.type:
        query = """
        MATCH (n)
        WHERE n.id IS NOT NULL AND $node_type IN labels(n)
        WITH n,
             [l IN labels(n) WHERE l <> 'NGDBNode'][0] AS biz_label
        RETURN n.id AS id,
               coalesce(biz_label, labels(n)[0]) AS type,
               n.concept AS concept
        LIMIT $limit
        """
        params = {"limit": args.limit, "node_type": args.type}
    else:
        query = """
        MATCH (n)
        WHERE n.id IS NOT NULL
        WITH n,
             [l IN labels(n) WHERE l <> 'NGDBNode'][0] AS biz_label
        RETURN n.id AS id,
               coalesce(biz_label, labels(n)[0]) AS type,
               n.concept AS concept
        LIMIT $limit
        """
        params = {"limit": args.limit}
    rows = []
    try:
        with driver.session() as session:
            result = session.run(query, **params)
            for record in result:
                rows.append({
                    "id": record["id"],
                    "type": record["type"],
                    "concept": record["concept"] if record["concept"] is not None else [],
                })
    finally:
        driver.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if args.type:
        print(f"已导出 {len(rows)} 个 type={args.type} 的节点到 {out_path.absolute()}")
    else:
        print(f"已导出 {len(rows)} 个节点到 {out_path.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
