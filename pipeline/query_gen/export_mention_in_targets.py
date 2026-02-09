#!/usr/bin/env python3
"""
导出 Neo4j 中所有被 mention_in 指向的节点到 JSON 文件。
即：MATCH ()-[:mention_in]->(n) 的 n，去重后导出 id、type、concept。
使用方式: conda activate autokg311 && python export_mention_in_targets.py
"""
import json
import argparse
from pathlib import Path

DEFAULT_NEO4J_URI = "bolt://localhost:7690"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "fei123456"
DEFAULT_OUTPUT = "mention_in_target_nodes.json"


def main():
    parser = argparse.ArgumentParser(description="导出所有被 mention_in 指向的节点到 JSON")
    parser.add_argument("--uri", default=DEFAULT_NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--user", default=DEFAULT_NEO4J_USER, help="Neo4j 用户")
    parser.add_argument("--password", default=DEFAULT_NEO4J_PASSWORD, help="Neo4j 密码")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="输出文件路径")
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
    # 所有作为 mention_in 目标节点的 n，去重；type 用业务 label（排除 NGDBNode）
    query = """
    MATCH ()-[:mention_in]->(n)
    WHERE n.id IS NOT NULL
    WITH DISTINCT n
    WITH n,
         [l IN labels(n) WHERE l <> 'NGDBNode'][0] AS biz_label
    RETURN n.id AS id,
           coalesce(biz_label, labels(n)[0]) AS type,
           n.concept AS concept
    ORDER BY n.id
    """
    rows = []
    try:
        with driver.session() as session:
            result = session.run(query)
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

    print(f"已导出 {len(rows)} 个被 mention_in 指向的节点到 {out_path.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
