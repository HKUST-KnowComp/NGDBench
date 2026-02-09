"""
清空 Neo4j 数据库中的所有数据（使用与 test_build.py 相同的连接配置）。
用法: python clear_db.py
"""

from build_base import Neo4jGraphBuilder

NEO4J_URI = "bolt://localhost:7691"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "fei123456"


def main() -> None:
    with Neo4jGraphBuilder(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    ) as builder:
        builder.clear_database()
    print("数据库已清空。")


if __name__ == "__main__":
    main()
