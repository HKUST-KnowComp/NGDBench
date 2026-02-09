"""
检查 Neo4j 数据库是否有数据（使用与 test_build.py 相同的连接配置）。
用法: python check_db.py  或  python -m db_builder.check_db
"""

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7691"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "fei123456"


def main() -> None:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            r = session.run("MATCH (n) RETURN count(n) as nodes")
            nodes = r.single()["nodes"]
            r2 = session.run("MATCH ()-[r]->() RETURN count(r) as rels")
            rels = r2.single()["rels"]
            print(f"节点数: {nodes}")
            print(f"关系数: {rels}")
            if nodes == 0 and rels == 0:
                print("结论: 数据库里没有数据")
            else:
                print("结论: 数据库里有数据")
        driver.close()
    except Exception as e:
        print("连接或查询失败:", e)


if __name__ == "__main__":
    main()
