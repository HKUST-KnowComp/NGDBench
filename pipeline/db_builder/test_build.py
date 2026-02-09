"""
使用Neo4jGraphBuilder将GraphML示例导入Docker中的Neo4j容器。

在运行前确保已启动容器，例如：
docker run -d \
  --name neo4j-520 \
  -p 7689:7687 \
  -e NEO4J_AUTH=neo4j/fei123456 \
  neo4j:5.20.0
"""

from pathlib import Path

from build_base import Neo4jGraphBuilder


NEO4J_URI = "bolt://localhost:7691"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "fei123456"

# 项目根目录为 `ngdb_benchmark`，本文件位于 `ngdb_benchmark/pipeline/db_builder/`
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data_gen"
    / "graph_gen"
    / "graph_buffer"
    / "ldbc_snb_bi.gpickle"
)
DATASET_NAME = "ldbc_bi"


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"找不到示例图文件: {GRAPH_PATH}")

    with Neo4jGraphBuilder(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        batch_size=500,
    ) as builder:
        summary = builder.build_from_file(
            file_path=GRAPH_PATH,
            dataset_name=DATASET_NAME,
            recreate_database=True,
        )

    print("导入结果:", summary)


if __name__ == "__main__":
    main()