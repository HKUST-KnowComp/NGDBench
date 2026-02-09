"""
面向Neo4j的图数据构建基类
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jGraphBuilder:
    """
    负责将NetworkX MultiGraph/MultiDiGraph加载到Neo4j（5.x）中的构建器。
    
    主要特性：
    - 支持.gpickle与.graphml文件
    - 为每个数据集单独创建/重建Neo4j数据库，实现数据集隔离
    - 按批写入节点与关系，适配多种节点/关系类型
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        base_label: str = "NGDBNode",
        batch_size: int = 500,
        logger: logging.Logger | None = None,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.base_label = self._sanitize_label(base_label, "NGDBNode")
        self.batch_size = max(1, batch_size)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.logger.info("Neo4jGraphBuilder 初始化完成，目标 URI: %s", self.uri)

    def close(self) -> None:
        """关闭驱动连接。"""
        if self.driver:
            self.driver.close()

    # 使对象可用于with上下文
    def __enter__(self) -> "Neo4jGraphBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # 核心入口
    def build_from_file(
        self,
        file_path: str | Path,
        dataset_name: str | None = None,
        recreate_database: bool = True,
    ) -> Dict[str, Any]:
        """
        从文件加载图并导入到Neo4j默认数据库（neo4j）。

        Args:
            file_path: .gpickle或.graphml文件路径
            dataset_name: 数据集名称（可选），默认取文件名（仅用于日志记录）
            recreate_database: 是否先清空整个数据库再导入（默认True）

        Returns:
            导入统计信息
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"未找到图文件: {path}")

        dataset_name_display = dataset_name or path.stem
        graph = self._load_graph(path)
        self.logger.info(
            "开始导入图: %s，节点数=%d，边数=%d，数据集=%s（导入到默认数据库neo4j）",
            path.name,
            graph.number_of_nodes(),
            graph.number_of_edges(),
            dataset_name_display,
        )

        self._prepare_database(recreate=recreate_database)
        self._create_node_id_constraint()

        node_count = self._import_nodes(graph)
        edge_count = self._import_edges(graph)

        summary = {
            "database": "neo4j",
            "nodes_imported": node_count,
            "edges_imported": edge_count,
            "source_file": str(path),
        }
        self.logger.info("导入完成: %s", summary)
        return summary

    def clear_database(self) -> None:
        """清空默认数据库（neo4j）中的所有节点与关系。使用分批删除避免内存溢出。"""
        self._prepare_database(recreate=True)

    # 数据库准备
    def _prepare_database(self, recreate: bool = True) -> None:
        """
        准备数据库（使用默认的neo4j数据库）。
        
        如果recreate=True，将清空整个数据库的所有数据。
        使用分批删除避免单次事务占用过多内存（防止 MemoryPoolOutOfMemoryError）。
        """
        default_db = "neo4j"
        delete_batch_size = 10_000  # 每批删除的节点数，避免事务内存超限

        try:
            with self.driver.session(database=default_db) as session:
                if recreate:
                    # 分批清空，避免 MATCH (n) DETACH DELETE n 单事务内存爆掉
                    total_deleted = 0
                    while True:
                        result = session.run(
                            "MATCH (n) WITH n LIMIT $limit DETACH DELETE n",
                            limit=delete_batch_size,
                        )
                        summary = result.consume()
                        counters = summary.counters
                        deleted = getattr(counters, "nodes_deleted", 0) or 0
                        total_deleted += deleted
                        if deleted == 0:
                            break
                        self.logger.debug("本批删除 %d 个节点，累计 %d", deleted, total_deleted)
                    self.logger.info("已清空数据库 %s 的所有数据（共 %d 个节点）", default_db, total_deleted)
                else:
                    # 检查是否已存在数据
                    check_query = "MATCH (n) RETURN count(n) as count"
                    result = session.run(check_query)
                    record = result.single()
                    if record and record["count"] > 0:
                        self.logger.warning(
                            "数据库 %s 已存在 %d 个节点，将追加数据（如需重建请设置recreate=True）",
                            default_db, record["count"]
                        )
            
            self.logger.info("数据库准备完成: %s", default_db)
        except Neo4jError as exc:
            self.logger.error("准备数据库失败: %s", exc)
            raise

    def _create_node_id_constraint(self) -> None:
        """为基础标签创建唯一约束，提升匹配性能。"""
        constraint_name = f"constraint_{self.base_label.lower()}_node_id"
        cypher = (
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
            f"FOR (n:{self.base_label}) REQUIRE n._node_id IS UNIQUE"
        )
        with self.driver.session(database="neo4j") as session:
            session.run(cypher)

    # 图加载
    def _load_graph(self, path: Path) -> nx.MultiDiGraph:
        """读取文件并统一转换为MultiDiGraph。"""
        suffix = path.suffix.lower()
        if suffix in {".gpickle", ".gpkl"}:
            import pickle
            with open(path, "rb") as f:
                graph = pickle.load(f)
        elif suffix == ".graphml":
            graph = nx.read_graphml(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

        if not isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph, nx.Graph)):
            raise TypeError(f"读取到的对象不是图: {type(graph)}")

        # 统一为有向多重图，便于处理多关系
        if isinstance(graph, nx.MultiDiGraph):
            return graph
        return nx.MultiDiGraph(graph)

    # 节点导入
    def _import_nodes(self, graph: nx.MultiDiGraph) -> int:
        total = 0
        batch: List[Dict[str, Any]] = []
        for node_id, attrs in graph.nodes(data=True):
            batch.append(self._prepare_node_record(node_id, attrs))
            if len(batch) >= self.batch_size:
                self._write_node_batch(batch)
                total += len(batch)
                batch.clear()
        if batch:
            self._write_node_batch(batch)
            total += len(batch)
        return total

    def _prepare_node_record(
        self, node_id: Any, attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        labels = self._extract_labels(attrs)
        if self.base_label not in labels:
            labels.insert(0, self.base_label)

        properties = self._clean_properties(attrs, drop_keys={"label", "type", "node_type"})
        properties["_node_id"] = str(node_id)

        label_string = ":".join(labels)
        
        # 调试：记录前几个节点的属性信息（仅在第一次时记录）
        if not hasattr(self, '_logged_sample_node'):
            sample_props = {k: v for k, v in list(properties.items())[:5]}  # 只显示前5个属性
            self.logger.debug(
                "示例节点属性 (node_id=%s, labels=%s): %s (共 %d 个属性)",
                node_id, labels, sample_props, len(properties)
            )
            self._logged_sample_node = True
        
        return {"id": str(node_id), "labels": labels, "label_string": label_string, "props": properties}

    def _write_node_batch(self, rows: List[Dict[str, Any]]) -> None:
        grouped = self._group_by_label_string(rows)
        with self.driver.session(database="neo4j") as session:
            for label_string, items in grouped.items():
                # 准备传递给 Cypher 的数据，确保结构正确
                # items 中每个元素已经包含 {"id", "labels", "label_string", "props"}
                # 我们需要传递给 Cypher 的是包含 id 和 props 的列表
                cypher_rows = [{"id": item["id"], "props": item["props"]} for item in items]
                
                # 调试：记录第一批的属性信息
                if not hasattr(self, '_logged_batch_props') and cypher_rows:
                    sample_row = cypher_rows[0]
                    sample_props_keys = list(sample_row["props"].keys())[:10]  # 显示前10个属性名
                    self.logger.info(
                        "导入节点属性示例: node_id=%s, 属性数量=%d, 属性名示例=%s",
                        sample_row["id"], len(sample_row["props"]), sample_props_keys
                    )
                    self._logged_batch_props = True
                
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (n:{label_string} {{_node_id: row.id}})
                    SET n += row.props
                    """,
                    rows=cypher_rows,
                )

    # 关系导入
    def _import_edges(self, graph: nx.MultiDiGraph) -> int:
        total = 0
        batch: List[Dict[str, Any]] = []
        for idx, (u, v, key, attrs) in enumerate(graph.edges(keys=True, data=True)):
            batch.append(self._prepare_edge_record(idx, u, v, key, attrs))
            if len(batch) >= self.batch_size:
                self._write_edge_batch(batch)
                total += len(batch)
                batch.clear()
        if batch:
            self._write_edge_batch(batch)
            total += len(batch)
        return total

    def _prepare_edge_record(
        self,
        idx: int,
        source: Any,
        target: Any,
        key: Any,
        attrs: Dict[str, Any],
    ) -> Dict[str, Any]:
        rel_type = self._extract_relation_type(attrs)
        properties = self._clean_properties(attrs, drop_keys={"label", "type", "relation"})
        edge_id = key if key is not None else idx
        properties["_edge_id"] = f"{source}->{target}:{rel_type}:{edge_id}"
        return {
            "source": str(source),
            "target": str(target),
            "type": rel_type,
            "props": properties,
        }

    def _write_edge_batch(self, rows: List[Dict[str, Any]]) -> None:
        grouped = self._group_by_relation(rows)
        with self.driver.session(database="neo4j") as session:
            for rel_type, items in grouped.items():
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (s:{self.base_label} {{_node_id: row.source}})
                    MATCH (t:{self.base_label} {{_node_id: row.target}})
                    MERGE (s)-[r:{rel_type} {{_edge_id: row.props._edge_id}}]->(t)
                    SET r += row.props
                    """,
                    rows=items,
                )

    # 辅助函数
    def _extract_labels(self, attrs: Dict[str, Any]) -> List[str]:
        candidates = attrs.get("label") or attrs.get("type") or attrs.get("node_type") or "Entity"
        if isinstance(candidates, (list, tuple, set)):
            labels = [self._sanitize_label(str(v), "Entity") for v in candidates if str(v).strip()]
        else:
            labels = [self._sanitize_label(str(candidates), "Entity")]
        return labels or ["Entity"]

    def _extract_relation_type(self, attrs: Dict[str, Any]) -> str:
        candidate = attrs.get("relation") or attrs.get("label") or attrs.get("type") or "RELATED_TO"
        return self._sanitize_relationship(candidate, "RELATED_TO")

    def _sanitize_label(self, label: str, default: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", label.strip())
        if cleaned and cleaned[0].isdigit():
            cleaned = f"N_{cleaned}"
        return cleaned or default

    def _sanitize_relationship(self, rel: str, default: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]", "_", rel.strip())
        if not cleaned or cleaned[0].isdigit():
            cleaned = f"R_{cleaned}" if cleaned else default
        return cleaned or default

    def _sanitize_db_name(self, name: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_\-]", "_", name.strip())
        if not cleaned:
            raise ValueError("数据库名称不能为空")
        return cleaned

    def _clean_properties(self, attrs: Dict[str, Any], drop_keys: Iterable[str]) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        for k, v in attrs.items():
            if k in drop_keys:
                continue
            props[k] = self._make_neo4j_safe(v)
        return props

    def _make_neo4j_safe(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._make_neo4j_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._make_neo4j_safe(v) for k, v in value.items()}
        return str(value)

    def _group_by_label_string(
        self, rows: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["label_string"], []).append(row)
        return grouped

    def _group_by_relation(
        self, rows: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["type"], []).append(row)
        return grouped