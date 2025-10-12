"""
文件数据源实现
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict
import networkx as nx
import pandas as pd
from .base import BaseDataSource


class FileDataSource(BaseDataSource):
    """从文件加载图数据的数据源"""
    
    def __init__(self, file_format: str = "auto", **kwargs):
        """
        初始化文件数据源
        
        Args:
            file_format: 文件格式 ('gml', 'graphml', 'edgelist', 'csv', 'auto')
            **kwargs: 其他配置参数
        """
        super().__init__(kwargs)
        self.file_format = file_format
        if kwargs.get('data_set_name'):
            self.data_set_name = kwargs.get('data_set_name')
        if kwargs.get('db_config'):
            self.db_config = kwargs.get('db_config')
        if kwargs.get('data_path'):
            self.data_path = kwargs.get('data_path')
        # else:
        #     self.data_set_name = os.path.basename(file_path)
        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    def load_data(self):
        if self.data_path is None:
            container_config = self.prepare_data_via_script()
        else:
            container_config = self.load_data_from_path()
        return container_config

        # if self.file_format == "auto":
        #     self.file_format = self._detect_format()
        
        # if self.file_format == "gml":
        #     graph = nx.read_gml(self.file_path)
        # elif self.file_format == "graphml":
        #     graph = nx.read_graphml(self.file_path)
        # elif self.file_format == "edgelist":
        #     graph = nx.read_edgelist(self.file_path)
        # elif self.file_format == "csv":
        #     graph = self._load_from_csv()
        # else:
        #     raise ValueError(f"不支持的文件格式: {self.file_format}")
        
        # if not self.validate_data(graph):
        #     raise ValueError("数据验证失败")
        
        # return graph

    def _load_ldbc_snb_bi_via_script(self):
        """调用外部脚本生成并装载 LDBC SNB BI 数据"""
        container_name = self._run_ldbc_snb_bi_prepare_script()
        _ = self._connect_to_neo4j_container(container_name)
        return container_name

    def prepare_data_via_script(self):
        if self.data_set_name == "ldbc_snb_bi":
            cfg = getattr(self, "config", {}) or {}
            base_dir = str(cfg.get("base_dir") or Path(__file__).resolve().parents[3])
            desired_scale_factor = str(cfg.get("desired_scale_factor", cfg.get("SF", 1)))
            available_memory = str(cfg.get("available_memory", cfg.get("LDBC_SNB_DATAGEN_MAX_MEM", "16g")))
            pagecache = str(cfg.get("neo4j_pagecache", "20G"))
            heapmax = str(cfg.get("neo4j_heap_max", "20G"))
            spark_home = str(cfg.get("SPARK_HOME", str(Path.home() / "spark-3.2.2-bin-hadoop3.2")))
            data_set_name = str(cfg.get("data_set_name", getattr(self, "data_set_name", "ldbc_snb_bi")))

            script_path = Path(__file__).resolve().parents[2] / "scripts" / "ldbc_snb_bi_prepare.sh"
            if not script_path.exists():
                raise FileNotFoundError(f"缺少脚本: {script_path}")

            env = os.environ.copy()
            # 生成容器名：包含数据集名称与 SF，转换为小写并替换非法字符
            safe_dataset = re.sub(r"[^a-z0-9]+", "-", data_set_name.lower()).strip("-")
            container_name = f"neo4j_{safe_dataset}_sf{desired_scale_factor}"
            env.update({
                "BASE_DIR": base_dir,
                "SF": desired_scale_factor,
                "LDBC_SNB_DATAGEN_MAX_MEM": available_memory,
                "NEO4J_PAGECACHE": pagecache,
                "NEO4J_HEAP_MAX": heapmax,
                "SPARK_HOME": spark_home,
                "NEO4J_CONTAINER_NAME": container_name,
            })

            subprocess.run(["bash", str(script_path)], check=True, env=env)

        return container_name

    def load_data_from_path(self):
        """从现有数据路径加载数据，如果是 ldbc_snb_bi 则执行 load-in-one-step.sh"""
        if self.data_set_name == "ldbc_snb_bi":
            cfg = getattr(self, "config", {}) or {}
            base_dir = str(cfg.get("base_dir") or Path(__file__).resolve().parents[3])
            desired_scale_factor = str(cfg.get("desired_scale_factor", cfg.get("SF", 1)))
            data_set_name = str(cfg.get("data_set_name", getattr(self, "data_set_name", "ldbc_snb_bi")))
            
            # 生成容器名：包含数据集名称与 SF，转换为小写并替换非法字符
            safe_dataset = re.sub(r"[^a-z0-9]+", "-", data_set_name.lower()).strip("-")
            container_name = f"neo4j_{safe_dataset}_sf{desired_scale_factor}"
            
            # 设置数据目录路径
            ldbc_data_dir = str(cfg.get("ldbc_snb_data_dir", f"{base_dir}/ldbc_snb_bi"))
            
            # 构建脚本路径 - 指向 ngdb_benchmark/scripts/ldbc_snb_bi/load_from_existing_data.sh
            script_path = Path(__file__).resolve().parents[2] / "scripts" / "ldbc_snb_bi" / "load_from_existing_data.sh"
            if not script_path.exists():
                raise FileNotFoundError(f"缺少脚本: {script_path}")
            
            # 获取 Neo4j 配置
            pagecache = str(cfg.get("neo4j_pagecache", "20G"))
            heapmax = str(cfg.get("neo4j_heap_max", "20G"))
            
            env = os.environ.copy()
            env.update({
                "SF": desired_scale_factor,
                "NEO4J_CONTAINER_NAME": container_name,
                "LDBC_SNB_DATA_DIR": ldbc_data_dir,
                "NEO4J_PAGECACHE": pagecache,
                "NEO4J_HEAP_MAX": heapmax,
            })
            
            # 执行脚本
            subprocess.run(["bash", str(script_path)], check=True, env=env)
            
            return container_name
        else:
            # 处理其他数据集类型
            return self.data_path

    def get_metadata(self) -> Dict[str, Any]:
        """获取文件数据集元信息"""
        graph = self.graph
        return {
            "file_path": self.file_path,
            "file_format": self.file_format,
            "file_size": os.path.getsize(self.file_path),
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "is_directed": graph.is_directed(),
            "node_attributes": list(graph.nodes(data=True))[0][1].keys() if graph.nodes() else [],
            "edge_attributes": list(graph.edges(data=True))[0][2].keys() if graph.edges() else []
        }
    
    def _detect_format(self) -> str:
        """自动检测文件格式"""
        ext = os.path.splitext(self.file_path)[1].lower()
        format_map = {
            ".gml": "gml",
            ".graphml": "graphml", 
            ".xml": "graphml",
            ".edgelist": "edgelist",
            ".txt": "edgelist",
            ".csv": "csv"
        }
        return format_map.get(ext, "edgelist")
    
    def _load_from_csv(self) -> nx.Graph:
        """从CSV文件加载图数据"""
        df = pd.read_csv(self.file_path)
        
        # 假设CSV格式为 source, target, [weight], [other_attributes]
        if len(df.columns) < 2:
            raise ValueError("CSV文件至少需要两列(source, target)")
        
        graph = nx.Graph()
        
        for _, row in df.iterrows():
            source, target = row.iloc[0], row.iloc[1]
            edge_attrs = {}
            
            # 添加边属性
            if len(row) > 2:
                for i, col in enumerate(df.columns[2:], 2):
                    edge_attrs[col] = row.iloc[i]
            
            graph.add_edge(source, target, **edge_attrs)
        
        return graph
