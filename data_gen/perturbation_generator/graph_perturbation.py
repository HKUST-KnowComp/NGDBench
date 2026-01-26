"""
图扰动模块 - 对 NetworkX MultiDiGraph 进行各种类型的扰动

支持的扰动类型：
1. incomplete_edges: 删除边模拟数据不完整
2. false_edges: 添加假边模拟错误关系
3. relation_type_noise: 替换关系类型模拟提取误分类
4. node_type_noise: 替换节点类型模拟实体分类错误
5. name_typos: 注入字符级噪声模拟OCR/NLP错误
"""

import json
import random
import copy
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np


class PerturbationType(Enum):
    """扰动类型枚举"""
    INCOMPLETE_EDGES = "incomplete_edges"
    FALSE_EDGES = "false_edges"
    RELATION_TYPE_NOISE = "relation_type_noise"
    NODE_TYPE_NOISE = "node_type_noise"
    NAME_TYPOS = "name_typos"
    ATTRIBUTE_NOISE = "attribute_noise"

@dataclass
class EdgeRecord:
    """边扰动记录"""
    source: str
    target: str
    edge_key: int
    original_attrs: Dict[str, Any]
    new_attrs: Optional[Dict[str, Any]] = None
    operation: str = ""  # 'deleted', 'added', 'modified'
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_key": self.edge_key,
            "original_attrs": self.original_attrs,
            "new_attrs": self.new_attrs,
            "operation": self.operation
        }


@dataclass
class NodeRecord:
    """节点扰动记录"""
    node_id: str
    original_attrs: Dict[str, Any]
    new_attrs: Optional[Dict[str, Any]] = None
    operation: str = ""  # 'modified'
    field_changed: str = ""  # 改变的字段名
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "original_attrs": self.original_attrs,
            "new_attrs": self.new_attrs,
            "operation": self.operation,
            "field_changed": self.field_changed
        }


@dataclass
class PerturbationLog:
    """扰动日志记录"""
    perturbation_type: str
    timestamp: str = ""
    records: List[Dict] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "perturbation_type": self.perturbation_type,
            "timestamp": self.timestamp,
            "records": self.records,
            "summary": self.summary
        }


class GraphPerturbation:
    """
    图扰动类 - 对 NetworkX MultiDiGraph 进行各种类型的扰动
    
    主要功能：
    1. 加载扰动配置文件
    2. 执行各种类型的扰动
    3. 记录所有扰动操作的详细信息
    4. 维护噪声节点和噪声边的列表
    """
    
    def __init__(self, graph: nx.MultiDiGraph, guide_file: Optional[str] = None):
        """
        初始化图扰动器
        
        Args:
            graph: NetworkX MultiDiGraph 对象
            guide_file: 扰动指导文件路径（JSON格式）
        """
        self.original_graph = graph
        self.graph = copy.deepcopy(graph)  # 创建副本进行扰动
        
        # 扰动配置
        self.guide_data: Dict = {}
        self.noise_profile: Dict[str, float] = {}
        self.noise_types: Dict = {}
        
        # 扰动记录
        self.perturbation_logs: List[PerturbationLog] = []
        
        # 噪声节点和边的集合
        self.noisy_nodes: Set[str] = set()  # 被扰动的节点ID集合
        self.noisy_edges: Set[Tuple[str, str, int]] = set()  # 被扰动的边 (src, dst, key) 集合
        self.deleted_edges: List[EdgeRecord] = []  # 被删除的边记录
        self.added_edges: List[EdgeRecord] = []  # 被添加的边记录
        self.modified_edges: List[EdgeRecord] = []  # 被修改的边记录
        self.modified_nodes: List[NodeRecord] = []  # 被修改的节点记录
        
        # Embedding 模型（用于语义相似度计算）
        self._embedding_model = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        # 加载指导文件
        if guide_file:
            self._load_guide_file(guide_file)
    
    def _load_guide_file(self, guide_file_path: str) -> None:
        """加载扰动指导文件"""
        try:
            with open(guide_file_path, 'r', encoding='utf-8') as f:
                self.guide_data = json.load(f)
            
            self.noise_types = self.guide_data.get('noise_types', {})
            self.noise_profile = self.guide_data.get('default_profile', {})
            print(f"✅ 已加载扰动指导文件: {guide_file_path}")
            print(f"   - 扰动类型: {list(self.noise_types.keys())}")
            print(f"   - 默认配置: {self.noise_profile}")
        except Exception as e:
            print(f"❌ 加载指导文件失败: {e}")
            self.guide_data = {}
    
    def set_noise_profile(self, profile: Dict[str, float]) -> None:
        """设置自定义扰动配置"""
        self.noise_profile = profile
        print(f"✅ 已设置自定义扰动配置: {profile}")
    
    # ==================== 语义相似度相关方法 ====================
    
    def _get_embedding_model(self):
        """获取或初始化 embedding 模型（使用 sentence-transformers）"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # 使用轻量级但效果好的模型
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ 已加载 sentence-transformers 模型: all-MiniLM-L6-v2")
            except ImportError:
                print("⚠️ 警告: sentence-transformers 未安装，将使用随机选择替代语义相似度")
                print("   请运行: pip install sentence-transformers")
                return None
            except Exception as e:
                print(f"⚠️ 加载 embedding 模型失败: {e}")
                return None
        return self._embedding_model
    
    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        获取文本的 embeddings
        
        Args:
            texts: 文本列表
            
        Returns:
            embeddings 数组，如果模型不可用则返回 None
        """
        model = self._get_embedding_model()
        if model is None:
            return None
        
        # 使用缓存
        uncached_texts = [t for t in texts if t not in self._embeddings_cache]
        if uncached_texts:
            embeddings = model.encode(uncached_texts)
            for text, emb in zip(uncached_texts, embeddings):
                self._embeddings_cache[text] = emb
        
        return np.array([self._embeddings_cache[t] for t in texts])
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度（余弦相似度）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 [0, 1]
        """
        embeddings = self._get_embeddings([text1, text2])
        if embeddings is None:
            return 0.0
        
        emb1, emb2 = embeddings[0], embeddings[1]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def _find_most_similar(self, target: str, candidates: List[str], 
                          exclude_self: bool = True) -> Tuple[str, float]:
        """
        从候选列表中找到与目标最相似的文本
        
        Args:
            target: 目标文本
            candidates: 候选文本列表
            exclude_self: 是否排除与自身相同的文本
            
        Returns:
            Tuple[str, float]: (最相似的文本, 相似度分数)
        """
        if not candidates:
            return target, 1.0
        
        # 过滤候选
        filtered_candidates = [c for c in candidates if c != target] if exclude_self else candidates
        if not filtered_candidates:
            return target, 1.0
        
        # 获取所有 embeddings
        all_texts = [target] + filtered_candidates
        embeddings = self._get_embeddings(all_texts)
        
        if embeddings is None:
            # 如果无法获取 embeddings，随机选择一个
            return random.choice(filtered_candidates), 0.5
        
        target_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        
        # 计算所有候选的相似度
        similarities = []
        for i, cand_emb in enumerate(candidate_embs):
            sim = np.dot(target_emb, cand_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(cand_emb))
            similarities.append((filtered_candidates[i], float(sim)))
        
        # 按相似度排序，返回最相似的
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[0]
    
    def apply_all_perturbations(self) -> nx.MultiDiGraph:
        """
        应用所有配置的扰动
        
        Returns:
            扰动后的图
        """
        import datetime
        
        print(f"\n{'='*60}")
        print("开始应用图扰动...")
        print(f"{'='*60}")
        print(f"原始图统计: 节点数={self.graph.number_of_nodes():,}, 边数={self.graph.number_of_edges():,}")
        
        for perturbation_type, ratio in self.noise_profile.items():
            if ratio <= 0:
                continue
            
            print(f"\n📌 应用扰动: {perturbation_type} (比例: {ratio})")
            
            log = PerturbationLog(
                perturbation_type=perturbation_type,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            if perturbation_type == PerturbationType.INCOMPLETE_EDGES.value:
                records, summary = self._apply_incomplete_edges(ratio)
            elif perturbation_type == PerturbationType.FALSE_EDGES.value:
                records, summary = self._apply_false_edges(ratio)
            elif perturbation_type == PerturbationType.RELATION_TYPE_NOISE.value:
                records, summary = self._apply_relation_type_noise(ratio)
            elif perturbation_type == PerturbationType.NODE_TYPE_NOISE.value:
                records, summary = self._apply_node_type_noise(ratio)
            elif perturbation_type == PerturbationType.NAME_TYPOS.value:
                records, summary = self._apply_name_typos(ratio)
            elif perturbation_type == PerturbationType.ATTRIBUTE_NOISE.value:
                records, summary = self._apply_attribute_noise(ratio)
            else:
                print(f"   ⚠️ 未知扰动类型: {perturbation_type}")
                continue
            
            log.records = records
            log.summary = summary
            self.perturbation_logs.append(log)
            
            print(f"   ✅ 完成: {summary}")
        
        print(f"\n{'='*60}")
        print("扰动完成!")
        print(f"{'='*60}")
        print(f"扰动后图统计: 节点数={self.graph.number_of_nodes():,}, 边数={self.graph.number_of_edges():,}")
        print(f"噪声节点数: {len(self.noisy_nodes):,}")
        print(f"噪声边数: {len(self.noisy_edges):,}")
        print(f"{'='*60}\n")
        
        return self.graph
    
    # ==================== 扰动算法实现 ====================
    
    def _apply_incomplete_edges(self, ratio: float) -> Tuple[List[Dict], Dict]:
        """
        应用不完整边扰动 - 删除边模拟数据不完整
        
        Args:
            ratio: 要删除的边比例
            
        Returns:
            (记录列表, 摘要)
        """
        records = []
        edges = list(self.graph.edges(keys=True, data=True))
        num_to_delete = int(len(edges) * ratio)
        
        if num_to_delete == 0:
            return records, {"deleted_count": 0}
        
        # 随机选择要删除的边
        edges_to_delete = random.sample(edges, min(num_to_delete, len(edges)))
        
        config = self.noise_types.get('incomplete_edges', {})
        avoid_dangling = config.get('constraints', {}).get('avoid_dangling_edges', True)
        
        deleted_count = 0
        for src, dst, key, data in edges_to_delete:
            # 检查是否会产生悬挂节点（度为0的节点）
            if avoid_dangling:
                src_degree = self.graph.degree(src)
                dst_degree = self.graph.degree(dst)
                if src_degree <= 1 or dst_degree <= 1:
                    continue  # 跳过，避免产生孤立节点
            
            # 记录原始边信息
            edge_record = EdgeRecord(
                source=src,
                target=dst,
                edge_key=key,
                original_attrs=dict(data),
                operation='deleted'
            )
            
            # 删除边
            self.graph.remove_edge(src, dst, key=key)
            
            # 更新记录
            self.deleted_edges.append(edge_record)
            self.noisy_edges.add((src, dst, key))
            records.append(edge_record.to_dict())
            deleted_count += 1
        
        summary = {
            "deleted_count": deleted_count,
            "target_count": num_to_delete
        }
        return records, summary
    
    def _apply_false_edges(self, ratio: float) -> Tuple[List[Dict], Dict]:
        """
        应用虚假边扰动 - 添加不存在的边
        
        Args:
            ratio: 要添加的边比例（相对于原始边数）
            
        Returns:
            (记录列表, 摘要)
        """
        records = []
        num_edges = self.graph.number_of_edges()
        num_to_add = int(num_edges * ratio)
        
        if num_to_add == 0:
            return records, {"added_count": 0}
        
        # 获取所有节点和现有边
        nodes = list(self.graph.nodes(data=True))
        existing_edges = set((u, v) for u, v, _ in self.graph.edges(keys=True))
        
        # 收集所有关系类型
        relation_types = set()
        for _, _, data in self.graph.edges(data=True):
            if 'label' in data:
                relation_types.add(data['label'])
            if 'relation' in data:
                relation_types.add(data['relation'])
        relation_types = list(relation_types) if relation_types else ['related_to']
        
        config = self.noise_types.get('false_edges', {})
        avoid_dangling = config.get('constraints', {}).get('avoid_dangling_edges', True)
        
        added_count = 0
        attempts = 0
        max_attempts = num_to_add * 10
        
        while added_count < num_to_add and attempts < max_attempts:
            attempts += 1
            
            # 随机选择两个不同的节点
            if len(nodes) < 2:
                break
            
            src_node, src_data = random.choice(nodes)
            dst_node, dst_data = random.choice(nodes)
            
            if src_node == dst_node:
                continue
            
            # 检查边是否已存在
            if (src_node, dst_node) in existing_edges:
                continue
            
            # 随机选择一个关系类型
            relation = random.choice(relation_types)
            
            # 创建边属性
            edge_attrs = {
                'label': relation,
                'relation': relation,
                'is_noise': True  # 标记为噪声边
            }
            
            # 添加边
            edge_key = self.graph.add_edge(src_node, dst_node, **edge_attrs)
            
            # 记录添加的边
            edge_record = EdgeRecord(
                source=src_node,
                target=dst_node,
                edge_key=edge_key if edge_key is not None else 0,
                original_attrs={},  # 原始不存在
                new_attrs=edge_attrs,
                operation='added'
            )
            
            self.added_edges.append(edge_record)
            self.noisy_edges.add((src_node, dst_node, edge_key if edge_key is not None else 0))
            existing_edges.add((src_node, dst_node))
            records.append(edge_record.to_dict())
            added_count += 1
        
        summary = {
            "added_count": added_count,
            "target_count": num_to_add
        }
        return records, summary
    
    def _apply_relation_type_noise(self, ratio: float) -> Tuple[List[Dict], Dict]:
        """
        应用关系类型噪声 - 使用语义相似度替换边的关系类型
        
        提取关系类型字段的值，使用 embedding 模型进行语义相似度比较，
        选出语义最相近的关系类型进行替换，模拟提取误分类。
        
        Args:
            ratio: 要修改的边比例
            
        Returns:
            (记录列表, 摘要)
        """
        records = []
        edges = list(self.graph.edges(keys=True, data=True))
        num_to_modify = int(len(edges) * ratio)
        
        if num_to_modify == 0:
            return records, {"modified_count": 0}
        
        # 收集所有关系类型
        relation_types = []
        for _, _, _, data in edges:
            if 'label' in data:
                relation_types.append(str(data['label']))
            elif 'relation' in data:
                relation_types.append(str(data['relation']))
        
        unique_relations = list(set(relation_types))
        
        if len(unique_relations) < 2:
            return records, {"modified_count": 0, "reason": "关系类型不足"}
        
        # 随机选择要修改的边
        edges_to_modify = random.sample(edges, min(num_to_modify, len(edges)))
        
        # 获取配置中的约束条件
        config = self.noise_types.get('relation_type_noise', {})
        same_family = config.get('constraints', {}).get('same_relation_family', True)
        
        modified_count = 0
        total_similarity = 0.0
        
        for src, dst, key, data in edges_to_modify:
            original_relation = data.get('label') or data.get('relation')
            
            if original_relation is None or str(original_relation).strip() == '':
                continue
            
            original_relation_str = str(original_relation)
            
            # 获取候选关系类型（排除原值）
            candidate_relations = [r for r in unique_relations if r != original_relation_str]
            
            if not candidate_relations:
                continue
            
            # 使用语义相似度找到最相似的关系类型
            if same_family:
                # 使用语义相似度选择最相近的关系类型
                new_relation, similarity = self._find_most_similar(
                    original_relation_str, 
                    candidate_relations
                )
            else:
                # 随机选择（不考虑语义相似度）
                new_relation = random.choice(candidate_relations)
                similarity = 0.0
            
            total_similarity += similarity
            
            # 保存原始属性
            original_attrs = dict(data)
            
            # 更新边属性
            if 'label' in self.graph[src][dst][key]:
                self.graph[src][dst][key]['label'] = new_relation
            if 'relation' in self.graph[src][dst][key]:
                self.graph[src][dst][key]['relation'] = new_relation
            if 'display_relation' in self.graph[src][dst][key]:
                self.graph[src][dst][key]['display_relation'] = new_relation
            
            # 标记为噪声并记录相似度
            self.graph[src][dst][key]['is_noise'] = True
            self.graph[src][dst][key]['noise_similarity'] = similarity
            
            # 创建记录（包含相似度信息）
            record_dict = {
                "source": src,
                "target": dst,
                "edge_key": key,
                "original_attrs": original_attrs,
                "new_attrs": dict(self.graph[src][dst][key]),
                "operation": "modified",
                "change": {
                    "original_value": original_relation_str,
                    "new_value": new_relation,
                    "modification_method": "semantic_similar_swap",
                    "similarity_score": similarity
                }
            }
            
            # 记录修改
            edge_record = EdgeRecord(
                source=src,
                target=dst,
                edge_key=key,
                original_attrs=original_attrs,
                new_attrs=dict(self.graph[src][dst][key]),
                operation='modified'
            )
            
            self.modified_edges.append(edge_record)
            self.noisy_edges.add((src, dst, key))
            records.append(record_dict)
            modified_count += 1
        
        avg_similarity = total_similarity / modified_count if modified_count > 0 else 0.0
        
        summary = {
            "modified_count": modified_count,
            "target_count": num_to_modify,
            "average_similarity": round(avg_similarity, 4),
            "unique_relations_count": len(unique_relations)
        }
        return records, summary
    
    def _apply_node_type_noise(self, ratio: float) -> Tuple[List[Dict], Dict]:
        """
        应用节点类型噪声 - 使用语义相似度替换节点的类型标签
        
        提取节点类型字段的值，使用 embedding 模型进行语义相似度比较，
        选出语义最相近的节点类型进行替换，模拟实体分类错误。
        
        Args:
            ratio: 要修改的节点比例
            
        Returns:
            (记录列表, 摘要)
        """
        records = []
        nodes = list(self.graph.nodes(data=True))
        num_to_modify = int(len(nodes) * ratio)
        
        if num_to_modify == 0:
            return records, {"modified_count": 0}
        
        # 收集所有节点类型
        node_types = []
        for _, data in nodes:
            if 'label' in data:
                node_types.append(str(data['label']))
            elif 'node_type' in data:
                node_types.append(str(data['node_type']))
        
        unique_types = list(set(node_types))
        
        if len(unique_types) < 2:
            return records, {"modified_count": 0, "reason": "节点类型不足"}
        
        # 获取配置中的约束条件
        config = self.noise_types.get('node_type_noise', {})
        allow_invalid = config.get('constraints', {}).get('allow_invalid_combinations', True)
        
        # 随机选择要修改的节点
        nodes_to_modify = random.sample(nodes, min(num_to_modify, len(nodes)))
        
        modified_count = 0
        total_similarity = 0.0
        
        for node_id, data in nodes_to_modify:
            original_type = data.get('label') or data.get('node_type')
            
            if original_type is None or str(original_type).strip() == '':
                continue
            
            original_type_str = str(original_type)
            
            # 获取候选节点类型（排除原值）
            candidate_types = [t for t in unique_types if t != original_type_str]
            
            if not candidate_types:
                continue
            
            # 使用语义相似度找到最相似的节点类型
            new_type, similarity = self._find_most_similar(
                original_type_str, 
                candidate_types
            )
            
            total_similarity += similarity
            
            # 保存原始属性
            original_attrs = dict(data)
            
            # 更新节点属性
            if 'label' in self.graph.nodes[node_id]:
                self.graph.nodes[node_id]['label'] = new_type
            if 'node_type' in self.graph.nodes[node_id]:
                self.graph.nodes[node_id]['node_type'] = new_type
            
            # 记录相似度
            self.graph.nodes[node_id]['noise_similarity'] = similarity
            
            # 创建记录（包含相似度信息）
            record_dict = {
                "node_id": node_id,
                "original_attrs": original_attrs,
                "new_attrs": dict(self.graph.nodes[node_id]),
                "operation": "modified",
                "field_changed": "node_type",
                "change": {
                    "original_value": original_type_str,
                    "new_value": new_type,
                    "modification_method": "semantic_similar_swap",
                    "similarity_score": similarity
                }
            }
            
            # 记录修改
            node_record = NodeRecord(
                node_id=node_id,
                original_attrs=original_attrs,
                new_attrs=dict(self.graph.nodes[node_id]),
                operation='modified',
                field_changed='node_type'
            )
            
            self.modified_nodes.append(node_record)
            self.noisy_nodes.add(node_id)
            records.append(record_dict)
            modified_count += 1
        
        avg_similarity = total_similarity / modified_count if modified_count > 0 else 0.0
        
        summary = {
            "modified_count": modified_count,
            "target_count": num_to_modify,
            "average_similarity": round(avg_similarity, 4),
            "unique_types_count": len(unique_types)
        }
        return records, summary
    
    def _find_node_name(self, data: Dict[str, Any], config: Dict = None) -> Tuple[Optional[str], Optional[str]]:
        """
        查找节点名称，支持多种可能的属性名
        
        Args:
            data: 节点属性字典
            config: 配置字典，可包含 name_fields 列表指定要查找的字段名
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (节点名称值, 属性名)
        """
        # 从配置中获取可能的名称字段列表
        if config:
            name_fields = config.get('name_fields', None)
            if name_fields:
                for field in name_fields:
                    if field in data and isinstance(data[field], str) and len(data[field]) >= 2:
                        return data[field], field
        
        # 默认优先级：name > 其他包含"name"的属性
        # 注意：不包含 'id'，因为节点的id不应该被修改
        # 1. 首先尝试 'name'
        if 'name' in data and isinstance(data['name'], str) and len(data['name']) >= 2:
            return data['name'], 'name'
        
        # 2. 查找所有包含 "name" 或 "Name" 的属性（不区分大小写）
        # 排除 'id' 和以 'id' 结尾的属性，避免修改节点标识符
        for attr_name, attr_value in data.items():
            if (attr_name.lower() != 'id' and 
                not attr_name.lower().endswith('id') and
                'name' in attr_name.lower() and 
                isinstance(attr_value, str) and 
                len(attr_value) >= 2):
                return attr_value, attr_name
        
        return None, None
    
    def _apply_name_typos(self, ratio: float) -> Tuple[List[Dict], Dict]:
        """
        应用名称拼写错误噪声 - 向节点名称注入字符级噪声
        
        Args:
            ratio: 要修改的节点比例
            
        Returns:
            (记录列表, 摘要)
        """
        records = []
        nodes = list(self.graph.nodes(data=True))
        num_to_modify = int(len(nodes) * ratio)
        
        if num_to_modify == 0:
            return records, {"modified_count": 0}
        
        config = self.noise_types.get('name_typos', {})
        typo_operations = config.get('operations', 
            ['character_substitution', 'character_deletion', 'character_insertion', 'case_alteration'])
        
        # 随机选择要修改的节点
        nodes_to_modify = random.sample(nodes, min(num_to_modify, len(nodes)))
        
        modified_count = 0
        for node_id, data in nodes_to_modify:
            # 尝试获取节点名称（支持多种属性名）
            original_name, name_field = self._find_node_name(data, config)
            
            # 如果找不到名称字段，跳过
            if original_name is None or name_field is None:
                continue
            
            # 保护：确保不会修改节点的id字段
            if name_field.lower() == 'id' or name_field.lower().endswith('id'):
                continue
            
            if len(original_name) < 3:
                continue
            
            # 引入拼写错误
            noisy_name = self._introduce_typo(original_name, typo_operations)
            
            if noisy_name == original_name:
                continue
            
            # 保存原始属性
            original_attrs = dict(data)
            
            # 更新节点名称（使用找到的属性名）
            self.graph.nodes[node_id][name_field] = noisy_name
            # 如果原属性名不是 'name'，也保存原始值到 'original_name' 以便追踪
            if name_field != 'name':
                self.graph.nodes[node_id]['original_name'] = original_name
            
            
            # 记录修改
            node_record = NodeRecord(
                node_id=node_id,
                original_attrs=original_attrs,
                new_attrs=dict(self.graph.nodes[node_id]),
                operation='modified',
                field_changed=name_field
            )
            
            self.modified_nodes.append(node_record)
            self.noisy_nodes.add(node_id)
            records.append(node_record.to_dict())
            modified_count += 1
        
        summary = {
            "modified_count": modified_count,
            "target_count": num_to_modify
        }
        return records, summary
    
    def _apply_attribute_noise(self, ratio: float) -> Tuple[List[Dict], Dict]:
        """
        应用属性噪声 - 对节点属性进行扰动
        
        对于数值属性：将数值乘以随机倍数因子 [10, 100, 1000, 10000] 或 [1/10, 1/100, 1/1000, 1/10000]
        对于字符串属性：使用拼写错误注入（类似 _introduce_typo）
        
        Args:
            ratio: 要修改的节点比例
            
        Returns:
            (记录列表, 摘要)
        """
        records = []
        nodes = list(self.graph.nodes(data=True))
        num_to_modify = int(len(nodes) * ratio)
        
        if num_to_modify == 0:
            return records, {"modified_count": 0}
        
        config = self.noise_types.get('attribute_noise', {})
        # 获取要排除的属性（如节点ID、类型等不应被扰动的属性）
        exclude_attrs = config.get('exclude_attributes', ['id', 'label', 'node_type', 'name'])
        # 获取字符串属性的拼写错误操作类型
        typo_operations = config.get('typo_operations', 
            ['character_substitution', 'character_deletion', 'character_insertion', 'case_alteration'])
        
        # 随机选择要修改的节点
        nodes_to_modify = random.sample(nodes, min(num_to_modify, len(nodes)))
        
        modified_count = 0
        numeric_attrs_count = 0
        string_attrs_count = 0
        
        for node_id, data in nodes_to_modify:
            # 保存原始属性
            original_attrs = dict(data)
            new_attrs = dict(data)
            modified = False
            changed_fields = []
            
            # 遍历所有属性
            for attr_name, attr_value in data.items():
                # 跳过排除的属性
                if attr_name in exclude_attrs:
                    continue
                
                # 保护：确保不会修改节点的id字段（包括所有以'id'结尾的属性）
                if attr_name.lower() == 'id' or attr_name.lower().endswith('id'):
                    continue
                
                # 判断属性类型并应用相应的扰动
                if isinstance(attr_value, (int, float)):
                    # 数值属性：乘以随机倍数因子
                    # 倍数因子选项：放大 [10, 100, 1000, 10000] 或缩小 [1/10, 1/100, 1/1000, 1/10000]
                    multipliers = [10, 100, 1000, 10000, 0.1, 0.01, 0.001, 0.0001]
                    multiplier = random.choice(multipliers)
                    new_value = attr_value * multiplier
                    
                    # 如果原值是整数且结果也是整数，保持为整数；否则转为浮点数
                    if isinstance(attr_value, int) and isinstance(new_value, float):
                        # 检查是否为整数（考虑浮点误差）
                        if abs(new_value - round(new_value)) < 1e-10:
                            new_value = int(round(new_value))
                    
                    new_attrs[attr_name] = new_value
                    modified = True
                    changed_fields.append(attr_name)
                    numeric_attrs_count += 1
                    
                elif isinstance(attr_value, str) and len(attr_value) >= 2:
                    # 字符串属性：使用拼写错误注入
                    noisy_value = self._introduce_typo(attr_value, typo_operations)
                    if noisy_value != attr_value:
                        new_attrs[attr_name] = noisy_value
                        modified = True
                        changed_fields.append(attr_name)
                        string_attrs_count += 1
            
            # 如果有修改，更新节点并记录
            if modified:
                # 更新图中的节点属性（只更新被修改的属性）
                for attr_name in changed_fields:
                    self.graph.nodes[node_id][attr_name] = new_attrs[attr_name]
                
                
                # 创建记录
                record_dict = {
                    "node_id": node_id,
                    "original_attrs": original_attrs,
                    "new_attrs": new_attrs,
                    "operation": "modified",
                    "field_changed": ",".join(changed_fields),
                    "change": {
                        "changed_fields": changed_fields,
                        "numeric_attrs_count": sum(1 for f in changed_fields 
                                                  if isinstance(original_attrs.get(f), (int, float))),
                        "string_attrs_count": sum(1 for f in changed_fields 
                                                 if isinstance(original_attrs.get(f), str))
                    }
                }
                
                # 记录修改
                node_record = NodeRecord(
                    node_id=node_id,
                    original_attrs=original_attrs,
                    new_attrs=new_attrs,
                    operation='modified',
                    field_changed=",".join(changed_fields)
                )
                
                self.modified_nodes.append(node_record)
                self.noisy_nodes.add(node_id)
                records.append(record_dict)
                modified_count += 1
        
        summary = {
            "modified_count": modified_count,
            "target_count": num_to_modify,
            "numeric_attrs_modified": numeric_attrs_count,
            "string_attrs_modified": string_attrs_count
        }
        return records, summary
    
    def _introduce_typo(self, text: str, operations: List[str]) -> str:
        """
        向文本中引入拼写错误
        
        Args:
            text: 原始文本
            operations: 可用的拼写错误操作列表
            
        Returns:
            带有拼写错误的文本
        """
        if not text or len(text) < 2:
            return text
        
        operation = random.choice(operations)
        text_list = list(text)
        
        if operation == 'character_substitution' and len(text_list) > 0:
            # 字符替换 - 使用相似字符
            pos = random.randint(0, len(text_list) - 1)
            similar_chars = {
                'a': ['o', 'e', 'q'], 'b': ['d', 'p'], 'c': ['e', 'o'],
                'd': ['b', 'p'], 'e': ['a', 'i', 'o'], 'f': ['t'],
                'g': ['q', 'j'], 'h': ['n', 'b'], 'i': ['l', '1', 'j'],
                'j': ['i', 'g'], 'k': ['x'], 'l': ['i', '1', 't'],
                'm': ['n', 'w'], 'n': ['m', 'h'], 'o': ['0', 'a', 'e'],
                'p': ['b', 'd', 'q'], 'q': ['g', 'p'], 'r': ['t'],
                's': ['5', 'z'], 't': ['f', 'l', '7'], 'u': ['v', 'w'],
                'v': ['u', 'w'], 'w': ['v', 'm'], 'x': ['k', 'z'],
                'y': ['v', 'j'], 'z': ['s', 'x'],
                '0': ['o', 'O'], '1': ['l', 'i', 'I'], '5': ['s', 'S'],
                '7': ['t', 'T']
            }
            char = text_list[pos].lower()
            if char in similar_chars:
                replacement = random.choice(similar_chars[char])
                # 保持原始大小写
                if text_list[pos].isupper():
                    replacement = replacement.upper()
                text_list[pos] = replacement
            else:
                # 随机替换
                if text_list[pos].isalpha():
                    alphabet = 'abcdefghijklmnopqrstuvwxyz'
                    if text_list[pos].isupper():
                        alphabet = alphabet.upper()
                    text_list[pos] = random.choice(alphabet)
        
        elif operation == 'character_deletion' and len(text_list) > 2:
            # 字符删除
            pos = random.randint(1, len(text_list) - 2)  # 避免删除首尾字符
            text_list.pop(pos)
        
        elif operation == 'character_insertion':
            # 字符插入
            pos = random.randint(1, len(text_list) - 1)  # 在中间插入
            # 插入相邻位置的字符副本（常见打字错误）
            char_to_insert = text_list[pos - 1] if random.random() > 0.5 else text_list[pos]
            text_list.insert(pos, char_to_insert)
        
        elif operation == 'case_alteration' and len(text_list) > 0:
            # 大小写改变
            pos = random.randint(0, len(text_list) - 1)
            if text_list[pos].isalpha():
                text_list[pos] = text_list[pos].swapcase()
        
        elif operation == 'mix':
            # 混合多种错误
            available_ops = ['character_substitution', 'character_deletion', 
                           'character_insertion', 'case_alteration']
            selected_ops = random.sample(available_ops, min(2, len(available_ops)))
            for op in selected_ops:
                text_list = list(self._introduce_typo(''.join(text_list), [op]))
        
        return ''.join(text_list)
    
    # ==================== 结果获取和保存 ====================
    
    def get_perturbed_graph(self) -> nx.MultiDiGraph:
        """获取扰动后的图"""
        return self.graph
    
    def get_noisy_nodes(self) -> Set[str]:
        """获取所有噪声节点的ID集合"""
        return self.noisy_nodes
    
    def get_noisy_edges(self) -> Set[Tuple[str, str, int]]:
        """获取所有噪声边的集合 (source, target, key)"""
        return self.noisy_edges
    
    def get_perturbation_summary(self) -> Dict:
        """获取扰动摘要"""
        return {
            "original_graph": {
                "nodes": self.original_graph.number_of_nodes(),
                "edges": self.original_graph.number_of_edges()
            },
            "perturbed_graph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            },
            "noise_statistics": {
                "noisy_nodes_count": len(self.noisy_nodes),
                "noisy_edges_count": len(self.noisy_edges),
                "deleted_edges_count": len(self.deleted_edges),
                "added_edges_count": len(self.added_edges),
                "modified_edges_count": len(self.modified_edges),
                "modified_nodes_count": len(self.modified_nodes)
            },
            "perturbation_logs": [log.to_dict() for log in self.perturbation_logs]
        }
    
    def get_detailed_records(self) -> Dict:
        """获取详细的扰动记录"""
        return {
            "deleted_edges": [r.to_dict() for r in self.deleted_edges],
            "added_edges": [r.to_dict() for r in self.added_edges],
            "modified_edges": [r.to_dict() for r in self.modified_edges],
            "modified_nodes": [r.to_dict() for r in self.modified_nodes],
            "noisy_nodes_list": list(self.noisy_nodes),
            "noisy_edges_list": [(s, t, k) for s, t, k in self.noisy_edges]
        }
    
    def save_perturbed_graph(self, output_path: str) -> None:
        """
        保存扰动后的图
        
        Args:
            output_path: 输出文件路径（.gpickle格式）
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"✅ 扰动后的图已保存: {output_path}")
    
    def save_perturbation_records(self, output_path: str) -> None:
        """
        保存扰动记录到 JSON 文件
        
        Args:
            output_path: 输出文件路径（.json格式）
        """
        records = {
            "summary": self.get_perturbation_summary(),
            "detailed_records": self.get_detailed_records()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ 扰动记录已保存: {output_path}")
    
    def save_noisy_elements(self, nodes_path: str, edges_path: str) -> None:
        """
        分别保存噪声节点和噪声边列表
        
        Args:
            nodes_path: 噪声节点列表输出路径
            edges_path: 噪声边列表输出路径
        """
        # 保存噪声节点
        with open(nodes_path, 'w', encoding='utf-8') as f:
            json.dump({
                "noisy_nodes": list(self.noisy_nodes),
                "count": len(self.noisy_nodes)
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 噪声节点列表已保存: {nodes_path}")
        
        # 保存噪声边
        with open(edges_path, 'w', encoding='utf-8') as f:
            json.dump({
                "noisy_edges": [(s, t, k) for s, t, k in self.noisy_edges],
                "count": len(self.noisy_edges)
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 噪声边列表已保存: {edges_path}")
    
    def save_all_with_timestamp(self, dataset_name: str, output_dir: str, 
                                 records_dir: Optional[str] = None,
                                 save_records: bool = True) -> Dict[str, str]:
        """
        保存扰动后的图和所有记录，使用时间戳命名
        
        命名格式: {dataset_name}_noise_{timestamp}.gpickle
        
        Args:
            dataset_name: 数据集名称（如 "Primekg"）
            output_dir: 图文件输出目录路径
            records_dir: 记录文件输出目录路径（如果为None则与output_dir相同）
            save_records: 是否同时保存扰动记录
            
        Returns:
            Dict[str, str]: 包含所有保存文件路径的字典
        """
        import datetime
        
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保图输出目录存在
        graph_output_path = Path(output_dir)
        graph_output_path.mkdir(parents=True, exist_ok=True)
        
        # 确保记录输出目录存在
        if records_dir:
            records_output_path = Path(records_dir)
        else:
            records_output_path = graph_output_path
        records_output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        base_name = f"{dataset_name}_noise_{timestamp}"
        
        # 文件路径
        graph_path = graph_output_path / f"{base_name}.gpickle"
        records_path = records_output_path / f"{base_name}_records.json"
        noisy_nodes_path = records_output_path / f"{base_name}_noisy_nodes.json"
        noisy_edges_path = records_output_path / f"{base_name}_noisy_edges.json"
        
        saved_files = {}
        
        # 保存扰动后的图
        self.save_perturbed_graph(str(graph_path))
        saved_files["graph"] = str(graph_path)
        
        # 保存记录
        if save_records:
            self.save_perturbation_records(str(records_path))
            self.save_noisy_elements(str(noisy_nodes_path), str(noisy_edges_path))
            saved_files["records"] = str(records_path)
            saved_files["noisy_nodes"] = str(noisy_nodes_path)
            saved_files["noisy_edges"] = str(noisy_edges_path)
        
        print(f"\n{'='*60}")
        print(f"📦 文件保存完成")
        print(f"   图文件目录: {output_dir}")
        if records_dir:
            print(f"   记录文件目录: {records_dir}")
        print(f"   基础名称: {base_name}")
        print(f"{'='*60}")
        
        return saved_files


def load_graph_from_gpickle(path: str) -> nx.MultiDiGraph:
    """从 gpickle 文件加载图"""
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    print(f"✅ 已加载图: {path}")
    print(f"   - 节点数: {graph.number_of_nodes():,}")
    print(f"   - 边数: {graph.number_of_edges():,}")
    return graph

