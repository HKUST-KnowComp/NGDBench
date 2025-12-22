from typing import Any, Dict, List, Tuple, Callable, Optional
import numpy as np
import random
import json
import os
import re
from pathlib import Path
import pandas as pd
from .base import BasePerturbationGenerator


class SemanticPerturbationGenerator(BasePerturbationGenerator):
    """Semantic perturbation generator - based on semantic logic to delete or modify data
    
    支持通过指导文件（如 paramkg.json）来定义和应用各种语义噪声
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化语义扰动生成器
        
        Args:
            config: 配置字典，可以包含:
                - guide_file: 指导文件路径（如 paramkg.json）
                - noise_profile: 噪声配置（如果不提供则使用指导文件中的 default_profile）
                - 其他配置项
        """
        super().__init__(config)
        self.guide_data = None
        self.noise_profile = None
        self._embedding_model = None
        self._embeddings_cache = {}
        
        # 加载指导文件
        guide_file = config.get('guide_file')
        if guide_file and os.path.exists(guide_file):
            self._load_guide_file(guide_file)
            # 使用配置中的 noise_profile 或指导文件中的 default_profile
            self.noise_profile = self.guide_data.get('default_profile', {})
    
    def apply_perturbation(self):
        perturbed_data_path = self._copy_dataset()
        # 如果要混合扰动，这里逻辑要改，perturbation_info得存下来
        perturbation_info = self._add_semantic_noise(perturbed_data_path)
        return perturbation_info
    
    def incomplete_perturb(self, perturbed_data_path: str) -> tuple:
        """语义扰动生成器的不完整性扰动（可选实现）"""
        return {"operations": []}
    
    def noise_perturb(self, perturbed_data_path: str) -> tuple:
        """语义扰动生成器的噪声扰动"""
        return self._add_semantic_noise(perturbed_data_path)
        
    def _load_guide_file(self, guide_file_path: str):
        """加载语义扰动指导文件"""
        try:
            with open(guide_file_path, 'r', encoding='utf-8') as f:
                self.guide_data = json.load(f)
            print(f"已加载语义扰动指导文件: {guide_file_path}")
        except Exception as e:
            print(f"加载指导文件失败: {e}")
            self.guide_data = {}
    
    def _get_embedding_model(self):
        """获取或初始化 embedding 模型（使用 sentence-transformers）"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # 使用轻量级但效果好的模型
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("已加载 sentence-transformers 模型: all-MiniLM-L6-v2")
            except ImportError:
                print("警告: sentence-transformers 未安装，请运行 pip install sentence-transformers")
                return None
            except Exception as e:
                print(f"加载 embedding 模型失败: {e}")
                return None
        return self._embedding_model
    
    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """获取文本的 embeddings"""
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
        """计算两个文本的语义相似度"""
        embeddings = self._get_embeddings([text1, text2])
        if embeddings is None:
            return 0.0
        
        # 计算余弦相似度
        emb1, emb2 = embeddings[0], embeddings[1]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def _find_most_similar(self, target: str, candidates: List[str], exclude_self: bool = True) -> Tuple[str, float]:
        """从候选列表中找到与目标最相似的文本
        
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
    
    def _add_semantic_noise(self, perturbed_data_path: str) -> Dict:
        """添加语义相关的噪声
        
        从指定路径读取 CSV 文件，根据指导文件中的 noise_types 和 noise_profile 来应用噪声，
        然后将修改后的 DataFrame 写回文件。
        """
        perturbation_info = {
            "method": "semantic_noise",
            "operations": [],
            "perturbed_data_path": perturbed_data_path
        }
        
        data_file_format = self.data_config.get("data_file_format", ".csv")
        operations = self._process_directory(perturbed_data_path, data_file_format, 'semantic')
        perturbation_info["operations"] = operations
        
        return perturbation_info
    
    
    def noise_perturb(self, perturbed_data_path: str) -> tuple:
        """语义扰动生成器的噪声扰动"""
        return self._add_semantic_noise(perturbed_data_path)
    
    def _process_file(self, file_path: str, filename: str, data_file_format: str, perturb_type: str) -> List[Dict]:
        """处理单个文件，应用语义噪声
        
        Args:
            file_path: 文件路径
            filename: 文件名
            data_file_format: 文件格式
            perturb_type: 扰动类型
            
        Returns:
            List[Dict]: 操作记录列表
        """
        operations = []
        
        try:
            df = self._read_file(file_path, data_file_format)
            
            if df is None or len(df) == 0:
                return operations
            
            # 如果有指导文件，使用指导文件定义的噪声类型
            if self.guide_data and self.noise_profile:
                noise_types = self.guide_data.get('noise_types', {})
                noise_config = self.guide_data.get('field_targets', {})
                for noise_type, ratio in self.noise_profile.items():
                    if ratio <= 0 or noise_type not in noise_types:
                        continue
                    
                    print(f"应用噪声类型: {noise_type} (比例: {ratio}) 到文件: {filename}")
                    
                    # 根据噪声类型调用对应的处理方法
                    if noise_type == 'false_edges':
                        df, ops = self._apply_false_edges(df, noise_config, ratio, file_path, filename)
                    elif noise_type == 'relation_type_noise':
                        df, ops = self._apply_relation_type_noise(df, noise_config, ratio, file_path, filename)
                    elif noise_type == 'name_typos':
                        df, ops = self._apply_name_typos(df, noise_config, ratio, file_path, filename)
                    # elif noise_type == 'source_conflicts':
                    #     df, ops = self._apply_source_conflicts(df, noise_config, ratio, file_path, filename)
                    elif noise_type == 'node_type_noise':
                        df, ops = self._apply_node_type_noise(df, noise_config, ratio, file_path, filename)
                    elif noise_type == 'id_corruption':
                        df, ops = self._apply_id_corruption(df, noise_config, ratio, file_path, filename)
                    # elif noise_type == 'duplicate_edges':
                    #     df, ops = self._apply_duplicate_edges(df, noise_config, ratio, file_path, filename)
                    elif noise_type == 'path_level_noise':
                        df, ops = self._apply_path_level_noise(df, noise_config, ratio, file_path, filename)
                    else:
                        ops = []
                    
                    operations.extend(ops)
            
            # 写回文件
            if len(operations) > 0:
                self._write_file(df, file_path, data_file_format)
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return operations
    
    # ========== 基于指导文件的噪声类型实现 ==========
    
    def _apply_false_edges(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                           file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用虚假边噪声
        
        根据 refer_fields 获取节点类型，然后对该类型的 ID 值进行采样替换，
        模拟从噪声源提取的错误生物医学关系。
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        # 获取 target_fields 和 refer_fields
        target_fields = config.get('target_fields', {}).get('base_fields', ['x_id', 'y_id'])
        refer_fields = config.get('target_fields', {}).get('refer_fields', ['x_type', 'y_type'])
        affiliation_fields = config.get('target_fields', {}).get('affiliation', ['x_name', 'y_name'])
        constraints = config.get('constraints', {})
        avoid_existing = constraints.get('avoid_existing_edges', True)
        type_consistent = constraints.get('type_consistent', True)
        
        # 检查必要的列是否存在
        available_target_fields = [f for f in target_fields if f in df.columns]
        available_refer_fields = [f for f in refer_fields if f in df.columns]
        
        if not available_target_fields:
            return df, operations
        
        # 随机选择要扰动的行
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            # 随机选择一个 target_field 进行替换
            target_field = random.choice(available_target_fields)
            
            # 找到对应的 refer_field（根据 x/y 前缀匹配）
            refer_field = None
            if type_consistent and available_refer_fields:
                # 根据 target_field 的前缀找对应的 refer_field
                prefix = target_field.split('_')[0]  # 'x' 或 'y'
                for rf in available_refer_fields:
                    if rf.startswith(prefix):
                        refer_field = rf
                        break
            
            original_value = df.at[row_idx, target_field]
            
            # 获取同类型的其他 ID 值进行采样
            if refer_field and refer_field in df.columns:
                node_type = df.at[row_idx, refer_field]
                # 获取同类型的所有 ID
                same_type_mask = df[refer_field] == node_type
                same_type_ids = df.loc[same_type_mask, target_field].unique().tolist()
            else:
                # 如果没有类型字段，从整列采样
                same_type_ids = df[target_field].unique().tolist()
            
            # 排除原始值
            candidate_ids = [id_val for id_val in same_type_ids if id_val != original_value]
            
            if not candidate_ids:
                continue
            
            # 随机选择一个新的 ID
            new_value = random.choice(candidate_ids)
            
            # 如果需要避免已存在的边，检查替换后是否会创建重复边
            if avoid_existing:
                # 构建边的唯一标识
                x_id_col = target_fields[0] if target_fields[0] in df.columns else None
                y_id_col = target_fields[1] if target_fields[1] in df.columns else None
                
                if x_id_col and y_id_col:
                    if target_field == x_id_col:
                        potential_x = new_value
                        potential_y = df.at[row_idx, y_id_col]
                    else:
                        potential_x = df.at[row_idx, x_id_col]
                        potential_y = new_value
                    
                    # 检查是否已存在这条边
                    existing_edges = df[(df[x_id_col] == potential_x) & (df[y_id_col] == potential_y)]
                    if len(existing_edges) > 0:
                        continue
            
            
            # 确定要同时修改的其它字段
            name_field = None
            original_name_value = None
            new_name_value = None
            
            if target_field == x_id_col:
                name_field = affiliation_fields[0]
            elif target_field == y_id_col:
                name_field = affiliation_fields[1]
            
            # 如果name字段存在，查找新ID对应的name值
            if name_field and name_field in df.columns:
                # 保存原始name值
                original_name_value = df.at[row_idx, name_field]
                
                # 从DataFrame中查找新ID对应的name值（在替换ID之前查找）
                # 优先从同类型的行中查找
                if refer_field and refer_field in df.columns:
                    node_type = df.at[row_idx, refer_field]
                    same_type_mask = df[refer_field] == node_type
                    # 排除当前行
                    matching_rows = df[same_type_mask & (df[target_field] == new_value) & (df.index != row_idx)]
                else:
                    # 排除当前行
                    matching_rows = df[(df[target_field] == new_value) & (df.index != row_idx)]
                
                if len(matching_rows) > 0:
                    # 取第一个匹配行的name值
                    new_name_value = matching_rows.iloc[0][name_field]
                else:
                    # 如果找不到匹配的name，保持原值
                    new_name_value = original_name_value
            
            # 执行替换ID
            df.at[row_idx, target_field] = new_value
            
            # 如果找到了新的name值，也替换name字段
            if name_field and name_field in df.columns and new_name_value is not None:
                df.at[row_idx, name_field] = new_name_value
            
            # 从文件名推断实体名称
            entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
            node_type_val = str(df.at[row_idx, refer_field]) if refer_field and refer_field in df.columns else "unknown"
            
            # 构建change信息，包含ID和name的替换
            change_info = {
                "original_value": str(original_value),
                "new_value": str(new_value),
                "modification_method": "id_swap",
                "node_type": node_type_val
            }
            
            # 如果也替换了name字段，添加到change信息中
            if name_field and name_field in df.columns and original_name_value is not None and new_name_value is not None:
                change_info["name_field"] = name_field
                change_info["original_name"] = str(original_name_value)
                change_info["new_name"] = str(new_name_value)
            
            operations.append({
                "meta": {
                    "operation_type": "false_edge",
                    "description": f"创建虚假边，替换{target_field}字段值" + (f"和{name_field}字段值" if name_field and name_field in df.columns else "")
                },
                "target": {
                    "dataset": self.data_config.get("dataset_name", "unknown"),
                    "entity_type": "edge",
                    "entity_name": entity_name,
                    "file_name": filename,
                    "file_path": file_path,
                    "scope": "single",
                    "location": {
                        "row_index": row_idx,
                        "column_name": target_field
                    }
                },
                "change": change_info
            })
        
        return df, operations
    
    def _apply_relation_type_noise(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                                    file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用关系类型噪声
        
        提取 target_fields 字段的值，对其他行该字段进行提取，
        使用 embedding 模型进行语义相似度比较，选出语义最相近的进行替换。
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        # 获取 target_fields
        target_fields = config.get('target_fields', ['relation', 'display_relation'])
        available_fields = [f for f in target_fields if f in df.columns]
        
        if not available_fields:
            return df, operations
        
        # 随机选择要扰动的行
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            # 随机选择一个 target_field 进行替换
            target_field = random.choice(available_fields)
            
            original_value = df.at[row_idx, target_field]
            
            if pd.isna(original_value) or str(original_value).strip() == '':
                continue
            
            original_value_str = str(original_value)
            
            # 获取该字段的所有唯一值（排除原值）
            all_values = df[target_field].dropna().unique().tolist()
            candidate_values = [str(v) for v in all_values if str(v) != original_value_str]
            
            if not candidate_values:
                continue
            
            # 使用语义相似度找到最相似的值
            most_similar_value, similarity = self._find_most_similar(original_value_str, candidate_values)
            
            # 更新值
            df.at[row_idx, target_field] = most_similar_value
            
            # 如果有多个相关字段（如 relation 和 display_relation），同步更新
            for field in available_fields:
                if field != target_field and field in df.columns:
                    if df.at[row_idx, field] == original_value:
                        df.at[row_idx, field] = most_similar_value
            
            # 从文件名推断实体名称
            entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
            
            operations.append({
                "meta": {
                    "operation_type": "relation_type_noise",
                    "description": f"将关系类型替换为语义相似的其他类型（相似度: {similarity:.3f}）"
                },
                "target": {
                    "dataset": self.data_config.get("dataset_name", "unknown"),
                    "entity_type": "edge",
                    "entity_name": entity_name,
                    "file_name": filename,
                    "file_path": file_path,
                    "scope": "single",
                    "location": {
                        "row_index": row_idx,
                        "column_name": target_field
                    }
                },
                "change": {
                    "original_value": original_value_str,
                    "new_value": most_similar_value,
                    "modification_method": "semantic_similar_swap",
                    "similarity_score": similarity
                }
            })
        
        return df, operations
    
    def _apply_name_typos(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                          file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用名称拼写错误噪声
        
        向实体名称注入字符级噪声，模拟OCR错误、NLP提取错误或同义词歧义
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        target_fields = config.get('target_fields', ['x_name', 'y_name'])
        available_fields = [f for f in target_fields if f in df.columns]
        
        if not available_fields:
            return df, operations
        
        typo_operations = config.get('operations', ['character_substitution', 'character_deletion', 
                                                    'character_insertion', 'case_alteration'])
        
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            for field in available_fields:
                if field in df.columns:
                    original_name = df.at[row_idx, field]
                    
                    if pd.isna(original_name) or not isinstance(original_name, str):
                        continue
                    
                    if len(original_name) > 2:  # 只对足够长的名称应用拼写错误
                        noisy_name = self._introduce_typo(original_name, typo_operations)
                        
                        if noisy_name != original_name:
                            df.at[row_idx, field] = noisy_name
                            
                            # 从文件名推断实体名称
                            entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                            
                            operations.append({
                                "meta": {
                                    "operation_type": "name_typo",
                                    "description": "向实体名称注入字符级噪声（模拟OCR/NLP错误）"
                                },
                                "target": {
                                    "dataset": self.data_config.get("dataset_name", "unknown"),
                                    "entity_type": "unknown",
                                    "entity_name": entity_name,
                                    "file_name": filename,
                                    "file_path": file_path,
                                    "scope": "single",
                                    "location": {
                                        "row_index": row_idx,
                                        "column_name": field
                                    }
                                },
                                "change": {
                                    "original_value": original_name,
                                    "new_value": noisy_name,
                                    "modification_method": "typo_injection"
                                }
                            })
        
        return df, operations
    
    def _introduce_typo(self, text: str, operations: List[str]) -> str:
        """向文本中引入拼写错误"""
        if not text or len(text) < 2:
            return text
        
        operation = random.choice(operations)
        text_list = list(text)
        
        if operation == 'character_substitution' and len(text_list) > 0:
            # 字符替换
            pos = random.randint(0, len(text_list) - 1)
            # 随机选择一个相似的字符
            similar_chars = {
                'a': ['o', 'e'], 'e': ['a', 'i'], 'i': ['e', 'l', '1'], 
                'o': ['a', '0'], 'l': ['i', '1'], 's': ['5'], 't': ['7']
            }
            char = text_list[pos].lower()
            if char in similar_chars:
                text_list[pos] = random.choice(similar_chars[char])
            else:
                text_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
        
        elif operation == 'character_deletion' and len(text_list) > 1:
            # 字符删除
            pos = random.randint(0, len(text_list) - 1)
            text_list.pop(pos)
        
        elif operation == 'character_insertion':
            # 字符插入
            pos = random.randint(0, len(text_list))
            text_list.insert(pos, random.choice('abcdefghijklmnopqrstuvwxyz'))
        
        elif operation == 'case_alteration' and len(text_list) > 0:
            # 大小写改变
            pos = random.randint(0, len(text_list) - 1)
            if text_list[pos].isalpha():
                text_list[pos] = text_list[pos].swapcase()
        
        return ''.join(text_list)
    
    def _apply_source_conflicts(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                                 file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用来源冲突噪声
        
        操作来源信息或注入来自其他典型生物医学数据库的冲突边
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        target_fields = config.get('target_fields', ['x_source', 'y_source'])
        available_fields = [f for f in target_fields if f in df.columns]
        
        if not available_fields:
            return df, operations
        
        # 获取所有已存在的来源值
        existing_sources = set()
        for field in available_fields:
            existing_sources.update(df[field].dropna().unique().tolist())
        
        # 添加一些典型的生物医学数据库来源
        alternative_sources = ['PubMed', 'DrugBank', 'UniProt', 'KEGG', 'Reactome', 'GO', 
                              'NCBI', 'OMIM', 'ChEMBL', 'STRING']
        all_sources = list(existing_sources.union(set(alternative_sources)))
        
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            for field in available_fields:
                if field in df.columns:
                    original_source = df.at[row_idx, field]
                    
                    # 选择一个不同的来源
                    candidates = [s for s in all_sources if s != original_source]
                    if candidates:
                        new_source = random.choice(candidates)
                        df.at[row_idx, field] = new_source
                        
                        # 从文件名推断实体名称
                        entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                        
                        operations.append({
                            "meta": {
                                "operation_type": "source_replacement",
                                "description": "替换数据来源信息，模拟来源冲突"
                            },
                            "target": {
                                "dataset": self.data_config.get("dataset_name", "unknown"),
                                "entity_type": "unknown",
                                "entity_name": entity_name,
                                "file_name": filename,
                                "file_path": file_path,
                                "scope": "single",
                                "location": {
                                    "row_index": row_idx,
                                    "column_name": field
                                }
                            },
                            "change": {
                                "original_value": str(original_source),
                                "new_value": new_source,
                                "modification_method": "source_swap"
                            }
                        })
        
        return df, operations
    
    def _apply_node_type_noise(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                                file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用节点类型噪声
        
        从 target_fields 列采样获取实体类型，然后进行替换以模拟实体分类错误
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        target_fields = config.get('target_fields', ['x_type', 'y_type'])
        available_fields = [f for f in target_fields if f in df.columns]
        
        if not available_fields:
            return df, operations
        
        # 从 target_fields 列收集所有实体类型（不再使用预定义的列表）
        entity_types = set()
        for field in available_fields:
            entity_types.update(df[field].dropna().unique().tolist())
        
        entity_types = list(entity_types)
        
        if len(entity_types) < 2:
            # 至少需要两种类型才能进行替换
            return df, operations
        
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            # 随机选择一个 target_field 进行替换
            target_field = random.choice(available_fields)
            
            original_type = df.at[row_idx, target_field]
            
            if pd.isna(original_type):
                continue
            
            # 从采样的类型中选择一个不同的类型
            alternative_types = [t for t in entity_types if t != original_type]
            
            if alternative_types:
                new_type = random.choice(alternative_types)
                df.at[row_idx, target_field] = new_type
                
                # 从文件名推断实体名称
                entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                
                operations.append({
                    "meta": {
                        "operation_type": "node_type_noise",
                        "description": "替换节点类型，模拟实体分类错误"
                    },
                    "target": {
                        "dataset": self.data_config.get("dataset_name", "unknown"),
                        "entity_type": "node",
                        "entity_name": entity_name,
                        "file_name": filename,
                        "file_path": file_path,
                        "scope": "single",
                        "location": {
                            "row_index": row_idx,
                            "column_name": target_field
                        }
                    },
                    "change": {
                        "original_value": str(original_type),
                        "new_value": str(new_type),
                        "modification_method": "type_swap"
                    }
                })
        
        return df, operations
    
    def _apply_id_corruption(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                              file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用ID损坏噪声
        
        通过截断、替换字符、删除前缀或生成无效ID来损坏ID
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        target_fields = config.get('target_fields', ['x_id', 'y_id'])
        available_fields = [f for f in target_fields if f in df.columns]
        
        if not available_fields:
            return df, operations
        
        corruption_operations = config.get('operations', ['truncate', 'character_swap', 
                                                          'prefix_replacement', 'random_id'])
        
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            # 随机选择一个 target_field 进行损坏
            target_field = random.choice(available_fields)
            
            original_id = df.at[row_idx, target_field]
            
            if pd.isna(original_id):
                continue
            
            original_id_str = str(original_id)
            corrupted_id = self._corrupt_id(original_id_str, corruption_operations)
            
            if corrupted_id != original_id_str:
                df.at[row_idx, target_field] = corrupted_id
                
                # 从文件名推断实体名称
                entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                
                operations.append({
                    "meta": {
                        "operation_type": "id_corruption",
                        "description": "损坏ID值（截断/字符替换/前缀替换等）"
                    },
                    "target": {
                        "dataset": self.data_config.get("dataset_name", "unknown"),
                        "entity_type": "unknown",
                        "entity_name": entity_name,
                        "file_name": filename,
                        "file_path": file_path,
                        "scope": "single",
                        "location": {
                            "row_index": row_idx,
                            "column_name": target_field
                        }
                    },
                    "change": {
                        "original_value": original_id_str,
                        "new_value": corrupted_id,
                        "modification_method": "id_corruption"
                    }
                })
        
        return df, operations
    
    def _corrupt_id(self, id_str: str, operations: List[str]) -> str:
        """损坏ID字符串"""
        if not id_str:
            return id_str
        
        operation = random.choice(operations)
        
        if operation == 'truncate' and len(id_str) > 2:
            # 截断ID
            truncate_len = random.randint(1, len(id_str) - 1)
            return id_str[:truncate_len]
        
        elif operation == 'character_swap' and len(id_str) > 1:
            # 交换字符
            id_list = list(id_str)
            pos1 = random.randint(0, len(id_list) - 1)
            pos2 = random.randint(0, len(id_list) - 1)
            id_list[pos1], id_list[pos2] = id_list[pos2], id_list[pos1]
            return ''.join(id_list)
        
        elif operation == 'prefix_replacement':
            # 替换前缀
            if re.match(r'^[A-Z]+', id_str):
                match = re.match(r'^([A-Z]+)(.*)$', id_str)
                if match:
                    prefix, rest = match.groups()
                    new_prefix = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') 
                                       for _ in range(len(prefix)))
                    return new_prefix + rest
        
        elif operation == 'random_id':
            # 生成随机ID（保持相似格式）
            if re.match(r'^[A-Z]{2}\d+$', id_str):
                # 类似 DB12345 格式
                return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2)) + \
                       ''.join(random.choice('0123456789') for _ in range(len(id_str) - 2))
            elif id_str.isdigit():
                # 纯数字ID
                return ''.join(random.choice('0123456789') for _ in range(len(id_str)))
        
        return id_str
    
    def _apply_duplicate_edges(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                                file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用重复边噪声
        
        复制现有边或注入矛盾版本以模拟异构数据源的合并
        """
        operations = []
        num_rows = len(df)
        num_to_duplicate = int(num_rows * ratio)
        
        if num_to_duplicate == 0:
            return df, operations
        
        dup_operations = config.get('operations', ['duplicate', 'contradict_relation'])
        rows_to_duplicate = random.sample(range(num_rows), min(num_to_duplicate, num_rows))
        
        new_rows = []
        
        for row_idx in rows_to_duplicate:
            operation_type = random.choice(dup_operations)
            
            if operation_type == 'duplicate':
                # 简单复制行
                duplicated_row = df.iloc[row_idx].copy()
                new_rows.append(duplicated_row)
                
                # 从文件名推断实体名称
                entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                
                operations.append({
                    "meta": {
                        "operation_type": "duplicate_edge",
                        "description": "复制现有边创建重复数据"
                    },
                    "target": {
                        "dataset": self.data_config.get("dataset_name", "unknown"),
                        "entity_type": "edge",
                        "entity_name": entity_name,
                        "file_name": filename,
                        "file_path": file_path,
                        "scope": "single",
                        "location": {
                            "row_index": row_idx
                        }
                    },
                    "change": {
                        "modification_method": "row_duplication"
                    }
                })
            
            elif operation_type == 'contradict_relation':
                # 创建矛盾的关系
                contradicted_row = df.iloc[row_idx].copy()
                
                # 如果有 relation 字段，添加否定前缀
                if 'relation' in df.columns:
                    original_relation = contradicted_row['relation']
                    contradicted_row['relation'] = f"not_{original_relation}"
                    
                    if 'display_relation' in df.columns:
                        contradicted_row['display_relation'] = f"not_{original_relation}"
                    
                    new_rows.append(contradicted_row)
                    
                    # 从文件名推断实体名称
                    entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                    
                    operations.append({
                        "meta": {
                            "operation_type": "contradict_relation",
                            "description": "创建矛盾关系边，模拟来源冲突"
                        },
                        "target": {
                            "dataset": self.data_config.get("dataset_name", "unknown"),
                            "entity_type": "edge",
                            "entity_name": entity_name,
                            "file_name": filename,
                            "file_path": file_path,
                            "scope": "single",
                            "location": {
                                "row_index": row_idx
                            }
                        },
                        "change": {
                            "original_value": str(original_relation),
                            "new_value": contradicted_row['relation'],
                            "modification_method": "relation_contradiction"
                        }
                    })
        
        # 添加新行到 DataFrame
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
        
        return df, operations
    
    def _apply_path_level_noise(self, df: pd.DataFrame, config: Dict[str, Any], ratio: float,
                                 file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """应用路径级噪声
        
        替换多跳生物医学推理链中的中间节点，模拟不正确的因果或机制链接
        通过随机替换 x_id 或 y_id 来实现
        """
        operations = []
        num_rows = len(df)
        num_to_perturb = int(num_rows * ratio)
        
        if num_to_perturb == 0:
            return df, operations
        
        target_fields = config.get('target_fields', ['x_id', 'y_id'])
        available_fields = [f for f in target_fields if f in df.columns]
        
        if not available_fields:
            return df, operations
        
        rows_to_perturb = random.sample(range(num_rows), min(num_to_perturb, num_rows))
        
        for row_idx in rows_to_perturb:
            # 随机选择一个 ID 字段进行替换
            target_field = random.choice(available_fields)
            
            original_id = df.at[row_idx, target_field]
            
            if pd.isna(original_id):
                continue
            
            # 从其他行采样一个不同的 ID
            all_ids = df[target_field].dropna().unique().tolist()
            candidate_ids = [id_val for id_val in all_ids if id_val != original_id]
            
            if candidate_ids:
                new_id = random.choice(candidate_ids)
                df.at[row_idx, target_field] = new_id
                
                # 从文件名推断实体名称
                entity_name = filename.replace(self.data_config.get("data_file_format", ".csv"), "")
                
                operations.append({
                    "meta": {
                        "operation_type": "path_level_noise",
                        "description": "替换多跳推理链中的中间节点ID"
                    },
                    "target": {
                        "dataset": self.data_config.get("dataset_name", "unknown"),
                        "entity_type": "edge",
                        "entity_name": entity_name,
                        "file_name": filename,
                        "file_path": file_path,
                        "scope": "single",
                        "location": {
                            "row_index": row_idx,
                            "column_name": target_field
                        }
                    },
                    "change": {
                        "original_value": str(original_id),
                        "new_value": str(new_id),
                        "modification_method": "intermediate_node_swap"
                    }
                })
        
        return df, operations
