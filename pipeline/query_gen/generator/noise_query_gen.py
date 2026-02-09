"""
噪声查询生成器：针对噪声节点和边生成查询
"""

import json
import os
import random
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

from .query_generator import (
    QueryGenerator, QueryBuilder, SchemaAnalyzer, Template, 
    PropertyType, QueryResult, logger, DEFAULT_EXCLUDED_LABELS
)


class NoiseDataLoader:
    """噪声数据加载器"""
    
    def __init__(self, perturb_record_dir: str):
        """
        初始化噪声数据加载器
        
        Args:
            perturb_record_dir: 扰动记录文件夹路径
        """
        self.perturb_record_dir = perturb_record_dir
        self.noisy_nodes: Dict[str, List[str]] = {}  # {label: [node_ids]}
        self.noisy_edges: List[Tuple[str, str]] = []  # [(start_node, end_node)]
        self.noisy_node_set: Set[str] = set()  # 所有噪声节点的集合，格式为 "Label:ID"
        self.noisy_edge_set: Set[Tuple[str, str]] = set()  # 所有噪声边的集合
    
    def load_noise_data(self, noise_prefix: Optional[str] = None):
        """
        加载噪声数据
        
        Args:
            noise_prefix: 噪声文件前缀，如果为None则自动查找最新的文件
        """
        # 查找噪声文件
        noisy_nodes_file, noisy_edges_file = self._find_noise_files(noise_prefix)
        
        if not noisy_nodes_file or not noisy_edges_file:
            raise FileNotFoundError(f"未找到噪声文件，请检查目录: {self.perturb_record_dir}")
        
        logger.info(f"加载噪声节点文件: {noisy_nodes_file}")
        logger.info(f"加载噪声边文件: {noisy_edges_file}")
        
        # 加载噪声节点
        with open(noisy_nodes_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            noisy_nodes_list = data.get('noisy_nodes', [])
        
        # 解析噪声节点：格式为 "Label:ID"
        for node_str in noisy_nodes_list:
            self.noisy_node_set.add(node_str)
            if ':' in node_str:
                label, node_id = node_str.split(':', 1)
                if label not in self.noisy_nodes:
                    self.noisy_nodes[label] = []
                self.noisy_nodes[label].append(node_id)
        
        # 加载噪声边
        with open(noisy_edges_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            noisy_edges_list = data.get('noisy_edges', [])
        
        # 解析噪声边：格式为 [start_node, end_node, ...]
        for edge in noisy_edges_list:
            if len(edge) >= 2:
                start_node = edge[0]
                end_node = edge[1]
                self.noisy_edges.append((start_node, end_node))
                self.noisy_edge_set.add((start_node, end_node))
        
        logger.info(f"加载了 {len(self.noisy_node_set)} 个噪声节点，{len(self.noisy_edges)} 条噪声边")
        logger.info(f"噪声节点标签分布: {list(self.noisy_nodes.keys())}")
    
    def _find_noise_files(self, noise_prefix: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        查找噪声文件
        
        Args:
            noise_prefix: 噪声文件前缀，如果为None则自动查找最新的文件
            
        Returns:
            (noisy_nodes_file, noisy_edges_file) 元组
        """
        perturb_dir = Path(self.perturb_record_dir)
        if not perturb_dir.exists():
            raise FileNotFoundError(f"扰动记录目录不存在: {self.perturb_record_dir}")
        
        if noise_prefix:
            # 使用指定的前缀
            noisy_nodes_file = perturb_dir / f"{noise_prefix}_noisy_nodes.json"
            noisy_edges_file = perturb_dir / f"{noise_prefix}_noisy_edges.json"
            
            if noisy_nodes_file.exists() and noisy_edges_file.exists():
                return str(noisy_nodes_file), str(noisy_edges_file)
        else:
            # 自动查找最新的文件
            noisy_nodes_files = list(perturb_dir.glob("*_noisy_nodes.json"))
            noisy_edges_files = list(perturb_dir.glob("*_noisy_edges.json"))
            
            if noisy_nodes_files and noisy_edges_files:
                # 按修改时间排序，选择最新的
                noisy_nodes_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                noisy_edges_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # 尝试匹配相同前缀的文件
                latest_nodes = noisy_nodes_files[0]
                prefix = latest_nodes.stem.replace('_noisy_nodes', '')
                matching_edges = perturb_dir / f"{prefix}_noisy_edges.json"
                
                if matching_edges.exists():
                    return str(latest_nodes), str(matching_edges)
        
        return None, None
    
    def get_noisy_node_ids_by_label(self, label: str) -> List[str]:
        """获取指定标签的噪声节点ID列表"""
        return self.noisy_nodes.get(label, [])
    
    def get_random_noisy_node(self, label: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """
        随机获取一个噪声节点
        
        Args:
            label: 如果指定，只从该标签中选择
            
        Returns:
            (label, node_id) 元组，如果没有则返回None
        """
        if label:
            if label in self.noisy_nodes and self.noisy_nodes[label]:
                return (label, random.choice(self.noisy_nodes[label]))
        else:
            # 随机选择一个标签
            if self.noisy_nodes:
                label = random.choice(list(self.noisy_nodes.keys()))
                if self.noisy_nodes[label]:
                    return (label, random.choice(self.noisy_nodes[label]))
        return None
    
    def get_random_noisy_edge(self) -> Optional[Tuple[str, str]]:
        """随机获取一条噪声边"""
        if self.noisy_edges:
            return random.choice(self.noisy_edges)
        return None
    
    def is_noisy_node(self, label: str, node_id: str) -> bool:
        """检查节点是否为噪声节点"""
        node_str = f"{label}:{node_id}"
        return node_str in self.noisy_node_set
    
    def is_noisy_edge(self, start_node: str, end_node: str) -> bool:
        """检查边是否为噪声边"""
        return (start_node, end_node) in self.noisy_edge_set


class CleanQueryBuilder(QueryBuilder):
    """干净查询构建器：确保生成的查询不包含噪声节点或边"""
    
    def __init__(
        self,
        schema: SchemaAnalyzer,
        noise_loader: NoiseDataLoader,
        excluded_return_props: Optional[Set[str]] = None,
        excluded_labels: Optional[Set[str]] = None,
        dataset: Optional[str] = None,
        driver = None,
    ):
        super().__init__(schema, excluded_return_props, excluded_labels, dataset, driver)
        self.noise_loader = noise_loader
    
    def _generate_param_value(self, param_name: str, template: Template, 
                              current_params: Dict[str, Any]) -> Optional[Any]:
        """生成参数值，确保不使用噪声节点或边"""
        
        # 对于LABEL参数，排除有噪声节点的标签（或只选择没有噪声节点的标签）
        if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
            all_labels = list(self.schema.labels.keys())
            if not all_labels:
                return None
            
            # 过滤掉被排除的标签
            if self.excluded_labels:
                all_labels = [lb for lb in all_labels if lb not in self.excluded_labels]
            
            # 优先选择没有噪声节点的标签，如果所有标签都有噪声节点，则随机选择
            clean_labels = [lb for lb in all_labels if lb not in self.noise_loader.noisy_nodes]
            if clean_labels:
                return random.choice(clean_labels)
            # 如果所有标签都有噪声节点，仍然随机选择（但会在VALUE参数中过滤）
            return random.choice(all_labels) if all_labels else None
        
        # 对于VALUE参数，确保不使用噪声节点的ID
        elif param_name in ('VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE'):
            # 找到对应的LABEL参数
            label = None
            for label_key in ['LABEL1', 'LABEL', 'LABEL2', 'LABEL3', 'START_LABEL', 'LABEL4']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            if not label or label not in self.schema.labels:
                return None
            
            # 获取该标签的所有属性
            if label not in self.schema.labels:
                return None
            
            # 尝试从属性中获取样本值
            prop = None
            for prop_key in ['PROP', 'PROP1', 'PROP2', 'FILTER_PROP', 'NODE_PROP', 'START_PROP']:
                if prop_key in current_params:
                    prop = current_params.get(prop_key)
                    break
            
            if prop and prop in self.schema.labels[label].properties:
                prop_info = self.schema.labels[label].properties[prop]
                if prop_info.sample_values:
                    # 从样本值中选择（这些值应该不是噪声节点的ID）
                    return random.choice(prop_info.sample_values)
            
            # 如果找不到属性值，尝试从数据库中采样非噪声节点
            # 这里我们使用父类的方法，但需要确保不使用噪声节点ID
            value = super()._generate_param_value(param_name, template, current_params)
            
            # 检查生成的值是否是噪声节点的ID
            if value and label in self.noise_loader.noisy_nodes:
                noisy_node_ids = self.noise_loader.noisy_nodes[label]
                if str(value) in noisy_node_ids:
                    # 如果是噪声节点ID，返回None（让查询构建失败）
                    return None
            
            return value
        
        # 其他参数使用父类的方法
        return super()._generate_param_value(param_name, template, current_params)
    
    def build_query_from_clean_node(self, template: Template, label: str, node_id: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        使用指定的干净节点构建查询
        
        Args:
            template: 查询模板
            label: 干净节点的标签
            node_id: 干净节点的ID
            
        Returns:
            (查询字符串, 使用的参数) 或 (None, {}) 如果无法构建
        """
        # 验证节点不是噪声节点
        if self.noise_loader.is_noisy_node(label, node_id):
            return None, {}
        
        params_used = {}
        
        # 填充LABEL参数
        label_params = []
        for param_name in template.parameters.keys():
            if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
                label_params.append(param_name)
        
        if not label_params:
            return None, {}
        
        # 填充所有LABEL参数
        for label_param in label_params:
            params_used[label_param] = label
        
        # 填充VALUE参数
        value_params = ['VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE']
        for value_param in value_params:
            if value_param in template.parameters:
                params_used[value_param] = node_id
        
        # 构建查询
        query = self._build_query_from_params(template, params_used)
        if query:
            return query, params_used
        
        return None, {}
    
    def _parse_node_string(self, node_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析节点字符串，格式: "Label:ID"
        
        Returns:
            (label, node_id) 元组
        """
        if ':' not in node_str:
            return None, None
        
        parts = node_str.split(':', 1)
        if len(parts) != 2:
            return None, None
        
        return parts[0], parts[1]
    
    def _build_query_from_params(self, template: Template, params_used: Dict[str, Any]) -> Optional[str]:
        """
        根据已填充的参数构建查询
        
        这个方法类似于父类的build_query，但使用预填充的参数
        """
        query = template.template
        
        try:
            # 对于未填充的参数，使用父类方法生成
            replacements = {}
            for param_name, param_type in template.parameters.items():
                if param_name in params_used:
                    value = params_used[param_name]
                else:
                    # 使用父类方法生成参数值
                    value = self._generate_param_value(param_name, template, params_used)
                    if value is None:
                        # 如果无法生成参数值，返回None
                        return None
                    params_used[param_name] = value
                
                # 验证LABEL参数是否在排除列表中
                if (param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL')):
                    if self.excluded_labels and value in self.excluded_labels:
                        # 如果生成的LABEL在排除列表中，返回None
                        return None
                
                # 替换模版中的参数
                if param_name == 'VALUE' or param_name in ('VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'END_VALUE'):
                    if isinstance(value, str):
                        replacement = f"'{value}'"
                    else:
                        replacement = str(value)
                else:
                    replacement = str(value)
                
                replacements[f'${param_name}'] = replacement
            
            # 一次性替换所有参数（按照键长度倒序，避免短键被先替换导致长键匹配失败）
            sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
            for key in sorted_keys:
                query = query.replace(key, replacements[key])
            
            # 对于 dataset 为 "mcp" 或 "multi_fin" 的情况，为 concept 节点添加 id 属性过滤
            if self.dataset in ("mcp", "multi_fin"):
                query = self._add_concept_id_filter(query, params_used)
            
            return query
            
        except Exception as e:
            logger.warning(f"构建查询失败: {e}")
            return None
    
    def build_query_from_clean_edge(self, template: Template, start_node_str: str, end_node_str: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        使用指定的干净边构建查询
        
        Args:
            template: 查询模板
            start_node_str: 起点节点字符串，格式: "Label:ID"
            end_node_str: 终点节点字符串，格式: "Label:ID"
            
        Returns:
            (查询字符串, 使用的参数) 或 (None, {}) 如果无法构建
        """
        # 解析节点
        start_label, start_id = self._parse_node_string(start_node_str)
        end_label, end_id = self._parse_node_string(end_node_str)
        
        if not start_label or not end_label:
            return None, {}
        
        # 验证边不是噪声边
        if self.noise_loader.is_noisy_edge(start_node_str, end_node_str):
            return None, {}
        
        # 验证节点不是噪声节点
        if self.noise_loader.is_noisy_node(start_label, start_id) or \
           self.noise_loader.is_noisy_node(end_label, end_id):
            return None, {}
        
        params_used = {}
        
        # 填充START_LABEL和START_VALUE
        if 'START_LABEL' in template.parameters:
            params_used['START_LABEL'] = start_label
        if 'START_VALUE' in template.parameters:
            params_used['START_VALUE'] = start_id
        
        # 填充END_LABEL和END_VALUE
        if 'END_LABEL' in template.parameters:
            params_used['END_LABEL'] = end_label
        if 'END_VALUE' in template.parameters:
            params_used['END_VALUE'] = end_id
        
        # 填充LABEL1和LABEL2（如果存在）
        if 'LABEL1' in template.parameters:
            params_used['LABEL1'] = start_label
        if 'LABEL2' in template.parameters:
            params_used['LABEL2'] = end_label
        
        # 填充对应的VALUE参数
        if 'LABEL1' in params_used and 'VALUE' in template.parameters:
            params_used['VALUE'] = start_id
        
        # 构建查询
        query = self._build_query_from_params(template, params_used)
        if query:
            return query, params_used
        
        return None, {}


class NoiseQueryBuilder(QueryBuilder):
    """噪声查询构建器：确保生成的查询包含噪声节点或边"""
    
    def __init__(
        self,
        schema: SchemaAnalyzer,
        noise_loader: NoiseDataLoader,
        excluded_return_props: Optional[Set[str]] = None,
        excluded_labels: Optional[Set[str]] = None,
        dataset: Optional[str] = None,
        driver = None,
    ):
        super().__init__(schema, excluded_return_props, excluded_labels, dataset, driver)
        self.noise_loader = noise_loader
        self.force_use_noise = True  # 强制使用噪声数据
    
    def build_query(self, template: Template) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        根据模版构建查询（保持与父类接口兼容）
        
        注意：这个方法现在主要用于回退情况。主要逻辑在 build_query_from_noise_node 和 build_query_from_noise_edge 中
        """
        # 回退到父类方法（但会优先使用噪声数据）
        return super().build_query(template)
    
    def build_query_from_noise_node(self, template: Template, label: str, node_id: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        使用指定的噪声节点构建查询
        
        Args:
            template: 查询模板
            label: 噪声节点的标签
            node_id: 噪声节点的ID
            
        Returns:
            (查询字符串, 使用的参数) 或 (None, {}) 如果无法构建
        """
        # 检查标签是否在排除列表中
        if self.excluded_labels and label in self.excluded_labels:
            return None, {}
        
        params_used = {}
        
        # 填充LABEL参数
        label_params = []
        for param_name in template.parameters.keys():
            if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
                label_params.append(param_name)
        
        if not label_params:
            return None, {}
        
        # 填充所有LABEL参数
        for label_param in label_params:
            params_used[label_param] = label
        
        # 填充VALUE参数
        value_params = ['VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE']
        for value_param in value_params:
            if value_param in template.parameters:
                params_used[value_param] = node_id
        
        # 构建查询
        query = self._build_query_from_params(template, params_used)
        if query:
            return query, params_used
        
        return None, {}
    
    def build_query_from_noise_edge(self, template: Template, start_node_str: str, end_node_str: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        使用指定的噪声边构建查询
        
        Args:
            template: 查询模板
            start_node_str: 起点节点字符串，格式: "Label:ID"
            end_node_str: 终点节点字符串，格式: "Label:ID"
            
        Returns:
            (查询字符串, 使用的参数) 或 (None, {}) 如果无法构建
        """
        # 解析节点
        start_label, start_id = self._parse_node_string(start_node_str)
        end_label, end_id = self._parse_node_string(end_node_str)
        
        if not start_label or not end_label:
            return None, {}
        
        # 检查标签是否在排除列表中
        if self.excluded_labels:
            if start_label in self.excluded_labels or end_label in self.excluded_labels:
                return None, {}
        
        params_used = {}
        
        # 填充START_LABEL和START_VALUE
        if 'START_LABEL' in template.parameters:
            params_used['START_LABEL'] = start_label
        if 'START_VALUE' in template.parameters:
            params_used['START_VALUE'] = start_id
        
        # 填充END_LABEL和END_VALUE
        if 'END_LABEL' in template.parameters:
            params_used['END_LABEL'] = end_label
        if 'END_VALUE' in template.parameters:
            params_used['END_VALUE'] = end_id
        
        # 填充LABEL1和LABEL2（如果存在）
        if 'LABEL1' in template.parameters:
            params_used['LABEL1'] = start_label
        if 'LABEL2' in template.parameters:
            params_used['LABEL2'] = end_label
        
        # 填充对应的VALUE参数
        if 'LABEL1' in params_used and 'VALUE' in template.parameters:
            params_used['VALUE'] = start_id
        
        # 构建查询
        query = self._build_query_from_params(template, params_used)
        if query:
            return query, params_used
        
        return None, {}
    
    def _fill_params_with_noise_nodes(self, template: Template, params_used: Dict[str, Any]) -> bool:
        """
        使用噪声节点填充模板参数
        
        Returns:
            如果成功填充了至少一个噪声节点参数，返回True
        """
        filled = False
        
        # 查找所有LABEL参数
        label_params = []
        for param_name in template.parameters.keys():
            if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
                label_params.append(param_name)
        
        # 如果没有LABEL参数，无法使用噪声节点
        if not label_params:
            return False
        
        # 随机选择一个有噪声节点的标签
        noisy_labels = list(self.noise_loader.noisy_nodes.keys())
        if not noisy_labels:
            return False
        
        # 过滤掉被排除的标签
        if self.excluded_labels:
            noisy_labels = [lb for lb in noisy_labels if lb not in self.excluded_labels]
            if not noisy_labels:
                return False
        
        # 选择一个噪声标签
        selected_label = random.choice(noisy_labels)
        noisy_node_ids = self.noise_loader.noisy_nodes[selected_label]
        if not noisy_node_ids:
            return False
        
        # 选择一个噪声节点ID
        selected_node_id = random.choice(noisy_node_ids)
        
        # 填充LABEL参数
        for label_param in label_params:
            params_used[label_param] = selected_label
            filled = True
        
        # 查找对应的VALUE参数并填充
        value_params = ['VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE']
        for value_param in value_params:
            if value_param in template.parameters:
                params_used[value_param] = selected_node_id
                filled = True
        
        return filled
    
    def _fill_params_with_noise_edges(self, template: Template, params_used: Dict[str, Any]) -> bool:
        """
        使用噪声边填充模板参数
        
        Returns:
            如果成功填充了噪声边参数，返回True
        """
        if not self.noise_loader.noisy_edges:
            return False
        
        # 随机选择一条噪声边
        edge = random.choice(self.noise_loader.noisy_edges)
        if len(edge) < 2:
            return False
        
        start_node_str = edge[0]  # 格式: "Label:ID"
        end_node_str = edge[1]    # 格式: "Label:ID"
        
        # 解析节点
        start_label, start_id = self._parse_node_string(start_node_str)
        end_label, end_id = self._parse_node_string(end_node_str)
        
        if not start_label or not end_label:
            return False
        
        filled = False
        
        # 填充START_LABEL和START_VALUE
        if 'START_LABEL' in template.parameters:
            params_used['START_LABEL'] = start_label
            filled = True
        if 'START_VALUE' in template.parameters:
            params_used['START_VALUE'] = start_id
            filled = True
        
        # 填充END_LABEL和END_VALUE
        if 'END_LABEL' in template.parameters:
            params_used['END_LABEL'] = end_label
            filled = True
        if 'END_VALUE' in template.parameters:
            params_used['END_VALUE'] = end_id
            filled = True
        
        # 填充LABEL1和LABEL2（如果存在）
        if 'LABEL1' in template.parameters and 'LABEL1' not in params_used:
            params_used['LABEL1'] = start_label
            filled = True
        if 'LABEL2' in template.parameters and 'LABEL2' not in params_used:
            params_used['LABEL2'] = end_label
            filled = True
        
        # 填充对应的VALUE参数
        if 'LABEL1' in params_used and 'VALUE' in template.parameters:
            params_used['VALUE'] = start_id
            filled = True
        
        return filled
    
    def _parse_node_string(self, node_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析节点字符串，格式: "Label:ID"
        
        Returns:
            (label, node_id) 元组
        """
        if ':' not in node_str:
            return None, None
        
        parts = node_str.split(':', 1)
        if len(parts) != 2:
            return None, None
        
        return parts[0], parts[1]
    
    def _build_query_from_params(self, template: Template, params_used: Dict[str, Any]) -> Optional[str]:
        """
        根据已填充的参数构建查询
        
        这个方法类似于父类的build_query，但使用预填充的参数
        """
        query = template.template
        
        try:
            # 对于未填充的参数，使用父类方法生成
            replacements = {}
            for param_name, param_type in template.parameters.items():
                if param_name in params_used:
                    value = params_used[param_name]
                else:
                    # 使用父类方法生成参数值
                    value = self._generate_param_value(param_name, template, params_used)
                    if value is None:
                        # 如果无法生成参数值，返回None
                        return None
                    params_used[param_name] = value
                
                # 验证LABEL参数是否在排除列表中
                if (param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL')):
                    if self.excluded_labels and value in self.excluded_labels:
                        # 如果生成的LABEL在排除列表中，返回None
                        return None
                
                # 替换模版中的参数
                if param_name == 'VALUE' or param_name in ('VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'END_VALUE'):
                    if isinstance(value, str):
                        replacement = f"'{value}'"
                    else:
                        replacement = str(value)
                else:
                    replacement = str(value)
                
                replacements[f'${param_name}'] = replacement
            
            # 一次性替换所有参数（按照键长度倒序，避免短键被先替换导致长键匹配失败）
            sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
            for key in sorted_keys:
                query = query.replace(key, replacements[key])
            
            # 对于 dataset 为 "mcp" 或 "multi_fin" 的情况，为 concept 节点添加 id 属性过滤
            if self.dataset in ("mcp", "multi_fin"):
                query = self._add_concept_id_filter(query, params_used)
            
            return query
            
        except Exception as e:
            logger.warning(f"构建查询失败: {e}")
            return None
    
    def _generate_param_value(self, param_name: str, template: Template, 
                              current_params: Dict[str, Any]) -> Optional[Any]:
        """生成参数值，优先使用噪声数据"""
        
        # 对于LABEL参数，优先选择有噪声节点的标签
        if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL'):
            # 获取有噪声节点的标签
            noisy_labels = list(self.noise_loader.noisy_nodes.keys())
            if noisy_labels:
                # 过滤掉被排除的标签
                if self.excluded_labels:
                    noisy_labels = [lb for lb in noisy_labels if lb not in self.excluded_labels]
                
                # 如果还有可用的噪声标签，优先使用
                if noisy_labels:
                    # 80%的概率选择噪声标签，20%的概率选择普通标签
                    if random.random() < 0.8:
                        return random.choice(noisy_labels)
            
            # 回退到父类的方法
            return super()._generate_param_value(param_name, template, current_params)
        
        # 对于VALUE参数，如果对应的LABEL有噪声节点，优先使用噪声节点的ID
        elif param_name in ('VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE'):
            # 找到对应的LABEL参数
            label = None
            for label_key in ['LABEL1', 'LABEL', 'LABEL2', 'LABEL3', 'START_LABEL', 'LABEL4']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            # 如果找到了标签且该标签有噪声节点，优先使用噪声节点的ID
            if label and label in self.noise_loader.noisy_nodes:
                noisy_node_ids = self.noise_loader.noisy_nodes[label]
                if noisy_node_ids:
                    # 80%的概率使用噪声节点ID
                    if random.random() < 0.8:
                        return random.choice(noisy_node_ids)
            
            # 回退到父类的方法
            return super()._generate_param_value(param_name, template, current_params)
        
        # 对于REF_PROP参数，如果对应的LABEL有噪声节点，也可能使用噪声节点的ID作为引用
        elif param_name == 'REF_PROP':
            # 找到对应的LABEL参数（通常是LABEL1）
            label = None
            for label_key in ['LABEL1', 'LABEL', 'START_LABEL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            # 如果找到了标签且该标签有噪声节点，可以考虑使用噪声节点的某个属性
            # 但REF_PROP通常是属性名，不是值，所以这里还是使用父类方法
            # 不过我们可以检查一下是否有特殊的处理需求
            pass
        
        # 其他参数使用父类的方法
        return super()._generate_param_value(param_name, template, current_params)


class NoiseQueryGenerator(QueryGenerator):
    """噪声查询生成器：继承QueryGenerator，但确保生成的查询包含噪声节点或边"""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 template_path: str = "/query_template/template.json",
                 perturb_record_dir: str = "/data_gen/perturbation_generator/perturb_record",
                 noise_prefix: Optional[str] = None,
                 exclude_internal_id_as_return: bool = True,
                 dataset: Optional[str] = None):
        """
        初始化噪声查询生成器
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            template_path: 模版文件路径
            perturb_record_dir: 扰动记录文件夹路径
            noise_prefix: 噪声文件前缀，如果为None则自动查找最新的文件
            exclude_internal_id_as_return: 是否在返回属性中排除内部ID字段
            dataset: 数据集名称
        """
        super().__init__(uri, user, password, template_path, exclude_internal_id_as_return, dataset)
        self.perturb_record_dir = perturb_record_dir
        self.noise_prefix = noise_prefix
        self.noise_loader = NoiseDataLoader(perturb_record_dir)
    
    def initialize(self):
        """初始化所有组件，包括加载噪声数据"""
        # 先调用父类的初始化
        super().initialize()
        
        # 加载噪声数据
        self.noise_loader.load_noise_data(self.noise_prefix)
        
        # 替换QueryBuilder为NoiseQueryBuilder和CleanQueryBuilder
        excluded_labels = set(DEFAULT_EXCLUDED_LABELS)
        if self.dataset in ("mcp", "multi_fin"):
            excluded_labels.update({"passage"})
        
        self.builder = NoiseQueryBuilder(
            self.schema,
            self.noise_loader,
            excluded_return_props=self.excluded_return_props,
            excluded_labels=excluded_labels,
            dataset=self.dataset,
            driver=self.driver,  # 传递 driver 用于节点采样
        )
        
        self.clean_builder = CleanQueryBuilder(
            self.schema,
            self.noise_loader,
            excluded_return_props=self.excluded_return_props,
            excluded_labels=excluded_labels,
            dataset=self.dataset,
            driver=self.driver,  # 传递 driver 用于节点采样
        )
        
        logger.info("噪声查询生成器初始化完成")
    
    def _sample_clean_nodes(self, label: str, sample_size: int = 100) -> List[Tuple[str, str]]:
        """
        从数据库中采样干净节点（非噪声节点）
        
        Args:
            label: 节点标签
            sample_size: 采样数量
            
        Returns:
            [(label, node_id), ...] 列表
        """
        clean_nodes = []
        
        try:
            with self.driver.session() as session:
                # 获取该标签的所有节点（采样）
                query = f"MATCH (n:`{label}`) RETURN n LIMIT {sample_size * 2}"  # 多采样一些，因为要过滤噪声节点
                result = session.run(query)
                
                for record in result:
                    node = record["n"]
                    node_dict = dict(node)
                    
                    # 尝试获取节点的ID（可能是id属性或其他唯一标识）
                    node_id = None
                    # 常见的ID字段名
                    for id_field in ['id', 'ID', 'nodeId', 'node_id', '_id']:
                        if id_field in node_dict:
                            node_id = str(node_dict[id_field])
                            break
                    
                    # 如果找不到ID字段，跳过
                    if not node_id:
                        continue
                    
                    # 检查是否是噪声节点
                    if not self.noise_loader.is_noisy_node(label, node_id):
                        clean_nodes.append((label, node_id))
                        if len(clean_nodes) >= sample_size:
                            break
        except Exception as e:
            logger.warning(f"采样干净节点失败 (label={label}): {e}")
        
        return clean_nodes
    
    def _sample_clean_edges(self, rel_type: Optional[str] = None, sample_size: int = 100) -> List[Tuple[str, str]]:
        """
        从数据库中采样干净边（非噪声边）
        
        Args:
            rel_type: 关系类型，如果为None则采样所有类型
            sample_size: 采样数量
            
        Returns:
            [(start_node_str, end_node_str), ...] 列表，格式为 "Label:ID"
        """
        clean_edges = []
        
        try:
            with self.driver.session() as session:
                # 构建查询
                if rel_type:
                    query = f"MATCH (a)-[r:`{rel_type}`]->(b) RETURN labels(a) as start_labels, labels(b) as end_labels, a, b LIMIT {sample_size * 2}"
                else:
                    query = f"MATCH (a)-[r]->(b) RETURN labels(a) as start_labels, labels(b) as end_labels, a, b LIMIT {sample_size * 2}"
                
                result = session.run(query)
                
                for record in result:
                    start_labels = record["start_labels"]
                    end_labels = record["end_labels"]
                    start_node = record["a"]
                    end_node = record["b"]
                    
                    if not start_labels or not end_labels:
                        continue
                    
                    start_label = start_labels[0]  # 使用第一个标签
                    end_label = end_labels[0]
                    
                    # 获取节点ID
                    start_dict = dict(start_node)
                    end_dict = dict(end_node)
                    
                    start_id = None
                    end_id = None
                    
                    # 先尝试从常见ID字段获取
                    for id_field in ['id', 'ID', 'nodeId', 'node_id', '_id']:
                        if id_field in start_dict:
                            start_id = str(start_dict[id_field])
                            break
                    
                    for id_field in ['id', 'ID', 'nodeId', 'node_id', '_id']:
                        if id_field in end_dict:
                            end_id = str(end_dict[id_field])
                            break
                    
                    # 如果找不到ID，尝试从所有字段中找（排除内部字段）
                    if not start_id:
                        for key, value in start_dict.items():
                            if key not in ['_node_id', 'file_id'] and value:
                                start_id = str(value)
                                break
                    
                    if not end_id:
                        for key, value in end_dict.items():
                            if key not in ['_node_id', 'file_id'] and value:
                                end_id = str(value)
                                break
                    
                    if not start_id or not end_id:
                        continue
                    
                    start_node_str = f"{start_label}:{start_id}"
                    end_node_str = f"{end_label}:{end_id}"
                    
                    # 检查是否是噪声边
                    if not self.noise_loader.is_noisy_edge(start_node_str, end_node_str):
                        clean_edges.append((start_node_str, end_node_str))
                        if len(clean_edges) >= sample_size:
                            break
        except Exception as e:
            logger.warning(f"采样干净边失败: {e}")
        
        return clean_edges
    
    def generate_samples(self, target_count: Optional[int] = None, 
                        max_attempts_multiplier: int = 10,
                        max_failures_per_template: int = 1000,
                        max_answer_count: int = 20,
                        min_attempts_per_template: int = 5,
                        reset_failures_interval: float = 0.25,
                        stats_output_path: Optional[str] = None,
                        success_per_template: int = 100,
                        realtime_output_path: Optional[str] = None) -> List[QueryResult]:
        """
        生成查询样本，遍历噪声节点/边，对每个尝试匹配模板生成查询
        
        策略：
        1. 遍历所有噪声节点/边
        2. 对每个噪声节点/边，尝试匹配所有可用模板
        3. 对于匹配成功的，生成查询并执行验证
        4. 收集所有成功的查询结果
        """
        if not self.schema:
            self.initialize()
        
        # 如果 target_count 为 None，则遍历所有噪声节点和边，不设置数量限制
        if target_count is None:
            target = None
            logger.info(f"开始生成噪声查询，将遍历所有噪声节点和边（无数量限制）")
        else:
            target = target_count
            logger.info(f"开始生成噪声查询，目标数量: {target}")
        
        logger.info(f"噪声节点数量: {len(self.noise_loader.noisy_node_set)}")
        logger.info(f"噪声边数量: {len(self.noise_loader.noisy_edges)}")
        
        # 获取可用模版
        usable_templates = self.matcher.get_usable_templates(self.template_loader.templates)
        if not usable_templates:
            logger.error("没有可用的模版")
            return []
        
        logger.info(f"可用模版数量: {len(usable_templates)}")
        
        self.results = []
        
        # 实时输出文件处理
        realtime_file = None
        realtime_count = 0
        if realtime_output_path:
            realtime_file = open(realtime_output_path, 'w', encoding='utf-8')
            realtime_file.write('[\n')  # 开始 JSON 数组
            logger.info(f"启用实时输出到文件: {realtime_output_path}")
        
        try:
            # 1. 遍历噪声节点，尝试匹配模板
            logger.info("开始处理噪声节点...")
            for node_str in list(self.noise_loader.noisy_node_set):
                # 只有当 target 不为 None 时才检查数量限制
                if target is not None and len(self.results) >= target:
                    break
                
                if ':' not in node_str:
                    continue
                
                label, node_id = node_str.split(':', 1)
                
                # 过滤掉被排除的标签
                if self.builder.excluded_labels and label in self.builder.excluded_labels:
                    continue
                
                # 尝试匹配所有模板
                for template in usable_templates:
                    # 只有当 target 不为 None 时才检查数量限制
                    if target is not None and len(self.results) >= target:
                        break
                    
                    # 使用噪声节点构建查询
                    query, params_used = self.builder.build_query_from_noise_node(template, label, node_id)
                    if not query:
                        continue
                    
                    # 执行查询并验证
                    success, answer, error = self.executor.execute(query)
                    
                    if not success:
                        continue
                    
                    # 过滤查询结果
                    if not answer:
                        continue
                    
                    # 排除所有值都是 null 的记录
                    filtered_answer = []
                    for record in answer:
                        has_non_null = any(value is not None for value in record.values())
                        if has_non_null:
                            filtered_answer.append(record)
                    
                    if not filtered_answer:
                        continue
                    
                    # 检查是否所有 count 类型的结果都为 0
                    count_fields = ['cnt', 'count', 'total', 'num', 'number']
                    all_counts_zero = True
                    has_count_field = False
                    
                    for record in filtered_answer:
                        for field in count_fields:
                            if field in record:
                                has_count_field = True
                                if record[field] is not None and record[field] != 0:
                                    all_counts_zero = False
                                    break
                        if not all_counts_zero:
                            break
                    
                    if has_count_field and all_counts_zero:
                        continue
                    
                    # 如果答案数量超过阈值，跳过
                    if len(filtered_answer) > max_answer_count:
                        continue
                    
                    # 结果通过所有过滤条件，添加到结果列表
                    result = QueryResult(
                        template_id=template.id,
                        template_type=template.type,
                        query=query,
                        parameters_used=params_used,
                        answer=filtered_answer,
                        success=True,
                        error_message=None,
                        is_noise_query=True  # 标记为噪声查询
                    )
                    
                    self.results.append(result)
                    
                    # 实时写入文件
                    if realtime_file:
                        template_type = result.template_type or "unknown"
                        template_id_with_prefix = f"{template_type}_{result.template_id}"
                        output_data = {
                            "template_id": template_id_with_prefix,
                            "template_type": result.template_type,
                            "query": result.query,
                            "parameters": result.parameters_used,
                            "answer": result.answer,
                            "is_noise_query": result.is_noise_query
                        }
                        if realtime_count > 0:
                            realtime_file.write(',\n')
                        json.dump(output_data, realtime_file, ensure_ascii=False, indent=2)
                        realtime_file.flush()
                        realtime_count += 1
                    
                    logger.debug(f"成功生成查询: 模板={template.id}, 噪声节点={node_str}, 结果数={len(filtered_answer)}")
            
            # 2. 遍历噪声边，尝试匹配模板
            logger.info("开始处理噪声边...")
            for edge in self.noise_loader.noisy_edges:
                # 只有当 target 不为 None 时才检查数量限制
                if target is not None and len(self.results) >= target:
                    break
                
                if len(edge) < 2:
                    continue
                
                start_node_str = edge[0]
                end_node_str = edge[1]
                
                # 解析节点标签，检查是否被排除
                start_label, _ = self.builder._parse_node_string(start_node_str)
                end_label, _ = self.builder._parse_node_string(end_node_str)
                
                if not start_label or not end_label:
                    continue
                
                if self.builder.excluded_labels:
                    if start_label in self.builder.excluded_labels or end_label in self.builder.excluded_labels:
                        continue
                
                # 尝试匹配所有模板
                for template in usable_templates:
                    # 只有当 target 不为 None 时才检查数量限制
                    if target is not None and len(self.results) >= target:
                        break
                    
                    # 使用噪声边构建查询
                    query, params_used = self.builder.build_query_from_noise_edge(template, start_node_str, end_node_str)
                    if not query:
                        continue
                    
                    # 执行查询并验证
                    success, answer, error = self.executor.execute(query)
                    
                    if not success:
                        continue
                    
                    # 过滤查询结果（与上面相同的逻辑）
                    if not answer:
                        continue
                    
                    filtered_answer = []
                    for record in answer:
                        has_non_null = any(value is not None for value in record.values())
                        if has_non_null:
                            filtered_answer.append(record)
                    
                    if not filtered_answer:
                        continue
                    
                    # 检查 count 字段
                    count_fields = ['cnt', 'count', 'total', 'num', 'number']
                    all_counts_zero = True
                    has_count_field = False
                    
                    for record in filtered_answer:
                        for field in count_fields:
                            if field in record:
                                has_count_field = True
                                if record[field] is not None and record[field] != 0:
                                    all_counts_zero = False
                                    break
                        if not all_counts_zero:
                            break
                    
                    if has_count_field and all_counts_zero:
                        continue
                    
                    # 如果答案数量超过阈值，跳过
                    if len(filtered_answer) > max_answer_count:
                        continue
                    
                    # 结果通过所有过滤条件，添加到结果列表
                    result = QueryResult(
                        template_id=template.id,
                        template_type=template.type,
                        query=query,
                        parameters_used=params_used,
                        answer=filtered_answer,
                        success=True,
                        error_message=None,
                        is_noise_query=True  # 标记为噪声查询
                    )
                    
                    self.results.append(result)
                    
                    # 实时写入文件
                    if realtime_file:
                        template_type = result.template_type or "unknown"
                        template_id_with_prefix = f"{template_type}_{result.template_id}"
                        output_data = {
                            "template_id": template_id_with_prefix,
                            "template_type": result.template_type,
                            "query": result.query,
                            "parameters": result.parameters_used,
                            "answer": result.answer,
                            "is_noise_query": result.is_noise_query
                        }
                        if realtime_count > 0:
                            realtime_file.write(',\n')
                        json.dump(output_data, realtime_file, ensure_ascii=False, indent=2)
                        realtime_file.flush()
                        realtime_count += 1
                    
                    logger.debug(f"成功生成查询: 模板={template.id}, 噪声边={start_node_str}->{end_node_str}, 结果数={len(filtered_answer)}")
            
            # 统计噪声查询数量
            noise_query_count = len(self.results)
            clean_query_target = max(1, noise_query_count // 4)  # 干净查询目标是噪声查询的1/4
            
            logger.info(f"噪声查询生成完成，共生成 {noise_query_count} 个噪声查询")
            logger.info(f"开始生成干净查询，目标数量: {clean_query_target}")
            
            # 3. 生成干净查询
            clean_query_count = 0
            max_clean_attempts = clean_query_target * 20  # 最多尝试次数
            clean_attempts = 0
            
            # 收集所有标签（用于采样干净节点）
            all_labels = list(self.schema.labels.keys())
            if self.builder.excluded_labels:
                all_labels = [lb for lb in all_labels if lb not in self.builder.excluded_labels]
            
            # 收集所有关系类型（用于采样干净边）
            all_rel_types = list(self.schema.relationships.keys())
            
            while clean_query_count < clean_query_target and clean_attempts < max_clean_attempts:
                clean_attempts += 1
                
                # 随机选择模板
                template = random.choice(usable_templates)
                
                # 随机决定使用节点还是边
                use_node = random.random() < 0.7  # 70%概率使用节点，30%概率使用边
                
                query = None
                params_used = {}
                
                if use_node and all_labels:
                    # 尝试使用干净节点
                    label = random.choice(all_labels)
                    clean_nodes = self._sample_clean_nodes(label, sample_size=10)
                    
                    if clean_nodes:
                        clean_label, clean_node_id = random.choice(clean_nodes)
                        query, params_used = self.clean_builder.build_query_from_clean_node(
                            template, clean_label, clean_node_id
                        )
                else:
                    # 尝试使用干净边
                    if all_rel_types:
                        rel_type = random.choice(all_rel_types)
                        clean_edges = self._sample_clean_edges(rel_type, sample_size=10)
                        
                        if clean_edges:
                            start_node_str, end_node_str = random.choice(clean_edges)
                            query, params_used = self.clean_builder.build_query_from_clean_edge(
                                template, start_node_str, end_node_str
                            )
                
                if not query:
                    continue
                
                # 执行查询并验证
                success, answer, error = self.executor.execute(query)
                
                if not success:
                    continue
                
                # 过滤查询结果（与噪声查询相同的逻辑）
                if not answer:
                    continue
                
                filtered_answer = []
                for record in answer:
                    has_non_null = any(value is not None for value in record.values())
                    if has_non_null:
                        filtered_answer.append(record)
                
                if not filtered_answer:
                    continue
                
                # 检查 count 字段
                count_fields = ['cnt', 'count', 'total', 'num', 'number']
                all_counts_zero = True
                has_count_field = False
                
                for record in filtered_answer:
                    for field in count_fields:
                        if field in record:
                            has_count_field = True
                            if record[field] is not None and record[field] != 0:
                                all_counts_zero = False
                                break
                    if not all_counts_zero:
                        break
                
                if has_count_field and all_counts_zero:
                    continue
                
                # 如果答案数量超过阈值，跳过
                if len(filtered_answer) > max_answer_count:
                    continue
                
                # 结果通过所有过滤条件，添加到结果列表
                result = QueryResult(
                    template_id=template.id,
                    template_type=template.type,
                    query=query,
                    parameters_used=params_used,
                    answer=filtered_answer,
                    success=True,
                    error_message=None,
                    is_noise_query=False  # 标记为干净查询
                )
                
                self.results.append(result)
                clean_query_count += 1
                
                # 实时写入文件
                if realtime_file:
                    template_type = result.template_type or "unknown"
                    template_id_with_prefix = f"{template_type}_{result.template_id}"
                    output_data = {
                        "template_id": template_id_with_prefix,
                        "template_type": result.template_type,
                        "query": result.query,
                        "parameters": result.parameters_used,
                        "answer": result.answer,
                        "is_noise_query": result.is_noise_query
                    }
                    if realtime_count > 0:
                        realtime_file.write(',\n')
                    json.dump(output_data, realtime_file, ensure_ascii=False, indent=2)
                    realtime_file.flush()
                    realtime_count += 1
                
                logger.debug(f"成功生成干净查询: 模板={template.id}, 结果数={len(filtered_answer)}")
            
            logger.info(f"生成完成，共生成 {len(self.results)} 个查询（噪声查询: {noise_query_count}, 干净查询: {clean_query_count}）")
            
        finally:
            # 关闭实时输出文件
            if realtime_file:
                realtime_file.write('\n]')  # 结束 JSON 数组
                realtime_file.close()
                logger.info(f"实时输出文件已关闭: {realtime_output_path}")
        
        return self.results
    
    def generate_judge_queries(self, template_file_path: str = None, max_unique_answers: int = 20) -> List[Dict[str, Any]]:
        """
        读取 template_judge1.json 中的模板，生成 template 和 anti_template 的查询，
        对比结果并找出独特答案（只存在于 template 中的和只存在于 anti_template 中的）
        确保生成的查询包含噪声节点或边
        
        Args:
            template_file_path: 模板文件路径，默认为 query_template/template_judge1.json
            max_unique_answers: 每个查询结果中最多返回的独特答案数量，默认 20
        
        Returns:
            包含每个模板的查询结果和独特答案的列表
        """
        if not self.schema:
            self.initialize()
        
        # 默认模板文件路径
        if template_file_path is None:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_file_path = os.path.join(current_dir, '../query_template/template_judge1.json')
        
        logger.info(f"读取模板文件: {template_file_path}")
        
        # 读取模板文件
        try:
            with open(template_file_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
        except Exception as e:
            logger.error(f"读取模板文件失败: {e}")
            return []
        
        results = []
        
        # 遍历所有模板类型
        for type_group in template_data:
            template_type = type_group.get('type', 'unknown')
            templates = type_group.get('templates', [])
            
            # 遍历每个模板
            for template_info in templates:
                template_id = template_info.get('id', 'unknown')
                template_str = template_info.get('template', '')
                anti_template_str = template_info.get('anti_template', '')
                parameters = template_info.get('parameters', {})
                
                if not template_str or not anti_template_str:
                    logger.warning(f"模板 {template_id} 缺少 template 或 anti_template，跳过")
                    continue
                
                logger.info(f"处理模板: {template_type}/{template_id}")
                
                # 创建 Template 对象用于构建查询
                template_obj = Template(
                    id=template_id,
                    template=template_str,
                    parameters=parameters,
                    type=template_type
                )
                
                # 尝试生成查询（可能需要多次尝试才能成功，并确保包含噪声节点或边）
                max_attempts = 200  # 增加尝试次数，因为需要确保包含噪声
                template_query = None
                anti_template_query = None
                params_used = None
                
                # 策略1: 优先尝试使用噪声节点构建查询
                # 查找需要 LABEL 参数的模板
                label_params = [p for p in template_obj.parameters.keys() 
                               if p.startswith('LABEL') or p in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL')]
                
                if label_params and self.noise_loader.noisy_nodes:
                    # 尝试使用噪声节点构建查询
                    noisy_labels = list(self.noise_loader.noisy_nodes.keys())
                    # 过滤掉被排除的标签（从 builder 中获取）
                    if hasattr(self.builder, 'excluded_labels') and self.builder.excluded_labels:
                        noisy_labels = [lb for lb in noisy_labels if lb not in self.builder.excluded_labels]
                    
                    # 随机选择标签和节点，而不是固定使用前N个，这样可以每次生成不同的查询
                    max_label_attempts = min(10, len(noisy_labels))
                    selected_labels = random.sample(noisy_labels, min(max_label_attempts, len(noisy_labels))) if noisy_labels else []
                    
                    for noisy_label in selected_labels:
                        if params_used:
                            break
                        noisy_node_ids = self.noise_loader.noisy_nodes[noisy_label]
                        # 随机选择节点，而不是固定使用前5个
                        max_node_attempts = min(5, len(noisy_node_ids))
                        if max_node_attempts > 0:
                            selected_nodes = random.sample(noisy_node_ids, max_node_attempts)
                            for node_id in selected_nodes:
                                try:
                                    query, params = self.builder.build_query_from_noise_node(
                                        template_obj, noisy_label, node_id
                                    )
                                    if query:
                                        template_query = query
                                        params_used = params
                                        break
                                except Exception as e:
                                    logger.debug(f"使用噪声节点构建查询失败: {e}")
                                    continue
                
                # 策略2: 如果策略1失败，尝试使用噪声边构建查询
                if not template_query and self.noise_loader.noisy_edges:
                    # 随机选择边，而不是固定使用前20条，这样可以每次生成不同的查询
                    max_edge_attempts = min(20, len(self.noise_loader.noisy_edges))
                    if max_edge_attempts > 0:
                        selected_edges = random.sample(self.noise_loader.noisy_edges, max_edge_attempts)
                        for edge in selected_edges:
                            if template_query:
                                break
                            try:
                                query, params = self.builder.build_query_from_noise_edge(
                                    template_obj, edge[0], edge[1]
                                )
                                if query:
                                    template_query = query
                                    params_used = params
                                    break
                            except Exception as e:
                                logger.debug(f"使用噪声边构建查询失败: {e}")
                                continue
                
                # 策略3: 如果前两种策略都失败，使用普通构建方法，但验证是否包含噪声
                if not template_query:
                    for attempt in range(max_attempts):
                        try:
                            # 使用 NoiseQueryBuilder 构建 template 查询
                            query, params = self.builder.build_query(template_obj)
                            if query:
                                # 验证查询是否包含噪声节点或边
                                if self._query_contains_noise(params):
                                    template_query = query
                                    params_used = params
                                    break
                        except Exception as e:
                            logger.debug(f"尝试 {attempt + 1}/{max_attempts} 构建查询失败: {e}")
                            continue
                
                # 如果成功构建了 template 查询，构建 anti_template 查询
                if template_query and params_used:
                    try:
                        # 构建 anti_template 查询（使用相同的参数）
                        anti_query = anti_template_str
                        replacements = {}
                        
                        for param_name, value in params_used.items():
                            # 替换模版中的参数
                            if param_name == 'VALUE' or param_name in ('VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'V', 'AID', 'BID', 'CID', 'ID', 'ID1', 'ID2'):
                                if isinstance(value, str):
                                    replacement = f"'{value}'"
                                else:
                                    replacement = str(value)
                            else:
                                replacement = str(value)
                            
                            replacements[f'${param_name}'] = replacement
                        
                        # 一次性替换所有参数（按照键长度倒序，避免短键被先替换导致长键匹配失败）
                        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
                        for key in sorted_keys:
                            anti_query = anti_query.replace(key, replacements[key])
                        
                        # 对于 dataset 为 "mcp" 或 "multi_fin" 的情况，添加 concept id 过滤
                        if self.dataset in ("mcp", "multi_fin"):
                            anti_query = self.builder._add_concept_id_filter(anti_query, params_used)
                        
                        anti_template_query = anti_query
                    except Exception as e:
                        logger.warning(f"构建 anti_template 查询失败: {e}")
                        template_query = None  # 重置，表示失败
                
                if not template_query or not anti_template_query:
                    logger.warning(f"模板 {template_id} 无法生成包含噪声的有效查询，跳过")
                    continue
                
                # 执行 template 查询
                logger.info(f"执行 template 查询: {template_id}")
                template_success, template_results, template_error = self.executor.execute(template_query)
                
                # 执行 anti_template 查询
                logger.info(f"执行 anti_template 查询: {template_id}")
                anti_template_success, anti_template_results, anti_template_error = self.executor.execute(anti_template_query)
                
                if not template_success:
                    logger.warning(f"Template 查询执行失败: {template_error}")
                    continue
                
                if not anti_template_success:
                    logger.warning(f"Anti-template 查询执行失败: {anti_template_error}")
                    continue
                
                # 将结果转换为可比较的格式（转换为字符串集合以便比较）
                def normalize_result(result: Dict) -> str:
                    """将结果字典转换为可比较的字符串"""
                    # 对字典按键排序，然后转换为 JSON 字符串
                    if isinstance(result, dict):
                        sorted_dict = dict(sorted(result.items()))
                        return json.dumps(sorted_dict, sort_keys=True, ensure_ascii=False)
                    else:
                        return json.dumps(result, sort_keys=True, ensure_ascii=False)
                
                # 转换为集合以便比较
                template_result_set = {normalize_result(r) for r in template_results}
                anti_template_result_set = {normalize_result(r) for r in anti_template_results}
                
                # 找出独特答案
                # 只存在于 template 中的答案
                unique_in_template = template_result_set - anti_template_result_set
                # 只存在于 anti_template 中的答案
                unique_in_anti_template = anti_template_result_set - template_result_set
                
                # 将字符串转换回字典
                def parse_result(s: str) -> Dict:
                    """将 JSON 字符串转换回字典"""
                    try:
                        return json.loads(s)
                    except:
                        return {"raw": s}
                
                # 获取最多 max_unique_answers 个独特答案
                unique_template_answers = [parse_result(s) for s in list(unique_in_template)[:max_unique_answers]]
                unique_anti_template_answers = [parse_result(s) for s in list(unique_in_anti_template)[:max_unique_answers]]
                
                # 构建结果
                result_item = {
                    "template_id": template_id,
                    "template_type": template_type,
                    "template_query": template_query,
                    "anti_template_query": anti_template_query,
                    "parameters_used": params_used,
                    "contains_noise": True,  # 标记为包含噪声
                    "template_results_count": len(template_results),
                    "anti_template_results_count": len(anti_template_results),
                    "unique_in_template_count": len(unique_in_template),
                    "unique_in_anti_template_count": len(unique_in_anti_template),
                    "unique_in_template_answers": unique_template_answers,
                    "unique_in_anti_template_answers": unique_anti_template_answers,
                    "all_template_results": template_results[:max_unique_answers * 2],  # 保存部分原始结果用于调试
                    "all_anti_template_results": anti_template_results[:max_unique_answers * 2]
                }
                
                results.append(result_item)
                logger.info(f"模板 {template_id} 处理完成: template 独特答案 {len(unique_template_answers)} 个, "
                           f"anti_template 独特答案 {len(unique_anti_template_answers)} 个")
        
        logger.info(f"总共处理了 {len(results)} 个模板")
        return results
    
    def _query_contains_noise(self, params_used: Dict[str, Any]) -> bool:
        """
        检查查询参数是否包含噪声节点或边
        
        Args:
            params_used: 查询使用的参数字典
        
        Returns:
            如果查询包含噪声节点或边，返回True
        """
        # 检查 VALUE 参数是否对应噪声节点
        value_params = ['VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'END_VALUE']
        label_params = ['LABEL', 'LABEL1', 'LABEL2', 'LABEL3', 'LABEL4', 'START_LABEL', 'END_LABEL']
        
        # 检查节点参数
        for value_param in value_params:
            if value_param in params_used:
                value = params_used[value_param]
                # 找到对应的 LABEL 参数
                for label_param in label_params:
                    if label_param in params_used:
                        label = params_used[label_param]
                        # 检查是否是噪声节点
                        if self.noise_loader.is_noisy_node(label, str(value)):
                            return True
        
        # 检查边参数（通过 START_VALUE 和 END_VALUE）
        if 'START_VALUE' in params_used and 'END_VALUE' in params_used:
            start_label = params_used.get('START_LABEL') or params_used.get('LABEL1')
            end_label = params_used.get('END_LABEL') or params_used.get('LABEL2')
            
            if start_label and end_label:
                start_node_str = f"{start_label}:{params_used['START_VALUE']}"
                end_node_str = f"{end_label}:{params_used['END_VALUE']}"
                
                # 检查是否是噪声边
                if self.noise_loader.is_noisy_edge(start_node_str, end_node_str):
                    return True
                
                # 或者检查起点或终点是否是噪声节点
                if self.noise_loader.is_noisy_node(start_label, str(params_used['START_VALUE'])) or \
                   self.noise_loader.is_noisy_node(end_label, str(params_used['END_VALUE'])):
                    return True
        
        # 如果以上都没有找到，尝试通过 LABEL 参数检查是否有噪声节点
        for label_param in label_params:
            if label_param in params_used:
                label = params_used[label_param]
                # 如果该标签有噪声节点，且 VALUE 参数使用了噪声节点的ID
                if label in self.noise_loader.noisy_nodes:
                    # 检查对应的 VALUE 参数
                    for value_param in value_params:
                        if value_param in params_used:
                            value = str(params_used[value_param])
                            if value in self.noise_loader.noisy_nodes[label]:
                                return True
        
        return False
