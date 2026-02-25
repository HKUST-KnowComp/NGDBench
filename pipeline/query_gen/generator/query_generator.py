"""
Neo4j Query Generator with Template-based Sampling
"""

import json
import random
import re
import time
import signal
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship
import logging
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认不作为返回属性的内部字段，用户可按需修改
DEFAULT_EXCLUDED_RETURN_PROPS: Set[str] = {"_node_id","file_id"}

# 默认在采样 label 参数时排除的“技术性标签”（例如统一基础标签）
DEFAULT_EXCLUDED_LABELS: Set[str] = {"NGDBNode"}

class PropertyType(Enum):
    """属性类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class PropertyInfo:
    """属性信息"""
    name: str
    prop_type: PropertyType
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class LabelInfo:
    """Label信息"""
    name: str
    properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    node_count: int = 0


@dataclass
class RelationshipInfo:
    """关系类型信息"""
    rel_type: str
    start_labels: Set[str] = field(default_factory=set)
    end_labels: Set[str] = field(default_factory=set)
    count: int = 0
    properties: Dict[str, PropertyInfo] = field(default_factory=dict)  # 关系属性


@dataclass
class Template:
    """模版类"""
    id: str
    template: str
    parameters: Dict[str, str]
    required_numeric_props: bool = False  # 是否需要数值属性
    type: Optional[str] = None  # 模版类型（如 "basic", "nested_loop" 等）


@dataclass
class QueryResult:
    """查询结果"""
    template_id: str
    template_type: Optional[str] = None  # 模版类型
    query: str = ""
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    answer: List[Dict] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    is_noise_query: Optional[bool] = None  # 是否为噪声查询，None表示未标记


class SchemaAnalyzer:
    """数据库Schema分析器"""
    
    def __init__(self, driver):
        self.driver = driver
        self.labels: Dict[str, LabelInfo] = {}
        self.relationships: Dict[str, RelationshipInfo] = {}
        self.triplets: Set[Tuple[str, str, str]] = set()  # (start_label, rel_type, end_label)
        self.total_nodes = 0
        self.total_edges = 0
    
    def analyze(self, sample_size: int = 100):
        """分析数据库schema"""
        logger.info("开始分析数据库Schema...")
        
        with self.driver.session() as session:
            # 获取所有labels
            self._analyze_labels(session)
            
            # 获取每个label的属性信息
            for label in self.labels:
                self._analyze_label_properties(session, label, sample_size)
            
            # 获取关系类型信息
            self._analyze_relationships(session, sample_size)
            
            # 获取统计信息
            self._get_statistics(session)
        
        logger.info(f"Schema分析完成: {len(self.labels)} labels, "
                   f"{len(self.relationships)} relationship types, "
                   f"{self.total_nodes} nodes, {self.total_edges} edges")
    
    def _analyze_labels(self, session):
        """获取所有labels"""
        result = session.run("CALL db.labels()")
        for record in result:
            label = record[0]
            self.labels[label] = LabelInfo(name=label)
    
    def _analyze_label_properties(self, session, label: str, sample_size: int):
        """分析特定label的属性"""
        # 采样节点获取属性
        query = f"MATCH (n:`{label}`) RETURN n LIMIT {sample_size}"
        result = session.run(query)
        
        properties_data: Dict[str, List[Any]] = {}
        count = 0
        
        for record in result:
            count += 1
            node = record["n"]
            for key, value in dict(node).items():
                if key not in properties_data:
                    properties_data[key] = []
                properties_data[key].append(value)
        
        self.labels[label].node_count = count
        
        # 推断属性类型
        for prop_name, values in properties_data.items():
            prop_type = self._infer_property_type(values)
            # 保留一些样本值用于生成查询
            sample_values = random.sample(values, min(10, len(values))) if values else []
            self.labels[label].properties[prop_name] = PropertyInfo(
                name=prop_name,
                prop_type=prop_type,
                sample_values=sample_values
            )
    
    def _infer_property_type(self, values: List[Any]) -> PropertyType:
        """推断属性类型"""
        non_none_values = [v for v in values if v is not None]
        if not non_none_values:
            return PropertyType.UNKNOWN
        
        # 检查第一个非空值的类型
        sample = non_none_values[0]
        if isinstance(sample, bool):
            return PropertyType.BOOLEAN
        elif isinstance(sample, int):
            return PropertyType.INTEGER
        elif isinstance(sample, float):
            return PropertyType.FLOAT
        elif isinstance(sample, str):
            return PropertyType.STRING
        else:
            return PropertyType.UNKNOWN
    
    def _analyze_relationships(self, session, sample_size: int = 100):
        """分析关系类型"""
        # 获取所有关系类型
        result = session.run("CALL db.relationshipTypes()")
        for record in result:
            rel_type = record[0]
            self.relationships[rel_type] = RelationshipInfo(rel_type=rel_type)
        
        # 获取每种关系的起始和结束label以及属性（采样）
        for rel_type in self.relationships:
            query = f"""
            MATCH (a)-[r:`{rel_type}`]->(b)
            RETURN labels(a) as start_labels, labels(b) as end_labels, r
            LIMIT {sample_size}
            """
            result = session.run(query)
            
            properties_data: Dict[str, List[Any]] = {}
            
            for record in result:
                start_labels = record["start_labels"]
                end_labels = record["end_labels"]
                
                for s_label in start_labels:
                    self.relationships[rel_type].start_labels.add(s_label)
                for e_label in end_labels:
                    self.relationships[rel_type].end_labels.add(e_label)
                
                # 记录三元组
                for s_label in start_labels:
                    for e_label in end_labels:
                        self.triplets.add((s_label, rel_type, e_label))
                
                # 分析关系属性
                rel = record["r"]
                if rel:
                    for key, value in dict(rel).items():
                        if key not in properties_data:
                            properties_data[key] = []
                        properties_data[key].append(value)
            
            # 推断关系属性类型并存储
            for prop_name, values in properties_data.items():
                prop_type = self._infer_property_type(values)
                sample_values = random.sample(values, min(10, len(values))) if values else []
                self.relationships[rel_type].properties[prop_name] = PropertyInfo(
                    name=prop_name,
                    prop_type=prop_type,
                    sample_values=sample_values
                )
    
    def _get_statistics(self, session):
        """获取节点和边的统计信息"""
        # 节点总数
        result = session.run("MATCH (n) RETURN count(n) as count")
        self.total_nodes = result.single()["count"]
        
        # 边总数
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        self.total_edges = result.single()["count"]
    
    def get_labels_with_property_type(self, prop_type: PropertyType) -> List[Tuple[str, str]]:
        """获取具有特定类型属性的label和属性名对"""
        results = []
        for label, info in self.labels.items():
            for prop_name, prop_info in info.properties.items():
                if prop_info.prop_type == prop_type:
                    results.append((label, prop_name))
        return results
    
    def get_numeric_properties(self, label: str) -> List[PropertyInfo]:
        """获取特定label的数值属性"""
        if label not in self.labels:
            return []
        return [
            prop for prop in self.labels[label].properties.values()
            if prop.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT)
        ]
    
    def get_string_properties(self, label: str) -> List[PropertyInfo]:
        """获取特定label的字符串属性"""
        if label not in self.labels:
            return []
        return [
            prop for prop in self.labels[label].properties.values()
            if prop.prop_type == PropertyType.STRING
        ]


class TemplateLoader:
    """模版加载器"""
    
    NUMERIC_OPERATORS = {'>', '<', '>=', '<=', '=', '<>'}
    STRING_OPERATORS = {'=', '<>', 'CONTAINS', 'STARTS WITH', 'ENDS WITH'}
    
    def __init__(self, template_path: str):
        self.template_path = template_path
        self.templates: List[Template] = []
    
    def load(self):
        """加载模版文件"""
        logger.info(f"加载模版文件: {self.template_path}")
        
        with open(self.template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持多种格式：
        # 1. 数组，每个元素包含 type 和 templates 字段
        # 2. 数组，每个元素直接是模板对象
        # 3. 对象，包含 templates 字段
        # 4. 单个模板对象
        templates_data = []
        
        if isinstance(data, list):
            # 检查数组中的元素是否包含 templates 字段（嵌套结构）
            for item in data:
                if isinstance(item, dict) and 'templates' in item:
                    # 嵌套结构：每个元素包含 type 和 templates
                    template_type = item.get('type', None)
                    # 为每个模板添加type信息
                    for t in item['templates']:
                        t_with_type = t.copy()
                        t_with_type['_template_type'] = template_type
                        templates_data.append(t_with_type)
                else:
                    # 扁平结构：每个元素直接是模板
                    templates_data.append(item)
        elif isinstance(data, dict) and 'templates' in data:
            templates_data = data['templates']
        else:
            templates_data = [data]
        
        for t in templates_data:
            template = Template(
                id=t['id'],
                template=t['template'],
                parameters=t['parameters'],
                required_numeric_props=self._requires_numeric_props(t['template']),
                type=t.get('_template_type') or t.get('type', None)
            )
            self.templates.append(template)
        
        logger.info(f"加载了 {len(self.templates)} 个模版")
    
    def _requires_numeric_props(self, template: str) -> bool:
        """检查模版是否需要数值属性（包含数值比较操作符）"""
        for op in self.NUMERIC_OPERATORS:
            if f' {op} ' in template or f'${op}' in template:
                return True
        return False
    
    def get_template_by_id(self, template_id: str) -> Optional[Template]:
        """根据ID获取模版"""
        for t in self.templates:
            if t.id == template_id:
                return t
        return None


class TemplateMatcher:
    """模版匹配器：检查模版是否适用于当前schema"""
    
    def __init__(self, schema: SchemaAnalyzer):
        self.schema = schema
    
    def can_use_template(self, template: Template) -> bool:
        """检查模版是否可以在当前schema下使用"""
        params = template.parameters
        
        # 检查是否需要LABEL
        if 'LABEL' in params or 'LABEL1' in params:
            if not self.schema.labels:
                return False
        
        # 检查是否需要关系类型
        if 'REL_TYPE' in params:
            if not self.schema.relationships:
                return False
        
        # 如果模版需要数值属性，检查是否有可用的
        if template.required_numeric_props:
            has_numeric = False
            for label_info in self.schema.labels.values():
                for prop in label_info.properties.values():
                    if prop.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                        has_numeric = True
                        break
                if has_numeric:
                    break
            if not has_numeric:
                return False
        
        return True
    
    def get_usable_templates(self, templates: List[Template]) -> List[Template]:
        """获取所有可用的模版"""
        return [t for t in templates if self.can_use_template(t)]


class QueryBuilder:
    """查询构建器：根据模版和schema生成具体查询"""
    
    NUMERIC_OPERATORS = ['>', '<', '>=', '<=', '=', '<>']
    STRING_OPERATORS = ['=', '<>', 'CONTAINS', 'STARTS WITH', 'ENDS WITH']
    
    def __init__(
        self,
        schema: SchemaAnalyzer,
        excluded_return_props: Optional[Set[str]] = None,
        excluded_labels: Optional[Set[str]] = None,
        dataset: Optional[str] = None,
        driver = None,
    ):
        self.schema = schema
        # 不作为返回属性的字段（例如内部ID）
        self.excluded_return_props: Set[str] = excluded_return_props or set()
        # 不希望作为 LABEL 参数出现的标签（例如 NGDBNode 这种基础标签）
        self.excluded_labels: Set[str] = excluded_labels or set()
        # 数据集名称，用于特殊处理（如为 concept 节点添加 id 过滤）
        self.dataset = dataset
        # 数据库驱动，用于采样节点
        self.driver = driver
        # 各 label 的节点总数缓存，用于随机 SKIP 以分散采样区域、减少重复查询
        self._label_count_cache: Dict[str, int] = {}
    
    def _get_label_count(self, label: str) -> int:
        """获取指定 label 的节点总数（带缓存），用于随机 SKIP 分散采样"""
        if label in self._label_count_cache:
            return self._label_count_cache[label]
        if not self.driver:
            return 0
        try:
            with self.driver.session() as session:
                query = f"MATCH (n:`{label}`) RETURN count(n) AS c"
                result = session.run(query)
                record = result.single()
                count = record["c"] if record else 0
                self._label_count_cache[label] = count
                return count
        except Exception as e:
            logger.debug(f"获取 label 节点数失败 (label={label}): {e}")
            return 0
    
    def build_query(self, template: Template) -> Tuple[Optional[str], Dict[str, Any]]:
        """根据模版构建查询"""
        params_used = {}
        query = template.template
        
        try:
            # 首先尝试从节点采样来填充相关参数
            # 识别需要从同一节点获取的参数组
            node_param_groups = self._identify_node_param_groups(template)
            
            # 为每个节点参数组采样节点并填充参数
            for group in node_param_groups:
                if not self._fill_params_from_sampled_node(template, group, params_used):
                    # 如果采样失败，回退到原来的方法
                    logger.debug(f"节点采样失败，回退到原方法: {group}")
            
            # 解析并填充剩余参数
            replacements = {}
            for param_name, param_type in template.parameters.items():
                # 如果参数已经填充，跳过
                if param_name in params_used:
                    value = params_used[param_name]
                else:
                    value = self._generate_param_value(param_name, template, params_used)
                    if value is None:
                        return None, {}
                    params_used[param_name] = value
                
                # 替换模版中的参数
                # 对于字符串值需要加引号
                if param_name == 'VALUE' or param_name in ('VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'V', 'AID', 'BID', 'CID', 'ID', 'ID1', 'ID2', ):
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
            
            return query, params_used
            
        except Exception as e:
            logger.warning(f"构建查询失败: {e}")
            return None, {}
    
    def _add_concept_id_filter(self, query: str, params_used: Dict[str, Any]) -> str:
        """为查询中的 concept 节点添加 id 属性过滤"""
        import re
        
        # 检查查询中是否使用了 concept 标签
        has_concept = False
        for param_name, value in params_used.items():
            if (param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'REL_NODE_LABEL')):
                if value == "concept":
                    has_concept = True
                    break
        
        if not has_concept:
            return query
        
        # 获取一个 concept 节点的 id 样本值
        concept_id_value = self._get_concept_id_sample()
        if not concept_id_value:
            # 如果没有找到 id 样本值，返回原查询
            return query
        
        # 在查询中查找所有 `:concept` 的出现，并添加 id 过滤
        # 使用更精确的匹配：查找 `:concept` 后面不是 `{` 的情况，然后添加 `{id: "xxx"}`
        # 同时处理已有属性的情况：`:concept {` -> `:concept {id: "xxx", `
        
        # 先处理已有属性的情况：`:concept {` -> `:concept {id: "xxx", `
        pattern1 = r':concept\s*\{'
        def replace_with_existing_props(match):
            return f':concept {{id: "{concept_id_value}", '
        query = re.sub(pattern1, replace_with_existing_props, query)
        
        # 再处理没有属性的情况：`:concept` 后面不是 `{`，添加 `{id: "xxx"}`
        # 匹配 `:concept` 后面跟着 `)`、`-`、`]`、空格或行尾（正向前瞻确保不是 `{`）
        pattern2 = r':concept(?=\s*[\)\-\]]|\s|$)'
        def replace_without_props(match):
            return f':concept {{id: "{concept_id_value}"}}'
        query = re.sub(pattern2, replace_without_props, query)
        
        return query
    
    def _get_concept_id_sample(self) -> Optional[str]:
        """获取一个 concept 节点的 id 属性样本值"""
        if "concept" not in self.schema.labels:
            return None
        
        concept_label_info = self.schema.labels["concept"]
        if "id" not in concept_label_info.properties:
            return None
        
        id_prop_info = concept_label_info.properties["id"]
        if id_prop_info.sample_values:
            # 如果是 mcp 或 multi-fin 数据集，过滤掉以数字结尾的值
            if self.dataset in ("mcp", "multi_fin"):
                filtered_values = [v for v in id_prop_info.sample_values 
                                 if not (isinstance(v, str) and v and v[-1].isdigit())]
                if filtered_values:
                    return random.choice(filtered_values)
                return None
            else:
                return random.choice(id_prop_info.sample_values)
        
        return None
    
    def _is_aggregate_query(self, template: Template, current_params: Dict[str, Any]) -> bool:
        """判断是否为聚合查询"""
        # 方法1：检查模板类型
        if template.type and 'aggregation' in template.type.lower():
            return True
        
        # 方法2：检查模板字符串中是否包含聚合函数
        template_upper = template.template.upper()
        aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'COLLECT']
        if any(func in template_upper for func in aggregate_functions):
            return True
        
        # 方法3：检查参数中是否已经有聚合函数参数
        if any(key.startswith('AGG_FUNC') or key == 'AGG_FUNC' for key in current_params.keys()):
            return True
        
        return False
    
    def _is_id_property(self, prop_name: str) -> bool:
        """判断属性名是否为ID相关属性"""
        prop_lower = prop_name.lower()
        # 检查是否以 id 结尾（如 loanId, userId, user_id）
        if prop_lower.endswith('id') or prop_lower.endswith('_id'):
            return True
        # 检查是否就是 id 本身
        if prop_lower == 'id':
            return True
        return False
    
    def _identify_node_param_groups(self, template: Template) -> List[Dict[str, str]]:
        """
        识别模板中需要从同一节点获取的参数组
        
        返回参数组列表，每个组包含：
        - label_param: 标签参数名（如 LABEL1, L1）
        - prop_params: 属性参数名列表（如 [PROP1, PROP_ID1]）
        - value_params: 值参数名列表（如 [VALUE, AID, V]）
        
        例如：对于模板 template.json:640-651
        - LABEL1 -> PROP1 -> VALUE
        - LABEL2 -> (可能有其他属性)
        - LABEL3 -> FILTER_PROP -> FILTER_VAL
        """
        groups = []
        params = template.parameters
        
        # 定义参数名映射关系
        # 格式: {label_param: [(prop_param, value_param), ...]}
        # 例如: {"LABEL1": [("PROP1", "VALUE")], "L1": [("P", "V"), ("PROP_ID1", "AID")]}
        
        # 识别 LABEL1 -> PROP1 -> VALUE 模式
        if 'LABEL1' in params:
            prop_value_pairs = []
            # PROP1 对应 VALUE
            if 'PROP1' in params and 'VALUE' in params:
                prop_value_pairs.append(('PROP1', 'VALUE'))
            # PROP2 可能对应其他值参数，这里先不处理，避免冲突
            if prop_value_pairs:
                groups.append({
                    'label_param': 'LABEL1',
                    'prop_value_pairs': prop_value_pairs
                })
        
        # 识别 L1 -> P -> V 模式（用于 management 模板）
        if 'L1' in params:
            prop_value_pairs = []
            if 'P' in params and 'V' in params:
                prop_value_pairs.append(('P', 'V'))
            # PROP_ID1 可能对应 AID、ID1 或批量模板中的 ID1_1（首行）
            if 'PROP_ID1' in params:
                if 'AID' in params:
                    prop_value_pairs.append(('PROP_ID1', 'AID'))
                elif 'ID1' in params:
                    prop_value_pairs.append(('PROP_ID1', 'ID1'))
                elif 'ID1_1' in params:
                    prop_value_pairs.append(('PROP_ID1', 'ID1_1'))
            if prop_value_pairs:
                groups.append({
                    'label_param': 'L1',
                    'prop_value_pairs': prop_value_pairs
                })
        
        # 识别 LABEL -> PROP -> VALUE 模式
        if 'LABEL' in params and 'PROP' in params and 'VALUE' in params:
            groups.append({
                'label_param': 'LABEL',
                'prop_value_pairs': [('PROP', 'VALUE')]
            })
        
        # 识别 LABEL -> PROP_ID -> VALUE 模式
        if 'LABEL' in params and 'PROP_ID' in params and 'VALUE' in params:
            groups.append({
                'label_param': 'LABEL',
                'prop_value_pairs': [('PROP_ID', 'VALUE')]
            })
        
        # 识别 GROUP_LABEL -> PROP_ID -> GID 模式
        if 'GROUP_LABEL' in params and 'PROP_ID' in params and 'GID' in params:
            groups.append({
                'label_param': 'GROUP_LABEL',
                'prop_value_pairs': [('PROP_ID', 'GID')]
            })
        
        # 识别 L -> PROP_ID -> VALUE 模式
        if 'L' in params and 'PROP_ID' in params and 'VALUE' in params:
            groups.append({
                'label_param': 'L',
                'prop_value_pairs': [('PROP_ID', 'VALUE')]
            })
        
        # 识别 L2 -> PROP_ID -> MID 模式
        if 'L2' in params and 'PROP_ID' in params and 'MID' in params:
            groups.append({
                'label_param': 'L2',
                'prop_value_pairs': [('PROP_ID', 'MID')]
            })
        
        # 识别 L3 -> PROP_ID -> VALUE/CID 模式
        if 'L3' in params and 'PROP_ID' in params:
            prop_value_pairs = []
            if 'VALUE' in params:
                prop_value_pairs.append(('PROP_ID', 'VALUE'))
            elif 'CID' in params:
                prop_value_pairs.append(('PROP_ID', 'CID'))
            if prop_value_pairs:
                groups.append({
                    'label_param': 'L3',
                    'prop_value_pairs': prop_value_pairs
                })
        
        # 识别 L4 -> PROP_ID -> VALUE 模式
        if 'L4' in params and 'PROP_ID' in params and 'VALUE' in params:
            groups.append({
                'label_param': 'L4',
                'prop_value_pairs': [('PROP_ID', 'VALUE')]
            })
        
        # 识别 RL -> PROP_ID -> VALUE 模式（关系标签）
        if 'RL' in params and 'PROP_ID' in params and 'VALUE' in params:
            groups.append({
                'label_param': 'RL',
                'prop_value_pairs': [('PROP_ID', 'VALUE')]
            })
        
        # 识别 LABEL2 -> PROP_ID -> BID1 模式（用于 CREATE_4 等：批量创建每行不同目标节点）
        if 'LABEL2' in params and 'PROP_ID' in params and 'BID1' in params:
            groups.append({
                'label_param': 'LABEL2',
                'prop_value_pairs': [('PROP_ID', 'BID1')]
            })
        
        # 识别 LABEL3 -> FILTER_PROP -> FILTER_VAL 模式
        if 'LABEL3' in params and 'FILTER_PROP' in params and 'FILTER_VAL' in params:
            groups.append({
                'label_param': 'LABEL3',
                'prop_value_pairs': [('FILTER_PROP', 'FILTER_VAL')]
            })
        
        # 识别 L2 -> PROP_ID2 -> BID/ID2/BID1/ID2_1 模式（用于 management 模板）
        if 'L2' in params and 'PROP_ID2' in params:
            prop_value_pairs = []
            # PROP_ID2 可能对应 BID、ID2 或批量模板中的 BID1、ID2_1（首行）
            if 'BID' in params:
                prop_value_pairs.append(('PROP_ID2', 'BID'))
            elif 'ID2' in params:
                prop_value_pairs.append(('PROP_ID2', 'ID2'))
            elif 'BID1' in params:
                prop_value_pairs.append(('PROP_ID2', 'BID1'))
            elif 'ID2_1' in params:
                prop_value_pairs.append(('PROP_ID2', 'ID2_1'))
            # 注意：PROP_ID2 也可能与 IDS 列表一起使用（用于 WHERE ... IN $IDS），
            # 但这种情况不需要从节点采样值，因为 IDS 是列表类型
            if prop_value_pairs:
                groups.append({
                    'label_param': 'L2',
                    'prop_value_pairs': prop_value_pairs
                })
        
        # 识别 TARGET_LABEL -> TARGET_PROP_ID -> TARGET_ID 模式（用于 management/MIX 模板）
        if 'TARGET_LABEL' in params and 'TARGET_PROP_ID' in params and 'TARGET_ID' in params:
            groups.append({
                'label_param': 'TARGET_LABEL',
                'prop_value_pairs': [('TARGET_PROP_ID', 'TARGET_ID')]
            })
        
        # 识别 L1 -> TARGET_PROP_ID1 -> TARGET_ID1 / L2 -> TARGET_PROP_ID2 -> TARGET_ID2（用于 MIX_2 等多关系模板）
        if 'L1' in params and 'TARGET_PROP_ID1' in params and 'TARGET_ID1' in params:
            groups.append({
                'label_param': 'L1',
                'prop_value_pairs': [('TARGET_PROP_ID1', 'TARGET_ID1')]
            })
        if 'L2' in params and 'TARGET_PROP_ID2' in params and 'TARGET_ID2' in params:
            groups.append({
                'label_param': 'L2',
                'prop_value_pairs': [('TARGET_PROP_ID2', 'TARGET_ID2')]
            })
        
        # 识别硬编码标签的情况：当模板中有 PROP_ID 和 VALUE 但没有 LABEL 参数时
        # 从模板字符串中提取硬编码的标签（如 :entity）
        if 'PROP_ID' in params and 'VALUE' in params:
            # 检查是否已经有 LABEL 相关的参数组
            has_label_param = any('LABEL' in group.get('label_param', '') or 
                                 group.get('label_param', '') in ['L', 'L1', 'L2', 'L3', 'L4'] 
                                 for group in groups)
            
            if not has_label_param:
                # 从模板字符串中提取硬编码的标签
                # 匹配模式: (var:label {$PROP_ID: $VALUE}) 或 (:label {$PROP_ID: $VALUE})
                hardcoded_labels = []
                # 查找所有硬编码的标签，例如 :entity, :passage 等
                # 匹配 (var:label {$PROP_ID: $VALUE}) 或 (var:label {$PROP_ID:$VALUE})
                pattern = r':(\w+)\s*\{[^}]*\$PROP_ID[^}]*\$VALUE[^}]*\}'
                matches = re.findall(pattern, template.template)
                if matches:
                    hardcoded_labels.extend(matches)
                
                # 去重
                hardcoded_labels = list(set(hardcoded_labels))
                
                # 如果找到了硬编码的标签，创建一个参数组
                if hardcoded_labels:
                    # 使用第一个找到的硬编码标签
                    hardcoded_label = hardcoded_labels[0]
                    groups.append({
                        'label_param': hardcoded_label,  # 直接使用标签名，而不是参数名
                        'prop_value_pairs': [('PROP_ID', 'VALUE')],
                        'is_hardcoded': True  # 标记这是硬编码的标签
                    })
        
        return groups
    
    def _fill_params_from_sampled_node(self, template: Template, group: Dict[str, Any], 
                                       params_used: Dict[str, Any]) -> bool:
        """
        从采样的节点中填充参数组
        
        Args:
            template: 查询模板
            group: 参数组，包含 label_param 和 prop_value_pairs
            params_used: 已使用的参数字典（会被更新）
        
        Returns:
            是否成功填充
        """
        if not self.driver:
            return False
        
        label_param = group['label_param']
        prop_value_pairs = group['prop_value_pairs']
        is_hardcoded = group.get('is_hardcoded', False)
        
        # 如果是硬编码标签，label_param 直接就是标签名（如 "entity"）
        if is_hardcoded:
            label = label_param
            # 验证标签是否存在
            if label not in self.schema.labels:
                logger.warning(f"硬编码标签 {label} 不存在于schema中")
                return False
        # 如果标签参数已经填充，使用已填充的标签
        elif label_param in params_used:
            label = params_used[label_param]
        else:
            # 选择一个标签
            all_labels = list(self.schema.labels.keys())
            if self.excluded_labels:
                all_labels = [lb for lb in all_labels if lb not in self.excluded_labels]
            
            if not all_labels:
                return False
            
            # 优先选择有可用属性的标签
            # 如果需要数值属性，优先选择有数值属性的标签
            if template.required_numeric_props:
                labels_with_numeric_props = []
                for lb in all_labels:
                    if lb in self.schema.labels:
                        props = list(self.schema.labels[lb].properties.keys())
                        if self.excluded_return_props:
                            props = [p for p in props if p not in self.excluded_return_props]
                        # 检查是否有数值属性
                        has_numeric = False
                        for p in props:
                            prop_info = self.schema.labels[lb].properties.get(p)
                            if prop_info and prop_info.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                                has_numeric = True
                                break
                        if has_numeric:
                            labels_with_numeric_props.append(lb)
                
                if labels_with_numeric_props:
                    label = random.choice(labels_with_numeric_props)
                else:
                    # 如果没有标签有数值属性，返回False
                    logger.warning(f"没有标签包含数值属性，但查询需要数值属性（AVG/SUM）")
                    return False
            else:
                # 不需要数值属性时，选择有可用属性的标签
                labels_with_props = []
                for lb in all_labels:
                    if lb in self.schema.labels:
                        props = list(self.schema.labels[lb].properties.keys())
                        if self.excluded_return_props:
                            props = [p for p in props if p not in self.excluded_return_props]
                        if props:
                            labels_with_props.append(lb)
                
                if labels_with_props:
                    label = random.choice(labels_with_props)
                else:
                    label = random.choice(all_labels)
        
        # 从数据库中采样一个该标签的节点
        sampled_node = self._sample_node_by_label(label)
        if not sampled_node:
            return False
        
        # 填充标签参数
        params_used[label_param] = label
        
        # 预先判断当前是否为聚合查询
        is_aggregate_query = self._is_aggregate_query(template, params_used)
        
        # 填充属性和值参数
        for prop_param, value_param in prop_value_pairs:
            # 如果属性参数和值参数都已填充，跳过
            if prop_param in params_used and value_param in params_used:
                continue
            
            # 如果属性参数已填充，使用已填充的属性
            if prop_param in params_used:
                selected_prop = params_used[prop_param]
                if selected_prop not in sampled_node:
                    # 如果已填充的属性不在采样节点中，返回False
                    return False
                prop_value = sampled_node[selected_prop]
                params_used[value_param] = prop_value
                continue
            
            # 如果值参数已填充但属性参数未填充，尝试从采样节点中找到对应的属性
            if value_param in params_used:
                target_value = params_used[value_param]
                # 在采样节点中查找匹配的属性
                for prop_name, prop_value in sampled_node.items():
                    if prop_value == target_value:
                        params_used[prop_param] = prop_name
                        break
                # 如果找到了匹配的属性，继续下一个
                if prop_param in params_used:
                    continue
            
            # 从采样的节点中选择一个属性
            node_props = list(sampled_node.keys())
            if self.excluded_return_props:
                node_props = [p for p in node_props if p not in self.excluded_return_props]
            
            # 对于聚合相关查询，普通属性参数（如 P、PROP 等）不应落到 ID 相关属性上
            # 注意：这里仅针对“普通属性参数”做过滤，真正的 ID 参数（如 PROP_ID / PROP_ID1 等）
            # 仍然允许选择 ID 属性，以避免破坏依赖 ID 语义的非聚合模板。
            if is_aggregate_query and not ('ID' in prop_param or prop_param.endswith('_ID')):
                non_id_props = [p for p in node_props if not self._is_id_property(p)]
                if non_id_props:
                    node_props = non_id_props
            
            if not node_props:
                return False
            
            # 如果需要数值属性（例如 AVG, SUM 等），只保留数值属性
            if template.required_numeric_props and label in self.schema.labels:
                num_props = []
                for p in node_props:
                    prop_info = self.schema.labels[label].properties.get(p)
                    if prop_info and prop_info.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                        num_props.append(p)
                
                if num_props:
                    node_props = num_props
                else:
                    # 如果当前标签没有数值属性，尝试从其他标签采样节点
                    logger.debug(f"标签 {label} 的采样节点没有数值属性，尝试从其他标签采样")
                    all_labels = list(self.schema.labels.keys())
                    if self.excluded_labels:
                        all_labels = [lb for lb in all_labels if lb not in self.excluded_labels]
                    
                    # 尝试其他标签
                    found_alternative = False
                    for candidate_label in all_labels:
                        if candidate_label == label:
                            continue
                        
                        # 检查候选标签是否有数值属性
                        if candidate_label not in self.schema.labels:
                            continue
                        
                        candidate_props = list(self.schema.labels[candidate_label].properties.keys())
                        if self.excluded_return_props:
                            candidate_props = [p for p in candidate_props if p not in self.excluded_return_props]
                        
                        # 检查是否有数值属性
                        has_numeric = False
                        for p in candidate_props:
                            prop_info = self.schema.labels[candidate_label].properties.get(p)
                            if prop_info and prop_info.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                                has_numeric = True
                                break
                        
                        if has_numeric:
                            # 尝试从候选标签采样节点
                            candidate_node = self._sample_node_by_label(candidate_label)
                            if candidate_node:
                                candidate_node_props = list(candidate_node.keys())
                                if self.excluded_return_props:
                                    candidate_node_props = [p for p in candidate_node_props if p not in self.excluded_return_props]
                                
                                # 检查采样节点中的数值属性
                                candidate_num_props = []
                                for p in candidate_node_props:
                                    if p in self.schema.labels[candidate_label].properties:
                                        prop_info = self.schema.labels[candidate_label].properties.get(p)
                                        if prop_info and prop_info.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                                            candidate_num_props.append(p)
                                
                                if candidate_num_props:
                                    # 找到有数值属性的节点，更新label和sampled_node
                                    logger.debug(f"从标签 {candidate_label} 采样到有数值属性的节点，使用该标签")
                                    label = candidate_label
                                    sampled_node = candidate_node
                                    node_props = candidate_num_props
                                    # 更新params_used中的label参数
                                    params_used[label_param] = label
                                    found_alternative = True
                                    break
                    
                    if not found_alternative:
                        # 如果仍然没有找到数值属性，返回False
                        logger.warning(f"所有标签都没有数值属性，但查询需要数值属性（AVG/SUM）")
                        return False
            
            # 根据属性参数名选择合适的属性
            selected_prop = None
            if 'ID' in prop_param or prop_param.endswith('_ID'):
                # 如果是ID属性，优先选择ID相关的属性
                id_props = [p for p in node_props if self._is_id_property(p)]
                if id_props:
                    selected_prop = random.choice(id_props)
                else:
                    selected_prop = random.choice(node_props)
            else:
                # 普通属性，随机选择
                selected_prop = random.choice(node_props)
            
            if selected_prop not in sampled_node:
                return False
            
            prop_value = sampled_node[selected_prop]
            
            # 填充属性参数
            params_used[prop_param] = selected_prop
            # 填充值参数
            params_used[value_param] = prop_value
        
        return True
    
    def _sample_node_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        从数据库中采样一个指定标签的节点。
        通过随机 SKIP 在整图中分散采样，避免总在同一区域采样导致重复查询。
        
        Args:
            label: 节点标签
        
        Returns:
            节点的属性字典，如果失败返回 None
        """
        if not self.driver:
            return None
        
        try:
            with self.driver.session() as session:
                total = self._get_label_count(label)
                # 在 [0, max(0, total-100)] 间随机跳过，使每次采样的“窗口”落在图的不同区域
                window_size = 100
                skip = random.randint(0, max(0, total - window_size)) if total > window_size else 0
                query = f"MATCH (n:`{label}`) WITH n SKIP $skip LIMIT {window_size} RETURN n"
                result = session.run(query, skip=skip)
                
                nodes = []
                for record in result:
                    node = record["n"]
                    node_dict = dict(node)
                    # 排除内部字段
                    if self.excluded_return_props:
                        node_dict = {k: v for k, v in node_dict.items() 
                                   if k not in self.excluded_return_props}
                    
                    # 如果是 mcp 或 multi-fin 数据集，排除 id 属性以数字结尾的节点
                    if self.dataset in ("mcp", "multi_fin") and node_dict:
                        # 检查所有可能的 id 属性（id, node_id, entity_id 等）
                        id_props = ["id", "node_id", "entity_id"]
                        should_exclude = False
                        for id_prop in id_props:
                            if id_prop in node_dict:
                                id_value = node_dict[id_prop]
                                if isinstance(id_value, str) and id_value and id_value[-1].isdigit():
                                    should_exclude = True
                                    break
                        if should_exclude:
                            continue
                    
                    if node_dict:
                        nodes.append(node_dict)
                
                if nodes:
                    return random.choice(nodes)
        except Exception as e:
            logger.warning(f"采样节点失败 (label={label}): {e}")
        
        return None
    
    def _get_template_constraints(self, template_str: str) -> List[Tuple[str, str, str]]:
        """
        解析模版字符串，提取 (StartLabelParam, RelParam, EndLabelParam) 约束
        """
        constraints = []
        
        # 移除属性部分以简化解析
        temp = re.sub(r'\{[^\}]*\}', '', template_str)
        
        # 1. 提取节点变量定义: var -> label_param
        # (var:$Param)
        var_to_param = {}
        for m in re.finditer(r'\((\w+)\s*:\s*\$(\w+)\)', temp):
            var_to_param[m.group(1)] = m.group(2)
            
        # 2. 提取关系约束
        # 查找所有带有参数的关系: -[:$REL]->
        # 匹配 [...$Param...]
        
        for m in re.finditer(r'\[[^\]]*\$(\w+)[^\]]*\]', temp):
            rel_param = m.group(1)
            start_pos = m.start()
            end_pos = m.end()
            
            # 检查方向，仅处理 ->
            is_right_directed = False
            # 检查 ] 后面是否有 -> (允许中间有空格)
            snippet = temp[end_pos:end_pos+5]
            if '>' in snippet and '-' in snippet: # 简单的检查
                 is_right_directed = True
            
            if not is_right_directed:
                continue

            # 向左查找起始节点 (...)
            left_paren_end = temp.rfind(')', 0, start_pos)
            if left_paren_end == -1: continue
            
            left_paren_start = temp.rfind('(', 0, left_paren_end)
            if left_paren_start == -1: continue
            
            start_node_str = temp[left_paren_start:left_paren_end+1]
            
            # 向右查找结束节点 (...)
            right_paren_start = temp.find('(', end_pos)
            if right_paren_start == -1: continue
            
            right_paren_end = temp.find(')', right_paren_start)
            if right_paren_end == -1: continue
            
            end_node_str = temp[right_paren_start:right_paren_end+1]
            
            # 解析 Start Node Param
            start_param = None
            m_start = re.search(r':\s*\$(\w+)', start_node_str)
            if m_start:
                start_param = m_start.group(1)
            else:
                m_var = re.search(r'\(\s*(\w+)\s*\)', start_node_str)
                if m_var:
                    var = m_var.group(1)
                    start_param = var_to_param.get(var)
            
            # 解析 End Node Param
            end_param = None
            m_end = re.search(r':\s*\$(\w+)', end_node_str)
            if m_end:
                end_param = m_end.group(1)
            else:
                m_var = re.search(r'\(\s*(\w+)\s*\)', end_node_str)
                if m_var:
                    var = m_var.group(1)
                    end_param = var_to_param.get(var)
            
            # 有 rel 和 end 即添加约束（start 可为空，如 MATCH (a)-[r:$R]->(b:$L1) 中 (a) 无标签）
            # 保证 L1 等 end 标签按关系 R 的语义从 schema.triplets 合法 end 中采样
            if rel_param and end_param:
                constraints.append((start_param or '', rel_param, end_param))
                
        return constraints

    def _generate_param_value(self, param_name: str, template: Template, 
                              current_params: Dict[str, Any]) -> Optional[Any]:
        """生成参数值"""
        
        # 获取模版约束
        constraints = self._get_template_constraints(template.template)

        # 处理所有LABEL变体（LABEL, LABEL1, LABEL2, TARGET_LABEL, START_LABEL, END_LABEL, GROUP_LABEL, GL等）
        # GROUP_LABEL/GL 必须按关系的 end 约束采样，避免出现 Company_Guarantee_Company->Account 等错误
        if param_name.startswith('LABEL') or param_name in ('START_LABEL', 'END_LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'SL', 'EL', 'RL', 'REL_NODE_LABEL', 'TARGET_LABEL', 'GROUP_LABEL', 'GL'):
            # 随机选择一个label，优先排除技术性基础标签（如 NGDBNode）
            all_labels = list(self.schema.labels.keys())
            if not all_labels:
                return None

            if self.excluded_labels:
                labels = [lb for lb in all_labels if lb not in self.excluded_labels]
                # 如果全被排除（极端情况），则回退到所有标签
                if not labels:
                    labels = all_labels
            else:
                labels = all_labels
            
            # 优先选择有可用属性的标签
            # 这样可以避免后续生成 PROP/NP 等参数时失败
            labels_with_props = []
            for lb in labels:
                if lb in self.schema.labels:
                    props = list(self.schema.labels[lb].properties.keys())
                    if self.excluded_return_props:
                        props = [p for p in props if p not in self.excluded_return_props]
                    if props:
                        labels_with_props.append(lb)
            
            if labels_with_props:
                labels = labels_with_props

            # 应用约束过滤
            candidates = set(labels)
            if self.schema.triplets:
                for start, rel, end in constraints:
                    # 如果当前参数是 Start Label
                    if start == param_name:
                        # 检查 Rel 是否已知
                        if rel in current_params:
                            r_val = current_params[rel]
                            valid_starts = {t[0] for t in self.schema.triplets if t[1] == r_val}
                            candidates &= valid_starts
                            
                            # 检查 End 是否已知
                            if end in current_params:
                                e_val = current_params[end]
                                valid_starts_strict = {t[0] for t in self.schema.triplets if t[1] == r_val and t[2] == e_val}
                                candidates &= valid_starts_strict
                        elif end in current_params:
                            # Rel 未知，但 End 已知
                            e_val = current_params[end]
                            valid_starts = {t[0] for t in self.schema.triplets if t[2] == e_val}
                            candidates &= valid_starts

                    # 如果当前参数是 End Label
                    if end == param_name:
                        if rel in current_params:
                            r_val = current_params[rel]
                            valid_ends = {t[2] for t in self.schema.triplets if t[1] == r_val}
                            candidates &= valid_ends
                            
                            if start in current_params:
                                s_val = current_params[start]
                                valid_ends_strict = {t[2] for t in self.schema.triplets if t[1] == r_val and t[0] == s_val}
                                candidates &= valid_ends_strict
                        elif start in current_params:
                            s_val = current_params[start]
                            valid_ends = {t[2] for t in self.schema.triplets if t[0] == s_val}
                            candidates &= valid_ends

            if not candidates:
                # 如果约束太严格导致没有候选，回退到所有可用标签
                # 这样可以避免因为约束过严而导致参数生成失败
                logger.debug(f"参数 {param_name} 的约束太严格，回退到所有可用标签")
                if labels:
                    return random.choice(labels)
                return None
            return random.choice(list(candidates))
        
        # 处理所有REL变体（REL_TYPE, REL1, REL2, REL3, R1, R2等）
        elif param_name.startswith('REL') and param_name != 'REL_PROP' or param_name in ('R1', 'R2', 'R3', 'REL', 'R'):
            # 如果是 REL_TYPE，并且已经通过 VALUE/PROP_ID 采样到了具体节点，
            # 优先从该真实节点上存在的关系类型中采样，避免与节点无关的随机关系类型
            if param_name == 'REL_TYPE':
                try:
                    rel_types_from_node = self._get_rel_types_for_value_node(template, current_params)
                except Exception as e:
                    logger.debug(f"从 VALUE 节点获取关系类型失败，回退到 schema 级采样: {e}")
                    rel_types_from_node = None
                if rel_types_from_node:
                    # 这里已经是基于具体 VALUE 节点真实存在的关系
                    return random.choice(rel_types_from_node)

            # 随机选择一个关系类型
            rel_types = list(self.schema.relationships.keys())
            # 如果是 mcp 或 multi_fin 数据集，排除 mention_in 和 is_participated_by
            if self.dataset in ("mcp", "multi_fin"):
                rel_types = [rt for rt in rel_types if rt not in ("mention_in", "is_participated_by")]
            if not rel_types:
                return None
            
            candidates = set(rel_types)
            had_strict_constraint = False  # 是否应用过 (L1,R1,L2) 严格约束
            if self.schema.triplets:
                for start, rel, end in constraints:
                    if rel == param_name:
                        if start in current_params:
                            s_val = current_params[start]
                            valid_rels = {t[1] for t in self.schema.triplets if t[0] == s_val}
                            candidates &= valid_rels
                        
                        if end in current_params:
                            e_val = current_params[end]
                            valid_rels = {t[1] for t in self.schema.triplets if t[2] == e_val}
                            candidates &= valid_rels
                            
                            if start in current_params:
                                s_val = current_params[start]
                                valid_rels_strict = {t[1] for t in self.schema.triplets if t[0] == s_val and t[2] == e_val}
                                candidates &= valid_rels_strict
                                had_strict_constraint = True
            
            if not candidates:
                # 若已按 (L1,R1,L2) 严格约束过滤，则不允许回退到随机关系，避免 Medium-Account 间出现 Company_Own_Account
                if had_strict_constraint:
                    return None
                # 否则回退：尝试随机返回一个关系类型
                if rel_types:
                    return random.choice(rel_types)
                return None
            return random.choice(list(candidates))
        
        elif param_name in ('REL_PROP', 'RP'):
            # 关系属性 - 从schema中获取真实的关系属性
            # 首先尝试从current_params中获取关系类型
            rel_type = None
            for rel_key in ['REL_TYPE', 'REL1', 'REL2', 'REL3', 'R1', 'R2', 'R3', 'REL', 'R']:
                if rel_key in current_params:
                    rel_type = current_params.get(rel_key)
                    break
            
            # 如果找到了关系类型，尝试从schema中获取该关系的属性
            if rel_type and rel_type in self.schema.relationships:
                rel_info = self.schema.relationships[rel_type]
                if rel_info.properties:
                    # 从真实的关系属性中选择
                    rel_props = list(rel_info.properties.keys())
                    # 排除内部字段
                    if self.excluded_return_props:
                        rel_props = [p for p in rel_props if p not in self.excluded_return_props]
                    if rel_props:
                        return random.choice(rel_props)
            
            # 如果没有找到关系类型或该关系没有属性，尝试从所有关系中查找有属性的关系
            # 收集所有有属性的关系类型
            rels_with_props = []
            for rt, rel_info in self.schema.relationships.items():
                # 如果是 mcp 或 multi_fin 数据集，排除 mention_in 和 is_participated_by
                if self.dataset in ("mcp", "multi_fin") and rt in ("mention_in", "is_participated_by"):
                    continue
                if rel_info.properties:
                    rels_with_props.append(rt)
            
            if rels_with_props:
                # 随机选择一个有属性的关系类型，然后选择其属性
                selected_rel_type = random.choice(rels_with_props)
                rel_info = self.schema.relationships[selected_rel_type]
                rel_props = list(rel_info.properties.keys())
                if self.excluded_return_props:
                    rel_props = [p for p in rel_props if p not in self.excluded_return_props]
                if rel_props:
                    return random.choice(rel_props)
            
            # 如果所有关系都没有属性，返回None（这样查询构建会失败，而不是生成一个不存在的属性）
            logger.warning(f"未找到关系属性，关系类型: {rel_type}, 模板: {template.id}")
            return None
        
        elif param_name in (
            'PROP', 'PROP1', 'PROP2',
            'P', 'P1', 'P2', 'SP', 'BP',
            'RET_PROP', 'RETURN_PROP', 'RET_PROP1', 'RET_PROP2', 'RET_PROP3',
            'FILTER_PROP', 'NODE_PROP', 'START_PROP', 'REF_PROP',
            'UPDATE_PROP',  # 用于 SET n.$UPDATE_PROP = $UPDATE_VAL
            'P', 'RET',
        ):
            # 需要根据已选的label来选择属性
            # 尝试从不同的label参数获取label
            label = None
            for label_key in ['LABEL2', 'LABEL3', 'LABEL4', 'LABEL1', 'LABEL', 'END_LABEL', 'START_LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'SL', 'EL', 'RL', 'TARGET_LABEL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            if not label or label not in self.schema.labels:
                # 如果找不到对应的label，随机选一个
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            
            properties = list(self.schema.labels[label].properties.keys())
            if not properties:
                return None

            # 无论是用于筛选还是返回，都不允许使用被排除的内部字段（例如 _node_id, file_id）
            if self.excluded_return_props:
                properties = [p for p in properties if p not in self.excluded_return_props]
                if not properties:
                    return None

            # 如果是聚合查询，排除ID相关的属性
            if self._is_aggregate_query(template, current_params):
                properties = [p for p in properties if not self._is_id_property(p)]
                if not properties:
                    return None
            
            # 如果需要数值属性（例如 AVG, SUM, > < 等），只保留数值属性
            if template.required_numeric_props:
                num_props = []
                for p in properties:
                    prop_info = self.schema.labels[label].properties.get(p)
                    if prop_info and prop_info.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                        num_props.append(p)
                
                if num_props:
                    properties = num_props
                else:
                    # 如果当前标签没有数值属性，尝试从其他标签查找
                    logger.debug(f"标签 {label} 没有数值属性，尝试从其他标签查找数值属性")
                    all_labels = list(self.schema.labels.keys())
                    if self.excluded_labels:
                        all_labels = [lb for lb in all_labels if lb not in self.excluded_labels]
                    
                    # 记录原始label，用于更新current_params
                    original_label = label
                    original_label_key = None
                    for label_key in ['LABEL2', 'LABEL3', 'LABEL4', 'LABEL1', 'LABEL', 'END_LABEL', 'START_LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'SL', 'EL', 'RL']:
                        if label_key in current_params and current_params[label_key] == original_label:
                            original_label_key = label_key
                            break
                    
                    # 尝试其他标签
                    found_alternative = False
                    for candidate_label in all_labels:
                        if candidate_label == label:
                            continue
                        
                        candidate_props = list(self.schema.labels[candidate_label].properties.keys())
                        if self.excluded_return_props:
                            candidate_props = [p for p in candidate_props if p not in self.excluded_return_props]
                        
                        # 如果是聚合查询，排除ID相关的属性
                        if self._is_aggregate_query(template, current_params):
                            candidate_props = [p for p in candidate_props if not self._is_id_property(p)]
                        
                        # 检查是否有数值属性
                        candidate_num_props = []
                        for p in candidate_props:
                            prop_info = self.schema.labels[candidate_label].properties.get(p)
                            if prop_info and prop_info.prop_type in (PropertyType.INTEGER, PropertyType.FLOAT):
                                candidate_num_props.append(p)
                        
                        if candidate_num_props:
                            # 找到有数值属性的标签，更新label和properties
                            logger.debug(f"找到标签 {candidate_label} 有数值属性，使用该标签替代 {original_label}")
                            label = candidate_label
                            properties = candidate_num_props
                            # 更新current_params中的label参数
                            if original_label_key:
                                current_params[original_label_key] = candidate_label
                            found_alternative = True
                            break
                    
                    # 如果仍然没有找到数值属性，返回None
                    if not found_alternative:
                        logger.warning(f"所有标签都没有数值属性，但查询需要数值属性（AVG/SUM）")
                        return None

            return random.choice(properties)
        
        elif param_name == 'PROP_ID':
            # 当参数是 PROP_ID 时，选择节点属性中含有 id 的属性
            # 模板中 (g:$GROUP_LABEL {$PROP_ID: $GID}) 的 PROP_ID 必须来自 GROUP_LABEL 对应标签，避免 Account {companyId}
            label = None
            for label_key in ['GROUP_LABEL', 'GL', 'LABEL2', 'LABEL3', 'LABEL4', 'LABEL1', 'LABEL', 'END_LABEL', 'START_LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'SL', 'EL', 'RL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            if not label or label not in self.schema.labels:
                # 如果找不到对应的label，随机选一个
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            
            properties = list(self.schema.labels[label].properties.keys())
            if not properties:
                return None

            # 筛选出名称包含 id 的属性
            id_properties = [p for p in properties if self._is_id_property(p)]
            
            if not id_properties:
                return None

            # 排除被排除的内部字段（例如 _node_id, file_id）
            if self.excluded_return_props:
                id_properties = [p for p in id_properties if p not in self.excluded_return_props]
                if not id_properties:
                    return None

            return random.choice(id_properties)
        
        elif param_name == 'PROP_ID1':
            # PROP_ID1 对应第一个节点的 ID 属性，通常与 L1 或 LABEL1 关联
            label = None
            for label_key in ['L1', 'LABEL1', 'START_LABEL', 'SL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            if not label or label not in self.schema.labels:
                # 如果找不到对应的label，随机选一个
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            
            properties = list(self.schema.labels[label].properties.keys())
            if not properties:
                return None

            # 筛选出名称包含 id 的属性
            id_properties = [p for p in properties if self._is_id_property(p)]
            
            if not id_properties:
                return None

            # 排除被排除的内部字段（例如 _node_id, file_id）
            if self.excluded_return_props:
                id_properties = [p for p in id_properties if p not in self.excluded_return_props]
                if not id_properties:
                    return None

            return random.choice(id_properties)
        
        elif param_name == 'PROP_ID2':
            # PROP_ID2 对应第二个节点的 ID 属性，通常与 L2 或 LABEL2 关联
            label = None
            for label_key in ['L2', 'LABEL2', 'END_LABEL', 'EL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            if not label or label not in self.schema.labels:
                # 如果找不到对应的label，随机选一个
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            
            properties = list(self.schema.labels[label].properties.keys())
            if not properties:
                return None

            # 筛选出名称包含 id 的属性
            id_properties = [p for p in properties if self._is_id_property(p)]
            
            if not id_properties:
                return None

            # 排除被排除的内部字段（例如 _node_id, file_id）
            if self.excluded_return_props:
                id_properties = [p for p in id_properties if p not in self.excluded_return_props]
                if not id_properties:
                    return None

            return random.choice(id_properties)
        
        elif param_name in ('TARGET_PROP_ID', 'TARGET_PROP_ID1', 'TARGET_PROP_ID2'):
            # 目标节点的 ID 属性名，与 TARGET_LABEL/L1/L2 关联（用于 management/MIX 模板）
            label = None
            if param_name == 'TARGET_PROP_ID':
                label_keys = ['TARGET_LABEL', 'L2', 'END_LABEL', 'EL']
            elif param_name == 'TARGET_PROP_ID1':
                label_keys = ['L1', 'START_LABEL', 'LABEL', 'L']
            else:  # TARGET_PROP_ID2
                label_keys = ['L2', 'TARGET_LABEL', 'END_LABEL', 'EL', 'LABEL', 'L']
            for label_key in label_keys:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            if not label or label not in self.schema.labels:
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            properties = list(self.schema.labels[label].properties.keys())
            if not properties:
                return None
            id_properties = [p for p in properties if self._is_id_property(p)]
            if not id_properties:
                return None
            if self.excluded_return_props:
                id_properties = [p for p in id_properties if p not in self.excluded_return_props]
                if not id_properties:
                    return None
            return random.choice(id_properties)
        
        elif param_name == 'OP':
            # 根据模版判断需要什么类型的操作符
            if template.required_numeric_props:
                return random.choice(self.NUMERIC_OPERATORS)
            else:
                return random.choice(self.STRING_OPERATORS)
        
        elif param_name in ('VALUE', 'VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'V', 'V1', 'V2', 'V3', 'V4', 'V5', 'SV', 'NV', 'NEW_VAL', 'REF_VAL', 'VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5'):
            # SET r.$RP = $VALUE 等：优先从关系属性的 sample_values 采样，避免合成 val_xxxx
            rel_type = current_params.get('R') or current_params.get('REL_TYPE') or current_params.get('REL')
            rp = current_params.get('RP') or current_params.get('REL_PROP')
            if rel_type and rp and rel_type in self.schema.relationships:
                rel_props = self.schema.relationships[rel_type].properties
                if rp in rel_props and rel_props[rp].sample_values:
                    pool = list(rel_props[rp].sample_values)
                    exclude = [current_params[k] for k in ('VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5') if k in current_params and k != param_name]
                    pool = [v for v in pool if v not in exclude]
                    value = random.choice(pool) if pool else random.choice(rel_props[rp].sample_values)
                    if value is not None:
                        return value
            # 根据属性获取一个样本值（必须从数据库对应 PROP 采样，PROP 必须从对应 LABEL 采样）
            # VALUE/VAL/V1..V5 对应 (n:$LABEL {$PROP: $VALUE1}) 或 (n:$LABEL {$PROP_ID: $VALUE1})
            # 当模板中为 {$PROP_ID: $VALUE1} 时，VALUE1 必须从 PROP_ID 对应属性（如 loanId）的 sample_values 采样，
            # 不能从 PROP（如 loanUsage）采样，否则会出现 loanId: 'medical expenses' 这类错误。
            prop = None
            label = None
            # 检测模板是否将 VALUE/VALUE1-5 与 PROP_ID 绑定在同一节点谓词中（如 {$PROP_ID: $VALUE1}）
            template_str = getattr(template, 'template', '') or ''
            value_bound_to_prop_id = (
                param_name in ('VALUE', 'VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5')
                and re.search(r'\$PROP_ID\s*:\s*\$VALUE(\d*)\b', template_str)
            )
            if value_bound_to_prop_id and current_params.get('PROP_ID'):
                # 明确从 PROP_ID 对应属性采样，避免用 PROP 的 sample_values 填到 PROP_ID 上
                prop_order = ['PROP_ID', 'PROP_ID1', 'PROP_ID2', 'PROP', 'P', 'PROP1', 'P1', 'PROP2', 'P2', 'FILTER_PROP', 'NODE_PROP', 'START_PROP', 'SP', 'NP', 'BP', 'GP', 'RP']
            elif param_name in ('VALUE', 'VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5'):
                prop_order = ['PROP', 'P', 'PROP1', 'P1', 'PROP2', 'P2', 'FILTER_PROP', 'NODE_PROP', 'START_PROP', 'PROP_ID', 'PROP_ID1', 'PROP_ID2', 'SP', 'NP', 'BP', 'GP', 'RP']
            else:
                prop_order = ['P', 'PROP', 'P1', 'PROP_ID1', 'PROP_ID', 'PROP_ID2', 'PROP1', 'PROP2', 'FILTER_PROP', 'NODE_PROP', 'START_PROP', 'P2', 'SP', 'NP', 'BP', 'GP', 'RP']
            for prop_key in prop_order:
                if prop_key in current_params:
                    prop = current_params.get(prop_key)
                    break
            for label_key in ['LABEL', 'L', 'L1', 'LABEL1', 'START_LABEL', 'LABEL2', 'L2', 'LABEL3', 'L3', 'L4', 'SL', 'EL', 'RL', 'GL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            value = None
            if prop:
                # 先尝试用已有 label
                if label and label in self.schema.labels and prop in self.schema.labels[label].properties:
                    prop_info = self.schema.labels[label].properties[prop]
                    if prop_info.sample_values:
                        pool = list(prop_info.sample_values)
                        exclude = [current_params[k] for k in ('VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5', 'V1', 'V2', 'V3', 'V4', 'V5', 'VAL1', 'VAL2', 'VAL3', 'VAL4', 'VAL5') if k in current_params and k != param_name]
                        pool = [v for v in pool if v not in exclude]
                        value = random.choice(pool) if pool else (random.choice(prop_info.sample_values) if prop_info.sample_values else None)
                if value is None:
                    # 无 label 或该 label 无此属性/无样本：在所有 label 中查找拥有该属性且 sample_values 非空的
                    for cand_label, label_info in self.schema.labels.items():
                        if self.excluded_labels and cand_label in self.excluded_labels:
                            continue
                        if prop not in label_info.properties:
                            continue
                        prop_info = label_info.properties[prop]
                        if not prop_info.sample_values:
                            continue
                        pool = list(prop_info.sample_values)
                        exclude = [current_params[k] for k in ('VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5', 'V1', 'V2', 'V3', 'V4', 'V5', 'VAL1', 'VAL2', 'VAL3', 'VAL4', 'VAL5') if k in current_params and k != param_name]
                        pool = [v for v in pool if v not in exclude]
                        value = random.choice(pool) if pool else random.choice(prop_info.sample_values)
                        break
            
            if value is not None:
                return value
            
            # 回退策略：如果无法从 schema 获取样本值，生成一个随机值
            # 尝试推断期望的类型
            expected_type = 'string'
            if hasattr(template, 'parameters') and param_name in template.parameters:
                expected_type = template.parameters[param_name]
            
            if expected_type in ('integer', 'long', 'int'):
                return random.randint(1, 1000)
            elif expected_type in ('float', 'double'):
                return random.uniform(0, 1000)
            elif expected_type == 'boolean':
                return random.choice([True, False])
            else:
                # 默认为字符串
                return f"val_{random.randint(1000, 9999)}"
        
        elif param_name in ('MIN_HOPS', 'MAX_HOPS'):
            # 跳数参数，返回随机整数
            if param_name == 'MIN_HOPS':
                return random.randint(1, 3)
            else:  # MAX_HOPS
                return random.randint(2, 5)
        
        elif param_name in ('D1', 'D2', 'D3'):
            # 方向参数，返回空字符串或'-'
            return random.choice(['', '-'])
        
        # 在 _generate_param_value 函数中添加以下代码段：

        # 1. 聚合函数参数（AGG_FUNC, AGG_FUNC1, AGG_FUNC2）
        elif param_name.startswith('AGG_FUNC') or param_name == 'AGG_FUNC':
            # 常见的聚合函数
            agg_functions = ['sum', 'avg', 'count', 'min', 'max', 'collect']
            return random.choice(agg_functions)

        # 2. 数值属性参数（NUM_PROP, NUM_PROP1, NUM_PROP2）
        elif param_name.startswith('NUM_PROP') or param_name in ('NUM_PROP', 'NP'):
            # 需要根据已选的label来选择数值类型的属性
            label = None
            for label_key in ['LABEL2', 'LABEL3', 'LABEL4', 'LABEL1', 'LABEL', 'END_LABEL', 'START_LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'SL', 'EL', 'RL']:
                if label_key in current_params:
                    label = current_params.get(label_key)
                    break
            
            if not label or label not in self.schema.labels:
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            
            # 筛选出数值类型的属性
            properties = self.schema.labels[label].properties
            numeric_props = []
            
            for prop_name, prop_info in properties.items():
                # 尝试通过样本值推断类型
                is_numeric = False
                
                # 方法1：检查是否有 type 属性
                if hasattr(prop_info, 'type') and prop_info.type in ('integer', 'float', 'double', 'long', 'number'):
                    is_numeric = True
                # 方法2：检查其他可能的类型属性名
                elif hasattr(prop_info, 'data_type') and prop_info.data_type in ('integer', 'float', 'double', 'long', 'number'):
                    is_numeric = True
                elif hasattr(prop_info, 'dtype') and prop_info.dtype in ('integer', 'float', 'double', 'long', 'number'):
                    is_numeric = True
                # 方法3：通过样本值推断
                elif hasattr(prop_info, 'sample_values') and prop_info.sample_values:
                    # 检查样本值是否为数值类型
                    sample = prop_info.sample_values[0]
                    if isinstance(sample, (int, float)) and not isinstance(sample, bool):
                        is_numeric = True
                
                if is_numeric:
                    numeric_props.append(prop_name)
            
            # 排除内部字段
            if self.excluded_return_props:
                numeric_props = [p for p in numeric_props if p not in self.excluded_return_props]
            
            # 如果是聚合查询，排除ID相关的属性
            if self._is_aggregate_query(template, current_params):
                numeric_props = [p for p in numeric_props if not self._is_id_property(p)]
            
            if not numeric_props:
                # 如果没有数值属性，返回所有属性中的一个
                all_props = list(properties.keys())
                if self.excluded_return_props:
                    all_props = [p for p in all_props if p not in self.excluded_return_props]
                # 如果是聚合查询，也要排除ID属性
                if self._is_aggregate_query(template, current_params):
                    all_props = [p for p in all_props if not self._is_id_property(p)]
                
                if not all_props:
                    # 如果当前标签没有可用属性，尝试从其他标签中查找
                    for candidate_label in self.schema.labels:
                        cand_props = list(self.schema.labels[candidate_label].properties.keys())
                        if self.excluded_return_props:
                            cand_props = [p for p in cand_props if p not in self.excluded_return_props]
                        
                        # 再次过滤ID属性（如果是聚合查询）
                        if self._is_aggregate_query(template, current_params):
                            cand_props = [p for p in cand_props if not self._is_id_property(p)]
                            
                        if cand_props:
                            return random.choice(cand_props)
                            
                    return None
                return random.choice(all_props)
            
            return random.choice(numeric_props)

        # 3. 操作符参数（OP1, OP2）
        elif param_name.startswith('OP') and param_name not in ('OP',):
            # 数值比较操作符
            return random.choice(self.NUMERIC_OPERATORS)

        # 4. 更新值参数（UPDATE_VAL, UPDATE_VAL1, UPDATE_REL_VAL, UPDATE_NODE_VAL）用于 SET n.$UPDATE_PROP = $UPDATE_VAL
        elif param_name in ('UPDATE_VAL', 'UPDATE_VAL1', 'UPDATE_REL_VAL', 'UPDATE_NODE_VAL'):
            expected_type = 'integer'
            if hasattr(template, 'parameters') and param_name in template.parameters:
                expected_type = template.parameters[param_name]
            if expected_type in ('integer', 'long', 'int'):
                return random.randint(0, 10000)
            if expected_type in ('float', 'double'):
                return round(random.uniform(0, 10000), 2)
            return random.randint(0, 10000)

        # 5. 阈值参数（THRESHOLD, THRESHOLD1, THRESHOLD2, DELETE_THRESHOLD）
        elif param_name == 'DELETE_THRESHOLD' or param_name.startswith('THRESHOLD') or param_name == 'THRESHOLD':
            # 返回一个随机的数值阈值（DELETE_THRESHOLD 用于 WHERE r.$REL_PROP < $DELETE_THRESHOLD）
            # 可以根据实际数据范围调整
            return random.choice([0, 1, 5, 10, 50, 100, 1000])

        # 6. 分类/类别参数（CATEGORY, CATEGORY1, CATEGORY2）
        elif param_name.startswith('CATEGORY') or param_name == 'CATEGORY':
            # 返回一个分类标签
            categories = ['High', 'Medium', 'Low', 'A', 'B', 'C', 'Type1', 'Type2']
            return random.choice(categories)

        # 7. 别名参数（GROUP_ALIAS, COLLECT_ALIAS, ALIAS）
        elif param_name.endswith('ALIAS') or param_name == 'ALIAS':
            # 返回一个别名
            aliases = ['result', 'value', 'data', 'item', 'entity', 'group', 'collection']
            return random.choice(aliases)

        # 8. 分组属性参数（GROUP_PROP）
        elif param_name in ('GROUP_PROP', 'GP'):
            # 需要根据GROUP_LABEL来选择属性
            label = current_params.get('GROUP_LABEL') or current_params.get('GL')
            
            if not label or label not in self.schema.labels:
                labels = list(self.schema.labels.keys())
                if not labels:
                    return None
                label = random.choice(labels)
            
            properties = list(self.schema.labels[label].properties.keys())
            if not properties:
                return None
            
            if self.excluded_return_props:
                properties = [p for p in properties if p not in self.excluded_return_props]
                if not properties:
                    return None
            
            # 如果是聚合查询，排除ID相关的属性
            if self._is_aggregate_query(template, current_params):
                properties = [p for p in properties if not self._is_id_property(p)]
                if not properties:
                    return None
            
            return random.choice(properties)

        # 9. 分组标签参数（GROUP_LABEL）
        elif param_name in ('GROUP_LABEL', 'GL'):
            all_labels = list(self.schema.labels.keys())
            if not all_labels:
                return None
            
            if self.excluded_labels:
                labels = [lb for lb in all_labels if lb not in self.excluded_labels]
                if not labels:
                    labels = all_labels
            else:
                labels = all_labels
    
            return random.choice(labels)

        # 10. 特定属性值参数（NAME, ID, GID, TARGET_ID, TARGET_ID1, TARGET_ID2, DELETE_ID, ID1_1, ID2_1, BID1, AID1 等）
        # 要求：VALUE 必须从数据库对应 PROP 采样，PROP 必须从对应 LABEL 采样
        # 支持批量模板中的 ID1_1/ID2_1（首行 ID）、BID1/BID2、AID1/AID2 等
        elif (param_name in ('NAME', 'GID', 'GID1', 'GID2', 'GID3', 'GID4', 'GID5', 'AID', 'BID', 'CID', 'ID', 'ID1', 'ID2', 'MID', 'TARGET_ID', 'TARGET_ID1', 'TARGET_ID2', 'DELETE_ID')
              or param_name.startswith('ID1_') or param_name.startswith('ID2_')
              or (param_name.startswith('BID') and param_name != 'BID')
              or (param_name.startswith('AID') and param_name != 'AID')):
            # 批量模板中 ID1_1/ID2_1 等价于 ID1/ID2，BID1/AID1 等价于 BID/AID
            base = param_name
            if param_name.startswith('ID1_') or param_name == 'ID1':
                base = 'ID1'
            elif param_name.startswith('ID2_') or param_name == 'ID2':
                base = 'ID2'
            elif param_name.startswith('BID'):
                base = 'BID'
            elif param_name.startswith('AID'):
                base = 'AID'
            elif param_name == 'TARGET_ID1':
                base = 'TARGET_ID1'
            elif param_name == 'TARGET_ID2':
                base = 'TARGET_ID2'
            # 确定目标属性名
            target_prop = 'name' if base == 'NAME' else 'id'
            
            # 尝试找到对应的 Label（必须从对应 LABEL 采样）
            label = None
            prop_to_use = None  # 明确使用该 label 下的哪个属性
            
            # 启发式规则：根据参数名推断关联的 Label 与 PROP
            label_keys = []
            if base in ('GID', 'GID1', 'GID2', 'GID3', 'GID4', 'GID5'):
                label_keys = ['GROUP_LABEL', 'GL']
                # GID 对应目标节点 (g:$GL {$PROP_ID: $GID})，必须用 PROP_ID 从 GL 采样
                prop_to_use = current_params.get('PROP_ID')
            elif base == 'CID':
                label_keys = ['L3']
                prop_to_use = current_params.get('PROP_ID')
            elif base in ('AID', 'ID1'):
                label_keys = ['L1', 'START_LABEL']
                prop_to_use = current_params.get('PROP_ID1') or current_params.get('PROP_ID')
            elif base in ('BID', 'ID2'):
                label_keys = ['L2', 'END_LABEL', 'LABEL2']
                prop_to_use = current_params.get('PROP_ID2') or current_params.get('PROP_ID')
            elif base == 'TARGET_ID':
                label_keys = ['TARGET_LABEL']
                prop_to_use = current_params.get('TARGET_PROP_ID')
            elif base == 'TARGET_ID1':
                label_keys = ['L1', 'START_LABEL', 'LABEL', 'L']
                prop_to_use = current_params.get('TARGET_PROP_ID1')
            elif base == 'TARGET_ID2':
                label_keys = ['L2', 'TARGET_LABEL', 'END_LABEL', 'EL']
                prop_to_use = current_params.get('TARGET_PROP_ID2')
            elif base == 'DELETE_ID':
                label_keys = ['LABEL', 'L1', 'L', 'START_LABEL']
                prop_to_use = current_params.get('PROP_ID')
            elif base == 'NAME':
                label_keys = ['LABEL', 'L1', 'L', 'START_LABEL']
            elif base == 'MID':
                label_keys = ['L2', 'END_LABEL']
                prop_to_use = current_params.get('PROP_ID')
            else:
                label_keys = ['LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'GL', 'SL', 'EL', 'RL', 'TARGET_LABEL']
            
            for key in label_keys:
                if key in current_params:
                    label = current_params[key]
                    break
            
            if not label:
                 for label_key in ['LABEL', 'L1', 'L2', 'L3', 'L4', 'L', 'GL', 'SL', 'EL', 'RL', 'TARGET_LABEL']:
                    if label_key in current_params:
                        label = current_params[label_key]
                        break
            
            if label and label in self.schema.labels:
                 props = self.schema.labels[label].properties
                 # 优先使用对应 PROP（PROP_ID/PROP_ID1/PROP_ID2）从该 LABEL 采样
                 if prop_to_use and prop_to_use in props and props[prop_to_use].sample_values:
                     pool = list(props[prop_to_use].sample_values)
                     # 排除已采样的同组 ID，避免重复（含 ID1_1/ID2_1/BID1/AID1 等）
                     exclude_keys = ('GID1', 'GID2', 'GID3', 'GID4', 'GID5', 'AID1', 'AID2', 'AID3', 'AID4', 'AID5', 'BID1', 'BID2', 'BID3', 'BID4', 'BID5', 'ID1_1', 'ID2_1', 'ID1_2', 'ID2_2')
                     exclude = [current_params[k] for k in exclude_keys if k in current_params and k != param_name]
                     pool = [v for v in pool if v not in exclude]
                     if pool:
                         return random.choice(pool)
                     return random.choice(props[prop_to_use].sample_values)
                 # 如果该 Label 有目标属性，返回其样本值
                 if target_prop in props and props[target_prop].sample_values:
                     return random.choice(props[target_prop].sample_values)
                 # 如果没有找到目标属性，但需要 id，尝试查找该 label 下任意有 sample_values 的属性（避免合成 id_）
                 if target_prop == 'id' or base in ('AID', 'BID', 'CID', 'ID1', 'ID2', 'TARGET_ID', 'DELETE_ID'):
                     for p in props:
                         if props[p].sample_values and (p.lower().endswith('id') or p.lower() in ('id', '_id', 'uid', 'uuid', 'code', 'loanid', 'accountid')):
                             pool = list(props[p].sample_values)
                             exclude_keys = ('AID1', 'AID2', 'AID3', 'AID4', 'AID5', 'BID1', 'BID2', 'BID3', 'BID4', 'BID5', 'ID1_1', 'ID2_1', 'ID1_2', 'ID2_2')
                             exclude = [current_params[k] for k in exclude_keys if k in current_params and k != param_name]
                             pool = [v for v in pool if v not in exclude]
                             if pool:
                                 return random.choice(pool)
                             return random.choice(props[p].sample_values)
                     # 仍无则尝试该 label 下任意有 sample_values 的属性
                     for p in props:
                         if props[p].sample_values:
                             pool = list(props[p].sample_values)
                             exclude_keys = ('AID1', 'AID2', 'AID3', 'AID4', 'AID5', 'BID1', 'BID2', 'BID3', 'BID4', 'BID5', 'ID1_1', 'ID2_1', 'ID1_2', 'ID2_2')
                             exclude = [current_params[k] for k in exclude_keys if k in current_params and k != param_name]
                             pool = [v for v in pool if v not in exclude]
                             if pool:
                                 return random.choice(pool)
                             return random.choice(props[p].sample_values)
            
            # 禁止合成 id_/Name_，必须从数据库采样；无法采样时返回 None，由调用方跳过该模板
            return None
        
        else:
            # 未知参数，记录警告并返回None而不是占位符
            logger.warning(f"未知参数类型: {param_name}，模板: {template.id}")
            return None

    def _get_rel_types_for_value_node(
        self,
        template: Template,
        current_params: Dict[str, Any],
    ) -> Optional[List[str]]:
        """
        基于 VALUE 对应的真实节点，获取其实际存在的关系类型列表。
        主要用于形如：
            MATCH (a:entity {$PROP_ID: $VALUE})-[r:$REL_TYPE]-(b:entity)
        这样的模板，保证 REL_TYPE 一定是该 VALUE 节点真实包含的关系类型。
        """
        # 目前仅在已经有 PROP_ID / VALUE 的前提下才有意义
        if "PROP_ID" not in current_params or "VALUE" not in current_params:
            return None

        if not self.driver:
            return None

        prop_name = current_params["PROP_ID"]
        value = current_params["VALUE"]

        # 从模板字符串中尽量解析出硬编码的标签（如 :entity）
        label = None
        try:
            # 注意：这里是 Python 原始字符串，正则中只需要单反斜杠
            pattern = r":(\w+)\s*\{[^}]*\$PROP_ID[^}]*\$VALUE[^}]*\}"
            match = re.search(pattern, template.template)
            if match:
                label = match.group(1)
        except Exception as e:
            logger.debug(f"解析硬编码标签失败: {e}")

        # 如果没解析到，就回退到默认的 'entity'（mcp 场景下常用）
        if not label:
            label = "entity"

        # 防御性处理：prop_name 应该是字符串
        if not isinstance(prop_name, str):
            return None

        # 基于具体节点查询其相连关系类型
        try:
            with self.driver.session() as session:
                cypher = (
                    f"MATCH (a:`{label}`) "           # label 使用反引号包裹
                    f"WHERE a.`{prop_name}` = $value "  # 属性名同样使用反引号
                    "MATCH (a)-[r]-(b) "             # 任意方向的关系
                    "RETURN collect(DISTINCT type(r)) AS rel_types "
                    "LIMIT 1"
                )
                record = session.run(cypher, value=value).single()
                if not record:
                    return None

                rel_types = record.get("rel_types") or []

                # mcp / multi_fin 下排除 mention_in 和 is_participated_by
                if self.dataset in ("mcp", "multi_fin"):
                    rel_types = [rt for rt in rel_types if rt not in ("mention_in", "is_participated_by")]

                # 返回非空列表时，调用方会在其中随机选择一个
                return rel_types or None

        except Exception as e:
            logger.debug(f"从 VALUE 节点获取关系类型时出错: {e}")
            return None


class QueryExecutor:
    """查询执行器"""
    
    def __init__(self, driver, timeout: int = 300, max_results: int = 1000):
        self.driver = driver
        self.timeout = timeout
        self.max_results = max_results  # 最大结果数量限制
    
    def execute(self, query: str, allow_empty: bool = False) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        执行查询，返回 (是否成功, 结果列表, 错误信息)
        
        Args:
            query: Cypher查询语句
            allow_empty: 是否允许结果为空（对于CREATE/DELETE等无返回值的操作应设为True）
        """
        @contextmanager
        def timeout_context(seconds):
            """超时上下文管理器（仅Unix系统）"""
            def timeout_handler(signum, frame):
                raise TimeoutError(f"查询执行超时 ({seconds}秒)")
            
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Windows系统不支持SIGALRM，直接yield（使用时间检查）
                yield
        
        try:
            # 检查查询是否已经包含LIMIT，如果没有且是可变长度路径查询，添加LIMIT
            query_with_limit = self._add_limit_if_needed(query)
            
            with self.driver.session() as session:
                result = session.run(query_with_limit)
                
                records = []
                record_count = 0
                start_time = time.time()
                
                # 使用超时上下文（如果支持SIGALRM）
                if hasattr(signal, 'SIGALRM'):
                    try:
                        with timeout_context(self.timeout):
                            for record in result:
                                if record_count >= self.max_results:
                                    logger.warning(f"查询结果超过最大限制 ({self.max_results})，已截断")
                                    break
                                
                                # 将记录转换为字典
                                record_dict = {}
                                for key in record.keys():
                                    value = record[key]
                                    # 处理节点对象
                                    if isinstance(value, Node):
                                        # Neo4j 节点对象：提取所有属性
                                        # 使用 dict() 转换，如果失败则尝试 items() 方法
                                        try:
                                            record_dict[key] = dict(value)
                                        except (TypeError, AttributeError):
                                            try:
                                                record_dict[key] = dict(value.items())
                                            except:
                                                # 如果都失败，尝试手动构建字典
                                                record_dict[key] = {k: v for k, v in value.items()}
                                    # 处理关系对象
                                    elif isinstance(value, Relationship):
                                        # Neo4j 关系对象：提取所有属性
                                        try:
                                            record_dict[key] = dict(value)
                                        except (TypeError, AttributeError):
                                            try:
                                                record_dict[key] = dict(value.items())
                                            except:
                                                record_dict[key] = {k: v for k, v in value.items()}
                                    # 处理其他可迭代对象（列表、集合等）
                                    elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, bytes)):
                                        try:
                                            # 如果是列表，递归处理每个元素
                                            if isinstance(value, (list, tuple)):
                                                processed_list = []
                                                for item in value:
                                                    if isinstance(item, Node):
                                                        try:
                                                            processed_list.append(dict(item))
                                                        except:
                                                            try:
                                                                processed_list.append(dict(item.items()))
                                                            except:
                                                                processed_list.append({k: v for k, v in item.items()})
                                                    elif isinstance(item, Relationship):
                                                        try:
                                                            processed_list.append(dict(item))
                                                        except:
                                                            try:
                                                                processed_list.append(dict(item.items()))
                                                            except:
                                                                processed_list.append({k: v for k, v in item.items()})
                                                    else:
                                                        processed_list.append(item)
                                                record_dict[key] = processed_list
                                            else:
                                                record_dict[key] = dict(value)
                                        except:
                                            record_dict[key] = str(value)
                                    else:
                                        record_dict[key] = value
                                records.append(record_dict)
                                record_count += 1
                    except TimeoutError:
                        logger.warning(f"查询执行超时 ({self.timeout}秒)，已停止")
                        return False, [], f"查询执行超时 ({self.timeout}秒)"
                else:
                    # Windows系统：使用时间检查
                    for record in result:
                        # 检查是否超时
                        elapsed = time.time() - start_time
                        if elapsed > self.timeout:
                            logger.warning(f"查询执行时间超过 {self.timeout} 秒，停止读取结果")
                            return False, [], f"查询执行超时 ({self.timeout}秒)"
                        
                        if record_count >= self.max_results:
                            logger.warning(f"查询结果超过最大限制 ({self.max_results})，已截断")
                            break
                        
                        # 将记录转换为字典
                        record_dict = {}
                        for key in record.keys():
                            value = record[key]
                            # 处理节点对象
                            if isinstance(value, Node):
                                # Neo4j 节点对象：提取所有属性
                                # 使用 dict() 转换，如果失败则尝试 items() 方法
                                try:
                                    record_dict[key] = dict(value)
                                except (TypeError, AttributeError):
                                    try:
                                        record_dict[key] = dict(value.items())
                                    except:
                                        # 如果都失败，尝试手动构建字典
                                        record_dict[key] = {k: v for k, v in value.items()}
                            # 处理关系对象
                            elif isinstance(value, Relationship):
                                # Neo4j 关系对象：提取所有属性
                                try:
                                    record_dict[key] = dict(value)
                                except (TypeError, AttributeError):
                                    try:
                                        record_dict[key] = dict(value.items())
                                    except:
                                        record_dict[key] = {k: v for k, v in value.items()}
                            # 处理其他可迭代对象（列表、集合等）
                            elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, bytes)):
                                try:
                                    # 如果是列表，递归处理每个元素
                                    if isinstance(value, (list, tuple)):
                                        processed_list = []
                                        for item in value:
                                            if isinstance(item, Node):
                                                try:
                                                    processed_list.append(dict(item))
                                                except:
                                                    try:
                                                        processed_list.append(dict(item.items()))
                                                    except:
                                                        processed_list.append({k: v for k, v in item.items()})
                                            elif isinstance(item, Relationship):
                                                try:
                                                    processed_list.append(dict(item))
                                                except:
                                                    try:
                                                        processed_list.append(dict(item.items()))
                                                    except:
                                                        processed_list.append({k: v for k, v in item.items()})
                                            else:
                                                processed_list.append(item)
                                        record_dict[key] = processed_list
                                    else:
                                        record_dict[key] = dict(value)
                                except:
                                    record_dict[key] = str(value)
                            else:
                                record_dict[key] = value
                        records.append(record_dict)
                        record_count += 1
                
                # 只有非空结果才算成功，除非明确允许空结果
                if records or allow_empty:
                    return True, records, None
                else:
                    return False, [], "查询结果为空"
                    
        except TimeoutError as e:
            return False, [], str(e)
        except Exception as e:
            return False, [], str(e)
    
    def _add_limit_if_needed(self, query: str) -> str:
        """如果查询是可变长度路径查询且没有LIMIT，添加LIMIT子句"""
        # 检查是否包含可变长度路径模式（如 *1..5, *..5, *1..）
        has_variable_length = bool(re.search(r'\*\s*\d*\s*\.\.\s*\d*', query, re.IGNORECASE))
        
        # 检查是否已经有LIMIT
        has_limit = bool(re.search(r'\bLIMIT\s+\d+', query, re.IGNORECASE))
        
        # 如果是有可变长度路径且没有LIMIT，添加LIMIT
        if has_variable_length and not has_limit:
            # 在RETURN子句后添加LIMIT
            # 使用正则表达式找到最后一个RETURN，在其后添加LIMIT
            # 但要小心处理子查询中的RETURN
            limit_value = min(self.max_results, 100)  # 可变长度路径查询限制为100条结果
            query = re.sub(
                r'(\bRETURN\b[^;]*(?:;|$))',
                rf'\1 LIMIT {limit_value}',
                query,
                flags=re.IGNORECASE
            )
            # 如果上面的替换没有生效（可能RETURN在最后），直接在末尾添加
            if 'LIMIT' not in query.upper():
                query = query.rstrip().rstrip(';') + f' LIMIT {limit_value}'
        
        return query


class QueryGenerator:
    """查询生成器主类"""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 template_path: str = "/query_template/template.json",
                 exclude_internal_id_as_return: bool = True,
                 dataset: Optional[str] = None):
        """
        初始化查询生成器
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            template_path: 模版文件路径
            exclude_internal_id_as_return: 是否在返回属性中排除内部ID字段（默认排除 `_node_id`）
            dataset: 数据集名称，如果为 "mcp" 或 "multi_fin"，采样时会排除 "concept" 和 "passage" 标签
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.template_path = template_path
        self.exclude_internal_id_as_return = exclude_internal_id_as_return
        self.dataset = dataset
        # 需要排除的返回属性列表，默认使用模块级配置，便于用户修改
        self.excluded_return_props: Set[str] = (
            set(DEFAULT_EXCLUDED_RETURN_PROPS) if exclude_internal_id_as_return else set()
        )
        
        # 初始化组件
        self.driver = None
        self.schema = None
        self.template_loader = None
        self.matcher = None
        self.builder = None
        self.executor = None
        
        # 结果存储
        self.results: List[QueryResult] = []
    
    def connect(self):
        """连接到Neo4j数据库"""
        logger.info(f"连接到Neo4j: {self.uri}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # 验证连接
        with self.driver.session() as session:
            session.run("RETURN 1")
        logger.info("连接成功")
    
    def initialize(self):
        """初始化所有组件"""
        if not self.driver:
            self.connect()
        
        # 分析Schema
        self.schema = SchemaAnalyzer(self.driver)
        self.schema.analyze()
        
        # 如果是 mcp 或 multi-fin 数据集，过滤掉 id 属性中以数字结尾的 sample_values
        if self.dataset in ("mcp", "multi_fin"):
            self._filter_id_samples_ending_with_digit()
        
        # 加载模版
        self.template_loader = TemplateLoader(self.template_path)
        self.template_loader.load()
        
        # 初始化其他组件
        self.matcher = TemplateMatcher(self.schema)
        
        # 根据 dataset 决定需要排除的标签
        excluded_labels = set(DEFAULT_EXCLUDED_LABELS)
        if self.dataset in ("mcp", "multi_fin"):
            excluded_labels.update({"passage"})  # 只排除 passage，不排除 concept
        
        self.builder = QueryBuilder(
            self.schema,
            excluded_return_props=self.excluded_return_props,
            excluded_labels=excluded_labels,
            dataset=self.dataset,  # 传递 dataset 信息
            driver=self.driver,  # 传递 driver 用于节点采样
        )
        self.executor = QueryExecutor(self.driver)
    
    def _filter_id_samples_ending_with_digit(self):
        """
        过滤掉所有 id 属性中以数字结尾的 sample_values
        用于 mcp 和 multi-fin 数据集
        """
        if not self.schema:
            return
        
        # 遍历所有 label
        for label_name, label_info in self.schema.labels.items():
            # 遍历所有属性
            for prop_name, prop_info in label_info.properties.items():
                # 检查是否是 id 相关属性
                if prop_name.lower() in ("id", "node_id", "entity_id"):
                    # 过滤掉以数字结尾的字符串值
                    original_count = len(prop_info.sample_values)
                    prop_info.sample_values = [
                        v for v in prop_info.sample_values
                        if not (isinstance(v, str) and v and v[-1].isdigit())
                    ]
                    filtered_count = len(prop_info.sample_values)
                    if original_count != filtered_count:
                        logger.info(f"过滤 {label_name}.{prop_name}: {original_count} -> {filtered_count} 个样本值")
    
    def get_target_sample_count(self) -> int:
        """获取目标采样数量（边数/16）"""
        if not self.schema:
            return 0
        return max(1, self.schema.total_edges // 16)
    
    def generate_samples(self, target_count: Optional[int] = None, 
                        max_attempts_multiplier: int = 10,
                        max_failures_per_template: int = 10000,
                        max_answer_count: int = 20,
                        min_attempts_per_template: int = 5,
                        reset_failures_interval: float = 0.25,
                        stats_output_path: Optional[str] = None,
                        success_per_template: int = 20,
                        realtime_output_path: Optional[str] = None) -> List[QueryResult]:
        """
        生成查询样本，按模版顺序，每个模版连续采样直到生成指定数量的成功查询
        
        Args:
            target_count: 目标样本数量，默认为边数/16
            max_attempts_multiplier: 最大尝试次数倍数
            max_failures_per_template: 每个模版的最大连续失败次数，超过后跳过该模版（默认50）
            max_answer_count: 查询结果的最大数量阈值，超过此数量的查询将被抛弃（默认20）
            min_attempts_per_template: 每个模版的最小尝试次数，确保每个模版至少被尝试这么多次（默认5，已废弃，保留以兼容）
            reset_failures_interval: 阶段性重置失败计数的间隔（已废弃，保留以兼容）
            stats_output_path: 统计信息输出文件路径，如果指定则会将统计信息写入该文件（默认None）
            success_per_template: 每个模版需要生成的成功查询数量（默认10）
            realtime_output_path: 实时输出文件路径，如果指定则会在每次生成成功查询时立即写入文件（默认None）
        
        Returns:
            QueryResult列表
        """
        if not self.schema:
            self.initialize()
        
        target = target_count or self.get_target_sample_count()
        max_attempts = target * max_attempts_multiplier
        
        logger.info(f"开始采样，目标数量: {target}, 最大尝试次数: {max_attempts}, 每个模版成功数量: {success_per_template}")
        
        # 获取可用模版
        usable_templates = self.matcher.get_usable_templates(self.template_loader.templates)
        if not usable_templates:
            logger.error("没有可用的模版")
            return []
        
        logger.info(f"可用模版数量: {len(usable_templates)}")
        
        self.results = []
        attempts = 0
        
        # 实时输出文件处理
        realtime_file = None
        realtime_count = 0
        if realtime_output_path:
            realtime_file = open(realtime_output_path, 'w', encoding='utf-8')
            realtime_file.write('[\n')  # 开始 JSON 数组
            logger.info(f"启用实时输出到文件: {realtime_output_path}")
        
        # 跟踪每个模版的使用情况（使用 (type, id) 作为键以确保唯一性）
        template_stats = {
            (template.type or 'unknown', template.id): {
                'template': template,
                'template_id': template.id,
                'template_type': template.type or 'unknown',
                'usage_count': 0,  # 使用次数
                'success_count': 0,  # 成功次数
                'failure_count': 0,  # 连续失败次数
                'last_attempt_failed': False
            }
            for template in usable_templates
        }
        
        # 按顺序遍历每个模版（使用 try-finally 确保文件正确关闭）
        try:
            for template in usable_templates:
                # 如果已达到目标数量或超过最大尝试次数，停止
                if len(self.results) >= target or attempts >= max_attempts:
                    break
                
                stats_key = (template.type or 'unknown', template.id)
                stats = template_stats[stats_key]
                
                logger.info(f"开始处理模版 [{stats['template_type']}] {template.id}，目标成功数量: {success_per_template}")
                
                # 对当前模版连续采样，直到生成指定数量的成功查询
                while stats['success_count'] < success_per_template:
                    # 检查是否达到全局限制
                    if len(self.results) >= target or attempts >= max_attempts:
                        break
                    
                    # 检查是否超过最大失败次数
                    if stats['failure_count'] >= max_failures_per_template:
                        logger.warning(f"模版 [{stats['template_type']}] {template.id} 连续失败 {stats['failure_count']} 次，跳过该模版")
                        break
                    
                    attempts += 1
                    stats['usage_count'] += 1
                    
                    # 构建查询
                    query, params_used = self.builder.build_query(template)
                    if not query:
                        stats['failure_count'] += 1
                        stats['last_attempt_failed'] = True
                        continue
                    
                    # 执行查询
                    success, answer, error = self.executor.execute(query)
                    
                    # 过滤查询结果
                    if success:
                        # 1. 如果查询结果为空，抛弃这个查询
                        if not answer:
                            stats['failure_count'] += 1
                            stats['last_attempt_failed'] = True
                            logger.debug(f"查询结果为空，抛弃查询: [{stats['template_type']}] {template.id}")
                            continue
                        
                        # 2. 排除所有值都是 null 的记录
                        filtered_answer = []
                        for record in answer:
                            # 检查记录中是否有至少一个非 null 的值
                            has_non_null = any(value is not None for value in record.values())
                            
                            # 如果记录中至少有一个非 null 值，则保留该记录（保留所有字段，包括 null 值）
                            if has_non_null:
                                filtered_answer.append(record)
                        
                        # 如果过滤后结果为空，抛弃这个查询
                        if not filtered_answer:
                            stats['failure_count'] += 1
                            stats['last_attempt_failed'] = True
                            logger.debug(f"过滤后查询结果为空，抛弃查询: [{stats['template_type']}] {template.id}")
                            continue
                        
                        # 3. 检查是否所有 count 类型的结果都为 0
                        # 常见的 count 字段名：cnt, count, total 等
                        count_fields = ['cnt', 'count', 'total', 'num', 'number']
                        
                        # 检查是否所有记录的 count 字段都为 0
                        all_counts_zero = True
                        has_count_field = False
                        
                        for record in filtered_answer:
                            for field in count_fields:
                                if field in record:
                                    has_count_field = True
                                    # 如果找到任何一个非零的 count 值，标记为 False
                                    if record[field] is not None and record[field] != 0:
                                        all_counts_zero = False
                                        break
                            if not all_counts_zero:
                                break
                        
                        # 如果存在 count 字段且所有值都为 0，抛弃查询
                        if has_count_field and all_counts_zero:
                            stats['failure_count'] += 1
                            stats['last_attempt_failed'] = True
                            logger.debug(f"查询结果的计数值全为0，抛弃查询: [{stats['template_type']}] {template.id}")
                            continue
                        
                        # 4. 如果答案数量超过阈值，抛弃这个查询
                        if len(filtered_answer) > max_answer_count:
                            stats['failure_count'] += 1
                            stats['last_attempt_failed'] = True
                            logger.debug(f"查询结果数量 ({len(filtered_answer)}) 超过阈值 ({max_answer_count})，抛弃查询: [{stats['template_type']}] {template.id}")
                            continue
                        
                        # 结果通过所有过滤条件，添加到结果列表
                        result = QueryResult(
                            template_id=template.id,
                            template_type=template.type,
                            query=query,
                            parameters_used=params_used,
                            answer=filtered_answer,
                            success=True,
                            error_message=None
                        )
                        
                        self.results.append(result)
                        stats['success_count'] += 1
                        stats['failure_count'] = 0  # 重置连续失败计数
                        stats['last_attempt_failed'] = False
                        
                        # 实时写入文件
                        if realtime_file:
                            # 格式化输出数据
                            template_type = result.template_type or "unknown"
                            template_id_with_prefix = f"{template_type}_{result.template_id}"
                            output_data = {
                                "template_id": template_id_with_prefix,
                                "template_type": result.template_type,
                                "query": result.query,
                                "parameters": result.parameters_used,
                                "answer": result.answer
                            }
                            # 如果不是第一个结果，先写逗号
                            if realtime_count > 0:
                                realtime_file.write(',\n')
                            # 写入 JSON 对象（使用 indent=2 保持格式一致）
                            json_str = json.dumps(output_data, ensure_ascii=False, indent=2, default=str)
                            # 将多行 JSON 的每行都缩进，使其在数组中正确对齐
                            # 跳过空行，只对非空行添加缩进
                            indented_lines = []
                            for line in json_str.split('\n'):
                                if line.strip():  # 非空行
                                    indented_lines.append('  ' + line)
                                else:  # 空行保持原样
                                    indented_lines.append(line)
                            indented_json = '\n'.join(indented_lines)
                            realtime_file.write(indented_json)
                            realtime_file.flush()  # 立即刷新到磁盘
                            realtime_count += 1
                        
                        logger.info(f"成功生成查询 [{len(self.results)}/{target}]: [{stats['template_type']}] {template.id} "
                                  f"(模版成功: {stats['success_count']}/{success_per_template}, 使用次数: {stats['usage_count']}, 结果数量: {len(filtered_answer)})")
                    else:
                        stats['failure_count'] += 1
                        stats['last_attempt_failed'] = True
                        logger.debug(f"查询失败: {error} (模版: [{stats['template_type']}] {template.id}, 连续失败: {stats['failure_count']})")
                
                # 完成当前模版
                if stats['success_count'] >= success_per_template:
                    logger.info(f"模版 [{stats['template_type']}] {template.id} 已完成，成功生成 {stats['success_count']} 个查询")
                elif stats['failure_count'] >= max_failures_per_template:
                    logger.warning(f"模版 [{stats['template_type']}] {template.id} 因连续失败过多而跳过，成功生成 {stats['success_count']} 个查询")
        finally:
            # 确保实时输出文件总是被正确关闭
            if realtime_file:
                try:
                    realtime_file.write('\n]')  # 结束 JSON 数组
                    realtime_file.close()
                    logger.info(f"实时输出完成，共写入 {realtime_count} 个查询到: {realtime_output_path}")
                except Exception as e:
                    logger.error(f"关闭实时输出文件时出错: {e}")
                    try:
                        realtime_file.close()
                    except:
                        pass
        
        # 输出统计信息
        logger.info(f"采样完成，成功生成 {len(self.results)} 个查询 (尝试 {attempts} 次)")
        
        # 统计成功和失败的模版
        successful_templates = [stats for stats in template_stats.values() if stats['success_count'] > 0]
        failed_templates = [stats for stats in template_stats.values() if stats['success_count'] == 0 and stats['usage_count'] > 0]
        unused_templates = [stats for stats in template_stats.values() if stats['usage_count'] == 0]
        
        logger.info(f"模版覆盖统计: 成功 {len(successful_templates)} 个, "
                   f"失败 {len(failed_templates)} 个, "
                   f"未使用 {len(unused_templates)} 个")
        
        logger.info("模版使用统计（按使用次数排序）:")
        for (template_type, template_id), stats in sorted(template_stats.items(), 
                                         key=lambda x: x[1]['usage_count'], 
                                         reverse=True):
            success_rate = (stats['success_count'] / stats['usage_count'] * 100) if stats['usage_count'] > 0 else 0.0
            logger.info(f"  [{template_type}] {template_id}: 使用 {stats['usage_count']} 次, "
                       f"成功 {stats['success_count']} 次 (成功率: {success_rate:.1f}%), "
                       f"连续失败 {stats['failure_count']} 次")
        
        # 如果有从未成功的模版，单独列出
        if failed_templates:
            logger.warning(f"以下 {len(failed_templates)} 个模版从未成功过:")
            for stats in sorted(failed_templates, key=lambda x: x['usage_count'], reverse=True):
                logger.warning(f"  [{stats['template_type']}] {stats['template_id']}: "
                             f"尝试 {stats['usage_count']} 次，全部失败")
        
        # 如果指定了统计信息输出路径，将统计信息写入文件
        if stats_output_path:
            self._export_stats_to_file(stats_output_path, template_stats, target, attempts, 
                                     successful_templates, failed_templates, unused_templates)
        
        return self.results
    
    def _export_stats_to_file(self, output_path: str, template_stats: Dict, target: int, 
                              attempts: int, successful_templates: List, 
                              failed_templates: List, unused_templates: List):
        """将统计信息导出到文件"""
        from datetime import datetime
        
        lines = []
        lines.append("=" * 80)
        lines.append("查询生成统计报告")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 总体统计
        lines.append("总体统计")
        lines.append("-" * 80)
        lines.append(f"目标查询数量: {target}")
        lines.append(f"成功生成查询数量: {len(self.results)}")
        lines.append(f"总尝试次数: {attempts}")
        if attempts > 0:
            success_rate = (len(self.results) / attempts) * 100
            lines.append(f"总体成功率: {success_rate:.2f}%")
        lines.append("")
        
        # 模版覆盖统计
        lines.append("模版覆盖统计")
        lines.append("-" * 80)
        lines.append(f"成功使用的模版数量: {len(successful_templates)}")
        lines.append(f"失败但尝试过的模版数量: {len(failed_templates)}")
        lines.append(f"未使用的模版数量: {len(unused_templates)}")
        lines.append(f"总模版数量: {len(template_stats)}")
        if len(template_stats) > 0:
            coverage_rate = (len(successful_templates) / len(template_stats)) * 100
            lines.append(f"模版覆盖率: {coverage_rate:.2f}%")
        lines.append("")
        
        # 详细模版使用统计（按使用次数排序）
        lines.append("详细模版使用统计（按使用次数排序）")
        lines.append("-" * 80)
        lines.append(f"{'模版类型':<20} {'模版ID':<30} {'使用次数':<10} {'成功次数':<10} {'成功率':<10} {'连续失败':<10}")
        lines.append("-" * 80)
        
        for (template_type, template_id), stats in sorted(template_stats.items(), 
                                         key=lambda x: x[1]['usage_count'], 
                                         reverse=True):
            success_rate = (stats['success_count'] / stats['usage_count'] * 100) if stats['usage_count'] > 0 else 0.0
            lines.append(f"{template_type:<20} {template_id:<30} {stats['usage_count']:<10} "
                        f"{stats['success_count']:<10} {success_rate:>8.1f}% {stats['failure_count']:<10}")
        lines.append("")
        
        # 从未成功的模版
        if failed_templates:
            lines.append(f"从未成功的模版（共 {len(failed_templates)} 个）")
            lines.append("-" * 80)
            lines.append(f"{'模版类型':<20} {'模版ID':<30} {'尝试次数':<10}")
            lines.append("-" * 80)
            for stats in sorted(failed_templates, key=lambda x: x['usage_count'], reverse=True):
                lines.append(f"{stats['template_type']:<20} {stats['template_id']:<30} {stats['usage_count']:<10}")
            lines.append("")
        
        # 未使用的模版
        if unused_templates:
            lines.append(f"未使用的模版（共 {len(unused_templates)} 个）")
            lines.append("-" * 80)
            lines.append(f"{'模版类型':<20} {'模版ID':<30}")
            lines.append("-" * 80)
            for stats in sorted(unused_templates, key=lambda x: (x['template_type'], x['template_id'])):
                lines.append(f"{stats['template_type']:<20} {stats['template_id']:<30}")
            lines.append("")
        
        # 按模版类型分组统计
        lines.append("按模版类型分组统计")
        lines.append("-" * 80)
        type_stats = {}
        for (template_type, template_id), stats in template_stats.items():
            if template_type not in type_stats:
                type_stats[template_type] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'unused': 0,
                    'total_usage': 0,
                    'total_success': 0
                }
            type_stats[template_type]['total'] += 1
            type_stats[template_type]['total_usage'] += stats['usage_count']
            type_stats[template_type]['total_success'] += stats['success_count']
            if stats['success_count'] > 0:
                type_stats[template_type]['successful'] += 1
            elif stats['usage_count'] > 0:
                type_stats[template_type]['failed'] += 1
            else:
                type_stats[template_type]['unused'] += 1
        
        lines.append(f"{'模版类型':<20} {'总数':<10} {'成功':<10} {'失败':<10} {'未使用':<10} {'总使用次数':<12} {'总成功次数':<12}")
        lines.append("-" * 80)
        for template_type in sorted(type_stats.keys()):
            stats = type_stats[template_type]
            lines.append(f"{template_type:<20} {stats['total']:<10} {stats['successful']:<10} "
                        f"{stats['failed']:<10} {stats['unused']:<10} {stats['total_usage']:<12} {stats['total_success']:<12}")
        lines.append("")
        
        lines.append("=" * 80)
        
        # 写入文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.info(f"统计信息已导出到: {output_path}")
        except Exception as e:
            logger.error(f"导出统计信息失败: {e}")
    
    def export_results(self, output_path: str):
        """导出结果到JSON文件"""
        output_data = []
        for r in self.results:
            # 在 template_id 前面加上 type 前缀
            template_type = r.template_type or "unknown"
            template_id_with_prefix = f"{template_type}_{r.template_id}"
            
            output_data.append({
                "template_id": template_id_with_prefix,
                "template_type": r.template_type,
                "query": r.query,
                "parameters": r.parameters_used,
                "answer": r.answer
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已导出到: {output_path}")
    
    def generate_judge_queries(self, template_file_path: str = None, max_unique_answers: int = 20) -> List[Dict[str, Any]]:
        """
        读取 template_judge1.json 中的模板，生成 template 和 anti_template 的查询，
        对比结果并找出独特答案（只存在于 template 中的和只存在于 anti_template 中的）
        
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
                
                # 尝试生成查询（可能需要多次尝试才能成功）
                max_attempts = 100
                template_query = None
                anti_template_query = None
                params_used = None
                
                for attempt in range(max_attempts):
                    try:
                        # 使用 QueryBuilder 构建 template 查询
                        query, params = self.builder.build_query(template_obj)
                        if query:
                            template_query = query
                            params_used = params
                            
                            # 创建 anti_template 对象并使用相同的参数构建查询
                            # 为了使用相同的参数，我们需要手动替换参数
                            anti_query = anti_template_str
                            replacements = {}
                            
                            for param_name, value in params.items():
                                # 替换模版中的参数
                                if param_name == 'VALUE' or param_name in ('VAL', 'FILTER_VAL', 'NODE_VALUE', 'START_VALUE', 'V','NEW_V' 'AID', 'BID', 'CID', 'ID', 'ID1', 'ID2', 'VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5'):
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
                                anti_query = self.builder._add_concept_id_filter(anti_query, params)
                            
                            anti_template_query = anti_query
                            break
                    except Exception as e:
                        logger.debug(f"尝试 {attempt + 1}/{max_attempts} 构建查询失败: {e}")
                        continue
                
                if not template_query or not anti_template_query:
                    logger.warning(f"模板 {template_id} 无法生成有效查询，跳过")
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
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("连接已关闭")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



