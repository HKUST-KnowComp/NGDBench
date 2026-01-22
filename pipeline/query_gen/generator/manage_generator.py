"""
Neo4j Management Query Generator (增删改操作查询生成器)
针对增删改操作生成查询，包括前置验证、操作执行和后置验证


"""

import json
import random
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from neo4j import GraphDatabase

# 导入 query_generator 中的相关类
from .query_generator import (
    SchemaAnalyzer,
    QueryBuilder,
    QueryExecutor,
    QueryResult,
    DEFAULT_EXCLUDED_RETURN_PROPS,
    DEFAULT_EXCLUDED_LABELS,
    logger
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ManagementTemplate:
    """管理操作模板类"""
    operation: str  # 操作类型：CREATE, MERGE, DELETE, SET, FOREACH
    difficulty: int
    title: str
    pre_validation: str  # 前置验证查询
    template: str  # 操作查询
    post_validation: str  # 后置验证查询
    parameters: Dict[str, str]  # 参数定义
    example: str


@dataclass
class ManagementQueryResult:
    """管理操作查询结果"""
    operation: str
    difficulty: int
    title: str
    template_id: str  # 基于 operation 和 difficulty 生成
    
    # 前置验证结果
    pre_validation_query: str
    pre_validation_params: Dict[str, Any]
    pre_validation_answer: List[Dict]
    pre_validation_success: bool
    
    # 操作执行结果
    template_query: str
    template_params: Dict[str, Any]
    template_success: bool
    
    # 后置验证结果
    post_validation_query: str
    post_validation_params: Dict[str, Any]
    post_validation_answer: List[Dict]
    post_validation_success: bool
    
    # 整体是否成功（所有三个查询都成功）
    overall_success: bool = False
    
    # 错误信息（有默认值的字段放在最后）
    pre_validation_error: Optional[str] = None
    template_error: Optional[str] = None
    post_validation_error: Optional[str] = None


class ManagementTemplateLoader:
    """管理操作模板加载器"""
    
    def __init__(self, template_path: str):
        self.template_path = template_path
        self.templates: List[ManagementTemplate] = []
    
    def load(self):
        """加载模板文件"""
        logger.info(f"加载管理操作模板文件: {self.template_path}")
        
        with open(self.template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析 JSON 结构
        for operation_group in data:
            operation = operation_group.get('operation', 'UNKNOWN')
            templates_list = operation_group.get('templates', [])
            
            for template_data in templates_list:
                template = ManagementTemplate(
                    operation=operation,
                    difficulty=template_data.get('difficulty', 1),
                    title=template_data.get('title', ''),
                    pre_validation=template_data.get('pre_validation', ''),
                    template=template_data.get('template', ''),
                    post_validation=template_data.get('post_validation', ''),
                    parameters=template_data.get('parameters', {}),
                    example=template_data.get('example', '')
                )
                self.templates.append(template)
        
        logger.info(f"加载了 {len(self.templates)} 个管理操作模板")
    
    def get_templates_by_operation(self, operation: str) -> List[ManagementTemplate]:
        """根据操作类型获取模板"""
        return [t for t in self.templates if t.operation == operation]
    
    def get_all_templates(self) -> List[ManagementTemplate]:
        """获取所有模板"""
        return self.templates


class ManagementQueryBuilder:
    """管理操作查询构建器：根据模板和schema生成具体查询"""
    
    def __init__(
        self,
        schema: SchemaAnalyzer,
        excluded_return_props: Optional[Set[str]] = None,
        excluded_labels: Optional[Set[str]] = None,
        dataset: Optional[str] = None,
        driver = None,
    ):
        self.schema = schema
        self.excluded_return_props: Set[str] = excluded_return_props or set()
        self.excluded_labels: Set[str] = excluded_labels or set()
        self.dataset = dataset
        
        # 复用 QueryBuilder 的逻辑
        self.query_builder = QueryBuilder(
            schema,
            excluded_return_props=excluded_return_props,
            excluded_labels=excluded_labels,
            dataset=dataset,
            driver=driver  # 传递 driver 用于节点采样
        )
    
    def build_queries(
        self, 
        template: ManagementTemplate
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
        """
        根据模板构建三个查询（pre_validation, template, post_validation）
        
        Returns:
            (pre_validation_query, template_query, post_validation_query, params_used)
            如果构建失败，返回的查询为 None
        """
        # 创建一个临时的 Template 对象用于参数生成
        from .query_generator import Template
        
        # 检查 pre_validation 和 post_validation 是否包含聚合函数
        # 如果包含，需要创建一个包含聚合函数的模板，以便正确过滤ID属性
        validation_queries = template.pre_validation + " " + template.post_validation
        has_aggregate = any(func in validation_queries.upper() 
                          for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'COLLECT'])
        
        # 检查是否包含需要数值属性的聚合函数（AVG, SUM）
        has_numeric_aggregate = any(func in validation_queries.upper() 
                                  for func in ['AVG', 'SUM'])
        
        # 使用 template 字段来生成参数（因为三个查询共享参数）
        # 如果验证查询包含聚合函数，将验证查询合并到模板字符串中，
        # 这样 _is_aggregate_query 方法就能通过检查模板字符串识别出聚合查询
        template_str = template.template
        if has_aggregate:
            # 将验证查询合并到模板字符串中，以便 _is_aggregate_query 能识别
            template_str = template.template + " " + validation_queries
        
        temp_template = Template(
            id=f"{template.operation}_{template.difficulty}",
            template=template_str,  # 如果包含聚合函数，这里会包含验证查询
            parameters=template.parameters,
            required_numeric_props=has_numeric_aggregate,  # 如果包含AVG或SUM，需要数值属性
            type=template.operation
        )
        
        # 首先尝试从节点采样来填充相关参数
        params_used = {}
        
        # 识别需要从同一节点获取的参数组
        node_param_groups = self.query_builder._identify_node_param_groups(temp_template)
        
        # 为每个节点参数组采样节点并填充参数
        for group in node_param_groups:
            if not self.query_builder._fill_params_from_sampled_node(temp_template, group, params_used):
                # 如果采样失败，回退到原来的方法
                logger.debug(f"节点采样失败，回退到原方法: {group}")
        
        # 生成剩余参数
        for param_name, param_type in template.parameters.items():
            # 如果参数已经填充，跳过
            if param_name in params_used:
                continue
            
            # 处理 list 类型参数
            if param_type == "list":
                value = self._generate_list_param(param_name, template, params_used)
            else:
                value = self.query_builder._generate_param_value(
                    param_name, 
                    temp_template, 
                    params_used
                )
            if value is None:
                logger.warning(f"无法生成参数 {param_name}，跳过该模板")
                return None, None, None, {}
            params_used[param_name] = value
        
        # 替换三个查询中的参数
        pre_validation_query = self._replace_parameters(
            template.pre_validation, 
            params_used
        )
        template_query = self._replace_parameters(
            template.template, 
            params_used
        )
        post_validation_query = self._replace_parameters(
            template.post_validation, 
            params_used
        )
        
        return pre_validation_query, template_query, post_validation_query, params_used
    
    def _generate_list_param(self, param_name: str, template: ManagementTemplate, 
                             current_params: Dict[str, Any]) -> Optional[List[Any]]:
        """生成列表类型参数的值"""
        # 根据参数名推断列表元素的类型
        if param_name in ('LIST', 'IDS', 'A_IDS', 'B_IDS'):
            # 这些参数通常是 ID 列表，从数据库中采样一些 ID 值
            # 尝试从已生成的参数中获取标签信息
            label = None
            for label_key in ['LABEL', 'L1', 'L2', 'L', 'GL', 'SL', 'EL']:
                if label_key in current_params:
                    label = current_params[label_key]
                    break
            
            if label and label in self.schema.labels:
                # 从该标签的节点中采样一些 id 值
                label_info = self.schema.labels[label]
                if 'id' in label_info.properties:
                    id_prop = label_info.properties['id']
                    if id_prop.sample_values:
                        # 随机选择 3-5 个 ID 值
                        available_count = len(id_prop.sample_values)
                        if available_count > 0:
                            sample_size = min(random.randint(3, 5), available_count)
                            if sample_size == available_count:
                                return id_prop.sample_values.copy()
                            else:
                                return random.sample(id_prop.sample_values, sample_size)
            
            # 如果没有找到，生成一些默认值
            # 根据参数名判断是字符串列表还是数字列表
            if param_name == 'LIST':
                # LIST 通常用于 FOREACH，可能是数字或字符串列表
                # 生成简单的数字列表
                return [1, 2, 3]
            else:
                # IDS 通常是字符串 ID 列表
                return ['id1', 'id2', 'id3']
        
        # 默认返回空列表
        logger.warning(f"无法生成列表参数 {param_name}，使用默认值")
        return []
    
    def _replace_parameters(self, query_template: str, params: Dict[str, Any]) -> str:
        """替换查询模板中的参数"""
        query = query_template
        replacements = {}
        
        # 不需要加引号的参数名（标签、属性名、关系类型等）
        no_quote_params = {'LABEL', 'LABEL1', 'LABEL2', 'LABEL3', 'LABEL4', 
                          'START_LABEL', 'END_LABEL', 'GROUP_LABEL',
                          'L', 'L1', 'L2', 'L3', 'L4', 'SL', 'EL', 'RL', 'GL',
                          'PROP', 'PROP1', 'PROP2', 'PROP_ID', 'PROP_ID1', 'PROP_ID2',
                          'GROUP_PROP', 'FILTER_PROP', 'NODE_PROP', 'START_PROP',
                          'P', 'P1', 'P2', 'SP', 'NP', 'BP', 'GP', 'RP',
                          'REL', 'REL_TYPE', 'REL1', 'REL2', 'REL3', 'R', 'R1', 'R2', 'R3',
                          'REL_PROP',  # 关系属性名，不需要引号
                          'MIN_HOPS', 'MAX_HOPS', 'D1', 'D2', 'D3', 'OP', 'OP1', 'OP2',
                          'AGG_FUNC', 'AGG_FUNC1', 'AGG_FUNC2', 'NUM_PROP', 'NUM_PROP1', 'NUM_PROP2'}
        
        # 列表参数名
        list_params = {'LIST', 'IDS', 'A_IDS', 'B_IDS'}
        
        def escape_string(s: str) -> str:
            """转义字符串中的单引号"""
            return s.replace("'", "\\'")
        
        for param_name, value in params.items():
            # 替换 $PARAM_NAME 格式的参数
            if param_name in list_params and isinstance(value, list):
                # 列表参数：转换为 Cypher 列表格式
                if value and isinstance(value[0], str):
                    # 字符串列表：转义单引号并加引号
                    list_str = ', '.join(f"'{escape_string(str(v))}'" for v in value)
                else:
                    # 数字列表或其他类型
                    list_str = ', '.join(str(v) for v in value)
                replacement = f"[{list_str}]"
            elif param_name in no_quote_params:
                # 标签、属性名、关系类型等：直接替换，不需要引号
                replacement = str(value)
            elif isinstance(value, str):
                # 字符串值：转义单引号并加引号
                replacement = f"'{escape_string(value)}'"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # 数值类型：直接替换，不需要引号
                replacement = str(value)
            elif isinstance(value, bool):
                # 布尔值：直接使用 true/false（Cypher关键字）
                replacement = str(value).lower()
            else:
                # 其他类型：转换为字符串
                replacement = str(value)
            
            replacements[f'${param_name}'] = replacement
        
        # 按照键长度倒序替换，避免短键被先替换
        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
        for key in sorted_keys:
            query = query.replace(key, replacements[key])
        
        # 修复量化路径模式的语法：Neo4j 5 使用逗号而不是两个点
        # 将 {数字..数字} 替换为 {数字,数字}
        # 注意：只替换花括号内的，不替换 *数字..数字 这种可变长度关系语法
        # 匹配 {数字..数字} 模式（量化路径模式）
        query = re.sub(r'\{(\d+)\.\.(\d+)\}', r'{\1,\2}', query)
        
        # 对于 dataset 为 "mcp" 或 "multi_fin" 的情况，为 concept 节点添加 id 属性过滤
        if self.dataset in ("mcp", "multi_fin"):
            query = self.query_builder._add_concept_id_filter(query, params)
        
        return query


class ManageGenerator:
    """管理操作查询生成器主类"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        template_path: str = "query_template/template_managemet.json",
        exclude_internal_id_as_return: bool = True,
        dataset: Optional[str] = None
    ):
        """
        初始化管理操作查询生成器
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            template_path: 模板文件路径
            exclude_internal_id_as_return: 是否在返回属性中排除内部ID字段
            dataset: 数据集名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.template_path = template_path
        self.exclude_internal_id_as_return = exclude_internal_id_as_return
        self.dataset = dataset
        
        self.excluded_return_props: Set[str] = (
            set(DEFAULT_EXCLUDED_RETURN_PROPS) if exclude_internal_id_as_return else set()
        )
        
        # 初始化组件
        self.driver = None
        self.schema = None
        self.template_loader = None
        self.builder = None
        self.executor = None
        
        # 结果存储
        self.results: List[ManagementQueryResult] = []
    
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
        
        # 加载模板
        self.template_loader = ManagementTemplateLoader(self.template_path)
        self.template_loader.load()
        
        # 初始化其他组件
        excluded_labels = set(DEFAULT_EXCLUDED_LABELS)
        if self.dataset in ("mcp", "multi_fin"):
            excluded_labels.update({"passage"})
        
        self.builder = ManagementQueryBuilder(
            self.schema,
            excluded_return_props=self.excluded_return_props,
            excluded_labels=excluded_labels,
            dataset=self.dataset,
            driver=self.driver  # 传递 driver 用于节点采样
        )
        self.executor = QueryExecutor(self.driver)
    
    def generate_samples(
        self,
        target_count: Optional[int] = None,
        max_attempts_multiplier: int = 10,
        max_failures_per_template: int = 100,
        operations: Optional[List[str]] = None,
        success_per_template: int = 5
    ) -> List[ManagementQueryResult]:
        """
        生成管理操作查询样本
        
        Args:
            target_count: 目标样本数量，默认为边数/16
            max_attempts_multiplier: 最大尝试次数倍数
            max_failures_per_template: 每个模板的最大连续失败次数
            operations: 要生成的操作类型列表，None表示所有操作
            success_per_template: 每个模板需要生成的成功查询数量
        
        Returns:
            ManagementQueryResult列表
        """
        if not self.schema:
            self.initialize()
        
        # 获取目标数量
        if target_count is None:
            target_count = max(1, self.schema.total_edges // 16)
        
        max_attempts = target_count * max_attempts_multiplier
        
        logger.info(f"开始生成管理操作查询，目标数量: {target_count}, 最大尝试次数: {max_attempts}")
        
        # 获取模板
        all_templates = self.template_loader.get_all_templates()
        if operations:
            all_templates = [t for t in all_templates if t.operation in operations]
        
        if not all_templates:
            logger.error("没有可用的模板")
            return []
        
        logger.info(f"可用模板数量: {len(all_templates)}")
        
        self.results = []
        attempts = 0
        
        # 跟踪每个模板的使用情况
        template_stats = {}
        for template in all_templates:
            template_id = f"{template.operation}_{template.difficulty}"
            template_stats[template_id] = {
                'template': template,
                'success_count': 0,
                'failure_count': 0,
                'usage_count': 0
            }
        
        # 按顺序遍历每个模板
        for template in all_templates:
            # 如果已达到目标数量或超过最大尝试次数，停止
            if len(self.results) >= target_count or attempts >= max_attempts:
                break
            
            template_id = f"{template.operation}_{template.difficulty}"
            stats = template_stats[template_id]
            
            logger.info(f"开始处理模板 [{template.operation}] difficulty={template.difficulty}: {template.title}")
            
            # 对当前模板连续采样
            while stats['success_count'] < success_per_template:
                # 检查是否达到全局限制
                if len(self.results) >= target_count or attempts >= max_attempts:
                    break
                
                # 检查是否超过最大失败次数
                if stats['failure_count'] >= max_failures_per_template:
                    logger.warning(f"模板 {template_id} 连续失败 {stats['failure_count']} 次，跳过该模板")
                    break
                
                attempts += 1
                stats['usage_count'] += 1
                
                # 构建查询
                pre_query, template_query, post_query, params_used = self.builder.build_queries(template)
                
                if not pre_query or not template_query or not post_query:
                    stats['failure_count'] += 1
                    logger.debug(f"构建查询失败: {template_id}")
                    continue
                
                # 依次执行三个查询
                # 1. 执行前置验证（允许结果为空，因为COUNT等聚合函数可能返回0）
                pre_success, pre_answer, pre_error = self.executor.execute(pre_query, allow_empty=True)
                
                # 2. 执行操作（CREATE/DELETE/SET等操作可能不返回结果，允许空结果）
                template_success, _, template_error = self.executor.execute(template_query, allow_empty=True)
                
                # 3. 执行后置验证（允许结果为空，因为COUNT等聚合函数可能返回0）
                post_success, post_answer, post_error = self.executor.execute(post_query, allow_empty=True)
                
                # 判断整体是否成功
                overall_success = pre_success and template_success and post_success
                
                if overall_success:
                    # 创建结果对象
                    result = ManagementQueryResult(
                        operation=template.operation,
                        difficulty=template.difficulty,
                        title=template.title,
                        template_id=template_id,
                        pre_validation_query=pre_query,
                        pre_validation_params=params_used,
                        pre_validation_answer=pre_answer,
                        pre_validation_success=pre_success,
                        pre_validation_error=pre_error,
                        template_query=template_query,
                        template_params=params_used,
                        template_success=template_success,
                        template_error=template_error,
                        post_validation_query=post_query,
                        post_validation_params=params_used,
                        post_validation_answer=post_answer,
                        post_validation_success=post_success,
                        post_validation_error=post_error,
                        overall_success=overall_success
                    )
                    
                    self.results.append(result)
                    stats['success_count'] += 1
                    stats['failure_count'] = 0
                    
                    logger.info(f"成功生成查询 [{len(self.results)}/{target_count}]: "
                              f"[{template.operation}] difficulty={template.difficulty} "
                              f"(模板成功: {stats['success_count']}/{success_per_template})")
                else:
                    stats['failure_count'] += 1
                    # 输出详细的失败信息，帮助调试
                    failure_reasons = []
                    if not pre_success:
                        failure_reasons.append(f"pre_validation失败: {pre_error}")
                    if not template_success:
                        failure_reasons.append(f"template失败: {template_error}")
                    if not post_success:
                        failure_reasons.append(f"post_validation失败: {post_error}")
                    
                    logger.warning(f"查询执行失败 [{template_id}]: {', '.join(failure_reasons)}")
            
            # 完成当前模板
            if stats['success_count'] >= success_per_template:
                logger.info(f"模板 {template_id} 已完成，成功生成 {stats['success_count']} 个查询")
            elif stats['failure_count'] >= max_failures_per_template:
                logger.warning(f"模板 {template_id} 因连续失败过多而跳过，成功生成 {stats['success_count']} 个查询")
        
        logger.info(f"生成完成，成功生成 {len(self.results)} 个管理操作查询 (尝试 {attempts} 次)")
        
        # 输出统计信息
        successful_templates = [s for s in template_stats.values() if s['success_count'] > 0]
        failed_templates = [s for s in template_stats.values() if s['success_count'] == 0 and s['usage_count'] > 0]
        
        logger.info(f"模板覆盖统计: 成功 {len(successful_templates)} 个, 失败 {len(failed_templates)} 个")
        
        return self.results
    
    def export_results(self, output_path: str):
        """导出结果到JSON文件"""
        output_data = []
        for r in self.results:
            output_data.append({
                "operation": r.operation,
                "difficulty": r.difficulty,
                "title": r.title,
                "template_id": r.template_id,
                "pre_validation": {
                    "query": r.pre_validation_query,
                    "parameters": r.pre_validation_params,
                    "answer": r.pre_validation_answer,
                    "success": r.pre_validation_success,
                    "error": r.pre_validation_error
                },
                "template": {
                    "query": r.template_query,
                    "parameters": r.template_params,
                    "success": r.template_success,
                    "error": r.template_error
                },
                "post_validation": {
                    "query": r.post_validation_query,
                    "parameters": r.post_validation_params,
                    "answer": r.post_validation_answer,
                    "success": r.post_validation_success,
                    "error": r.post_validation_error
                },
                "overall_success": r.overall_success
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已导出到: {output_path}")
    
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
