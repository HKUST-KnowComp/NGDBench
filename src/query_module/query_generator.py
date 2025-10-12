"""
查询生成器 - 统一的查询生成接口
"""

from typing import Any, Dict, List, Optional
import networkx as nx
from .read_queries import ReadQueryModule
from .update_queries import UpdateQueryModule


class QueryGenerator:
    """查询生成器 - 统一管理读查询和更新查询的生成"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化查询生成器
        
        Args:
            config: 查询生成配置
        """
        self.config = config
        self.read_config = config.get('read_queries', {})
        self.update_config = config.get('update_queries', {})
        
        # 初始化子模块
        self.read_module = ReadQueryModule(self.read_config)
        self.update_module = UpdateQueryModule(self.update_config)
        
        # 查询类型分布
        self.query_type_ratio = config.get('query_type_ratio', {
            'read': 0.7,
            'update': 0.3
        })
        
    def generate_all_queries(self, graph: nx.Graph) -> Dict[str, List[Dict[str, Any]]]:
        """
        生成所有类型的查询
        
        Args:
            graph: 输入图数据
            
        Returns:
            包含不同类型查询的字典
        """
        queries = {
            'read_queries': [],
            'update_queries': [],
            'benchmark_queries': [],
            'stress_test_queries': [],
            'consistency_test_queries': []
        }
        
        # 生成读查询
        if self.query_type_ratio.get('read', 0) > 0:
            queries['read_queries'] = self.read_module.generate_queries(graph)
            queries['benchmark_queries'].extend(
                self.read_module.generate_benchmark_queries(graph)
            )
        
        # 生成更新查询
        if self.query_type_ratio.get('update', 0) > 0:
            queries['update_queries'] = self.update_module.generate_queries(graph)
            queries['stress_test_queries'].extend(
                self.update_module.generate_stress_test_queries(graph)
            )
            queries['consistency_test_queries'].extend(
                self.update_module.generate_consistency_test_queries(graph)
            )
        
        return queries
    
    def generate_mixed_queries(self, graph: nx.Graph, total_queries: int = 50) -> List[Dict[str, Any]]:
        """
        生成混合类型的查询序列
        
        Args:
            graph: 输入图数据
            total_queries: 总查询数量
            
        Returns:
            混合查询列表
        """
        mixed_queries = []
        
        # 计算各类型查询数量
        num_read = int(total_queries * self.query_type_ratio.get('read', 0.7))
        num_update = total_queries - num_read
        
        # 生成读查询
        if num_read > 0:
            read_config = self.read_config.copy()
            read_config['num_queries'] = num_read
            read_module = ReadQueryModule(read_config)
            mixed_queries.extend(read_module.generate_queries(graph))
        
        # 生成更新查询
        if num_update > 0:
            update_config = self.update_config.copy()
            update_config['num_queries'] = num_update
            update_module = UpdateQueryModule(update_config)
            mixed_queries.extend(update_module.generate_queries(graph))
        
        # 随机打乱查询顺序
        import random
        random.shuffle(mixed_queries)
        
        # 添加序列号
        for i, query in enumerate(mixed_queries):
            query['sequence_id'] = i + 1
            query['total_queries'] = len(mixed_queries)
        
        return mixed_queries
    
    def generate_workload_queries(self, graph: nx.Graph, workload_type: str = 'balanced') -> List[Dict[str, Any]]:
        """
        生成特定工作负载的查询
        
        Args:
            graph: 输入图数据
            workload_type: 工作负载类型 ('read_heavy', 'write_heavy', 'balanced', 'analytical')
            
        Returns:
            工作负载查询列表
        """
        if workload_type == 'read_heavy':
            return self._generate_read_heavy_workload(graph)
        elif workload_type == 'write_heavy':
            return self._generate_write_heavy_workload(graph)
        elif workload_type == 'analytical':
            return self._generate_analytical_workload(graph)
        else:  # balanced
            return self._generate_balanced_workload(graph)
    
    def _generate_read_heavy_workload(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成读密集型工作负载"""
        config = self.read_config.copy()
        config.update({
            'num_queries': 40,
            'query_distribution': {
                'node_query': 0.25,
                'edge_query': 0.15,
                'path_query': 0.25,
                'subgraph_query': 0.20,
                'centrality_query': 0.10,
                'community_query': 0.05
            }
        })
        
        read_module = ReadQueryModule(config)
        queries = read_module.generate_queries(graph)
        
        # 添加少量更新查询
        update_config = self.update_config.copy()
        update_config['num_queries'] = 10
        update_module = UpdateQueryModule(update_config)
        queries.extend(update_module.generate_queries(graph))
        
        import random
        random.shuffle(queries)
        return queries
    
    def _generate_write_heavy_workload(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成写密集型工作负载"""
        config = self.update_config.copy()
        config.update({
            'num_queries': 35,
            'query_distribution': {
                'node_operations': 0.4,
                'edge_operations': 0.4,
                'attribute_operations': 0.2
            }
        })
        
        update_module = UpdateQueryModule(config)
        queries = update_module.generate_queries(graph)
        
        # 添加少量读查询
        read_config = self.read_config.copy()
        read_config['num_queries'] = 15
        read_module = ReadQueryModule(read_config)
        queries.extend(read_module.generate_queries(graph))
        
        import random
        random.shuffle(queries)
        return queries
    
    def _generate_analytical_workload(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成分析型工作负载"""
        config = self.read_config.copy()
        config.update({
            'num_queries': 30,
            'query_distribution': {
                'node_query': 0.1,
                'edge_query': 0.1,
                'path_query': 0.2,
                'subgraph_query': 0.3,
                'centrality_query': 0.2,
                'community_query': 0.1
            }
        })
        
        read_module = ReadQueryModule(config)
        queries = read_module.generate_queries(graph)
        
        # 添加基准查询
        queries.extend(read_module.generate_benchmark_queries(graph))
        
        return queries
    
    def _generate_balanced_workload(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成平衡型工作负载"""
        return self.generate_mixed_queries(graph, 50)
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """获取查询生成统计信息"""
        read_stats = self.read_module.get_query_statistics()
        update_stats = self.update_module.get_query_statistics()
        
        return {
            "read_queries": read_stats,
            "update_queries": update_stats,
            "total_queries": read_stats.get("total_queries", 0) + update_stats.get("total_queries", 0),
            "query_type_ratio": self.query_type_ratio
        }
    
    def validate_query_sequence(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证查询序列的有效性
        
        Args:
            queries: 查询序列
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # 检查查询格式
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                validation_result["errors"].append(f"查询 {i} 不是字典格式")
                validation_result["is_valid"] = False
                continue
            
            required_fields = ['type', 'parameters']
            for field in required_fields:
                if field not in query:
                    validation_result["errors"].append(f"查询 {i} 缺少必需字段: {field}")
                    validation_result["is_valid"] = False
        
        # 检查查询依赖关系
        self._check_query_dependencies(queries, validation_result)
        
        # 统计信息
        query_types = {}
        for query in queries:
            qtype = query.get('type', 'unknown')
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        validation_result["statistics"] = {
            "total_queries": len(queries),
            "query_types": query_types,
            "has_read_queries": any(q.get('operation') != 'update' for q in queries),
            "has_update_queries": any(q.get('operation') == 'update' for q in queries)
        }
        
        return validation_result
    
    def _check_query_dependencies(self, queries: List[Dict[str, Any]], validation_result: Dict[str, Any]):
        """检查查询间的依赖关系"""
        created_nodes = set()
        created_edges = set()
        removed_nodes = set()
        removed_edges = set()
        
        for i, query in enumerate(queries):
            query_type = query.get('type', '')
            operation = query.get('operation', '')
            params = query.get('parameters', {})
            
            if query_type == 'node_addition' and operation == 'add':
                node = params.get('node')
                if node in created_nodes:
                    validation_result["warnings"].append(f"查询 {i}: 尝试重复添加节点 {node}")
                created_nodes.add(node)
                
            elif query_type == 'node_removal' and operation == 'remove':
                node = params.get('node')
                if node in removed_nodes:
                    validation_result["warnings"].append(f"查询 {i}: 尝试重复删除节点 {node}")
                if node in created_nodes and node not in removed_nodes:
                    # 这是合理的：先创建后删除
                    pass
                removed_nodes.add(node)
                
            elif query_type == 'edge_addition' and operation == 'add':
                source, target = params.get('source'), params.get('target')
                edge = (source, target)
                if edge in created_edges:
                    validation_result["warnings"].append(f"查询 {i}: 尝试重复添加边 {edge}")
                if source in removed_nodes or target in removed_nodes:
                    validation_result["errors"].append(f"查询 {i}: 尝试在已删除的节点间添加边")
                    validation_result["is_valid"] = False
                created_edges.add(edge)
                
            elif query_type == 'edge_removal' and operation == 'remove':
                source, target = params.get('source'), params.get('target')
                edge = (source, target)
                if edge in removed_edges:
                    validation_result["warnings"].append(f"查询 {i}: 尝试重复删除边 {edge}")
                removed_edges.add(edge)
