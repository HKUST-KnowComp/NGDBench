"""
查询模块基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import networkx as nx


class BaseQueryModule(ABC):
    """查询模块基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化查询模块
        
        Args:
            config: 查询配置参数
        """
        self.config = config
        self.query_type = config.get('query_type', 'read')  # 'read', 'update'
        self._generated_queries = []
        
    @abstractmethod
    def generate_queries(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """
        生成查询集合
        
        Args:
            graph: 输入图数据
            
        Returns:
            查询列表
        """
        pass
    
    @abstractmethod
    def validate_query(self, query: Dict[str, Any]) -> bool:
        """
        验证查询的有效性
        
        Args:
            query: 查询对象
            
        Returns:
            验证结果
        """
        pass
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        获取查询统计信息
        
        Returns:
            查询统计信息
        """
        if not self._generated_queries:
            return {"total_queries": 0}
        
        query_types = {}
        for query in self._generated_queries:
            qtype = query.get('type', 'unknown')
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        return {
            "total_queries": len(self._generated_queries),
            "query_types": query_types,
            "query_type_distribution": {k: v/len(self._generated_queries) 
                                     for k, v in query_types.items()}
        }
    
    def filter_queries(self, queries: List[Dict[str, Any]], 
                      criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据条件过滤查询
        
        Args:
            queries: 查询列表
            criteria: 过滤条件
            
        Returns:
            过滤后的查询列表
        """
        filtered = []
        
        for query in queries:
            match = True
            
            # 按类型过滤
            if 'type' in criteria and query.get('type') != criteria['type']:
                match = False
            
            # 按复杂度过滤
            if 'complexity' in criteria:
                query_complexity = query.get('complexity', 'medium')
                if query_complexity != criteria['complexity']:
                    match = False
            
            # 按参数过滤
            if 'has_parameters' in criteria:
                has_params = bool(query.get('parameters'))
                if has_params != criteria['has_parameters']:
                    match = False
            
            if match:
                filtered.append(query)
        
        return filtered
    
    def get_supported_query_types(self) -> List[str]:
        """
        获取支持的查询类型
        
        Returns:
            支持的查询类型列表
        """
        return []
    
    def estimate_query_complexity(self, query: Dict[str, Any], graph: nx.Graph) -> str:
        """
        估算查询复杂度
        
        Args:
            query: 查询对象
            graph: 图数据
            
        Returns:
            复杂度等级 ('low', 'medium', 'high')
        """
        query_type = query.get('type', '')
        
        # 基于查询类型的简单复杂度估算
        high_complexity_types = {
            'all_pairs_shortest_path', 'community_detection', 
            'subgraph_isomorphism', 'maximum_flow'
        }
        
        medium_complexity_types = {
            'shortest_path', 'centrality_calculation', 
            'neighborhood_query', 'pattern_matching'
        }
        
        if query_type in high_complexity_types:
            return 'high'
        elif query_type in medium_complexity_types:
            return 'medium'
        else:
            return 'low'
    
    def create_query_template(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """
        创建查询模板
        
        Args:
            query_type: 查询类型
            **kwargs: 查询参数
            
        Returns:
            查询模板
        """
        template = {
            "type": query_type,
            "parameters": kwargs,
            "timestamp": None,
            "complexity": "medium",
            "expected_result_type": "unknown"
        }
        
        return template
