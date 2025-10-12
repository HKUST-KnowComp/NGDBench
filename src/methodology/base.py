"""
算法执行基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import networkx as nx
import time


class BaseMethodology(ABC):
    """算法执行基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化算法执行器
        
        Args:
            config: 算法配置参数
        """
        self.config = config
        self.algorithm_name = config.get('algorithm_name', 'unknown')
        self.algorithm_params = config.get('algorithm_params', {})
        self._execution_history = []
        
    @abstractmethod
    def execute(self, graph: nx.Graph, queries: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        执行算法
        
        Args:
            graph: 输入图数据
            queries: 查询列表（可选）
            
        Returns:
            算法执行结果
        """
        pass
    
    @abstractmethod
    def validate_input(self, graph: nx.Graph) -> bool:
        """
        验证输入数据的有效性
        
        Args:
            graph: 输入图数据
            
        Returns:
            验证结果
        """
        pass
    
    def preprocess_graph(self, graph: nx.Graph) -> nx.Graph:
        """
        预处理图数据
        
        Args:
            graph: 原始图数据
            
        Returns:
            预处理后的图数据
        """
        # 默认不进行预处理，子类可以重写
        return graph.copy()
    
    def postprocess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理算法结果
        
        Args:
            results: 原始算法结果
            
        Returns:
            后处理后的结果
        """
        # 默认不进行后处理，子类可以重写
        return results
    
    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        获取执行元数据
        
        Returns:
            执行元数据
        """
        return {
            "algorithm_name": self.algorithm_name,
            "algorithm_params": self.algorithm_params,
            "execution_count": len(self._execution_history),
            "last_execution": self._execution_history[-1] if self._execution_history else None
        }
    
    def _record_execution(self, graph: nx.Graph, results: Dict[str, Any], 
                         execution_time: float, queries: Optional[List[Dict]] = None):
        """
        记录执行信息
        
        Args:
            graph: 输入图数据
            results: 执行结果
            execution_time: 执行时间
            queries: 查询列表
        """
        execution_record = {
            "timestamp": time.time(),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "execution_time": execution_time,
            "num_queries": len(queries) if queries else 0,
            "results_keys": list(results.keys()) if isinstance(results, dict) else [],
            "success": True
        }
        self._execution_history.append(execution_record)
    
    def _record_failure(self, graph: nx.Graph, error: Exception, 
                       queries: Optional[List[Dict]] = None):
        """
        记录执行失败信息
        
        Args:
            graph: 输入图数据
            error: 异常信息
            queries: 查询列表
        """
        failure_record = {
            "timestamp": time.time(),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "num_queries": len(queries) if queries else 0,
            "error": str(error),
            "error_type": type(error).__name__,
            "success": False
        }
        self._execution_history.append(failure_record)
    
    def get_supported_query_types(self) -> List[str]:
        """
        获取支持的查询类型
        
        Returns:
            支持的查询类型列表
        """
        return []
    
    def is_query_supported(self, query: Dict[str, Any]) -> bool:
        """
        检查查询是否被支持
        
        Args:
            query: 查询对象
            
        Returns:
            是否支持该查询
        """
        query_type = query.get('type', '')
        return query_type in self.get_supported_query_types()
    
    def execute_with_timing(self, graph: nx.Graph, queries: Optional[List[Dict]] = None) -> Tuple[Dict[str, Any], float]:
        """
        执行算法并记录时间
        
        Args:
            graph: 输入图数据
            queries: 查询列表
            
        Returns:
            (算法结果, 执行时间)
        """
        start_time = time.time()
        try:
            results = self.execute(graph, queries)
            execution_time = time.time() - start_time
            self._record_execution(graph, results, execution_time, queries)
            return results, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(graph, e, queries)
            raise e
