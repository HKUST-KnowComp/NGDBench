"""
数据源基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import networkx as nx


class BaseDataSource(ABC):
    """数据源基类，定义数据获取接口"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据源
        
        Args:
            config: 数据源配置参数
        """
        self.config = config or {}
        self._graph = None
        self._metadata = {}
    
    @abstractmethod
    def load_data(self) -> nx.Graph:
        """
        加载数据并返回图对象
        
        Returns:
            NetworkX图对象
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据集元信息
        
        Returns:
            包含数据集信息的字典
        """
        pass
    
    def validate_data(self, graph: nx.Graph) -> bool:
        """
        验证数据完整性
        
        Args:
            graph: 待验证的图对象
            
        Returns:
            验证结果
        """
        if graph is None or graph.number_of_nodes() == 0:
            return False
        return True
    
    @property
    def graph(self) -> nx.Graph:
        """获取已加载的图对象"""
        if self._graph is None:
            self._graph = self.load_data()
        return self._graph
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """获取数据集元信息"""
        if not self._metadata:
            self._metadata = self.get_metadata()
        return self._metadata
