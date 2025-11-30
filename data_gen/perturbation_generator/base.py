"""
扰动生成器基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import copy
import os
import shutil
import gzip
import tempfile

class BasePerturbationGenerator(ABC):
    """扰动生成器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_config = config.get('perturbation', {})
        self.data_config = config.get('data_source', {})
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
        
    @abstractmethod
    def apply_perturbation(self):
        """
        Apply perturbation to the dataset and 
        return the perturbation information.
        """
        pass

    @abstractmethod
    def incomplete_perturb(self, perturbed_data_path: str) -> tuple:
        """
        不完整性扰动
        
        Args:
            perturbed_data_path: 扰动后的数据路径
            
        """
        pass
    
    @abstractmethod
    def noise_perturb(self, perturbed_data_path: str) -> tuple:
        """
        噪声扰动
        
        Args:
            perturbed_data_path: 扰动后的数据路径
        """
        pass

    def create_groundtruth(self, graph: nx.Graph) -> nx.Graph:
        """
        创建真实基准数据（原始数据的副本）
        
        Args:
            graph: 原始图数据
            
        Returns:
            真实基准图数据
        """
        return copy.deepcopy(graph)
    
    def get_perturbation_stats(self, original: nx.Graph, perturbed: nx.Graph) -> Dict[str, Any]:
        """
        计算扰动统计信息
        
        Args:
            original: 原始图
            perturbed: 扰动后的图
            
        Returns:
            扰动统计信息
        """
        return {
            "original_nodes": original.number_of_nodes(),
            "original_edges": original.number_of_edges(),
            "perturbed_nodes": perturbed.number_of_nodes(),
            "perturbed_edges": perturbed.number_of_edges(),
            "nodes_removed": original.number_of_nodes() - perturbed.number_of_nodes(),
            "edges_removed": original.number_of_edges() - perturbed.number_of_edges(),
            "perturbation_ratio_nodes": 1 - (perturbed.number_of_nodes() / original.number_of_nodes()),
            "perturbation_ratio_edges": 1 - (perturbed.number_of_edges() / original.number_of_edges())
        }
    
    def validate_perturbation(self, original: nx.Graph, perturbed: nx.Graph) -> bool:
        """
        验证扰动结果的有效性
        
        Args:
            original: 原始图
            perturbed: 扰动后的图
            
        Returns:
            验证结果
        """
        # 基本验证：扰动后的图不应该比原图更大
        if (perturbed.number_of_nodes() > original.number_of_nodes() or 
            perturbed.number_of_edges() > original.number_of_edges()):
            return False
        
        # 扰动后的图应该仍然是有效的图结构
        if perturbed.number_of_nodes() == 0:
            return False
            
        return True

    def _copy_dataset(self) -> str:
        """
        Copy the dataset to the perturbed directory 
        and return the perturbed directory path.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        
        root_data_path = self.data_config.get("root_data_path", "gnd_dataset/ldbc_snb_bi/out-sf1/graphs/csv/bi/composite-projected-fk/initial_snapshot")
        
        # Extract dataset name from path after gnd_dataset/
        dataset_name = root_data_path.split("gnd_dataset/")[1].split("/")[0]
        
        base_path = root_data_path.replace("gnd_dataset", "perturbed_dataset")
        perturbed_root_path = base_path.replace(dataset_name, f"{dataset_name}_{timestamp}")
        
        if os.path.exists(perturbed_root_path):
            return perturbed_root_path
        else:
            os.makedirs(os.path.dirname(perturbed_root_path), exist_ok=True)
            shutil.copytree(root_data_path, perturbed_root_path)
            return perturbed_root_path
    
    def _process_directory(self, dir_path: str, data_file_format: str, perturb_type: str) -> List[Dict]:
        """
        递归处理目录中的所有文件（通用方法）
        
        Args:
            dir_path: 目录路径
            perturbation_config: 扰动配置
            data_file_format: 文件格式
            perturb_type: 扰动类型 ('incompleteness' 或 'noise')
            
        Returns:
            List[Dict]: 操作记录列表
        """
        operations = []
        
        if not os.path.exists(dir_path):
            return operations
        
        for item_name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item_name)
            
            # Recursively process the directory.
            if os.path.isdir(item_path):
                sub_operations = self._process_directory(item_path, data_file_format, perturb_type)
                operations.extend(sub_operations)
            # 如果是文件，根据格式处理
            elif os.path.isfile(item_path):
                if self._is_target_file(item_name, data_file_format):
                    file_operations = self._process_file(item_path, item_name, data_file_format, perturb_type)
                    operations.extend(file_operations)
        
        return operations
    
    def _is_target_file(self, filename: str, data_file_format: str) -> bool:
        """
        判断文件是否为目标格式
        
        Args:
            filename: 文件名
            data_file_format: 目标文件格式
            
        Returns:
            bool: 是否为目标文件
        """
        return filename.endswith(data_file_format)
    
    def _read_file(self, file_path: str, data_file_format: str) -> pd.DataFrame:
        """
        根据文件格式读取文件
        
        Args:
            file_path: 文件路径
            data_file_format: 文件格式
            
        Returns:
            pd.DataFrame: 读取的数据
        """
        try:
            file_size = os.path.getsize(file_path)
            min_size = 20 if data_file_format.endswith(".gz") else 1
            if file_size < min_size:
                print(f"文件内容为空，跳过: {file_path} (大小: {file_size} 字节)")
                return None
                
            if data_file_format == ".csv.gz":
                # 压缩的CSV文件
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return pd.read_csv(f,sep="|")
            elif data_file_format == ".csv":
                # 普通CSV文件
                return pd.read_csv(file_path,sep="|")
            elif data_file_format in [".parquet", ".pq"]:
                # Parquet文件
                return pd.read_parquet(file_path)
            elif data_file_format == ".json":
                # JSON文件
                return pd.read_json(file_path)
            elif data_file_format in [".tsv", ".tsv.gz"]:
                # TSV文件
                if data_file_format.endswith(".gz"):
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        return pd.read_csv(f, sep='\t')
                else:
                    return pd.read_csv(file_path, sep='\t')
            else:
                # 默认尝试作为CSV读取
                return pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None
    
    def _write_file(self, df: pd.DataFrame, file_path: str, data_file_format: str):
        """
        根据文件格式写入文件（通用方法）
        
        Args:
            df: 要写入的数据
            file_path: 文件路径
            data_file_format: 文件格式
        """
        try:
            if data_file_format == ".csv.gz":
                # 压缩的CSV文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
                    df.to_csv(temp_file.name, index=False)
                    
                    # 压缩文件
                    with open(temp_file.name, 'rb') as f_in:
                        with gzip.open(file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # 删除临时文件
                    os.unlink(temp_file.name)
            elif data_file_format == ".csv":
                # 普通CSV文件
                df.to_csv(file_path, index=False)
            elif data_file_format in [".parquet", ".pq"]:
                # Parquet文件
                df.to_parquet(file_path, index=False)
            elif data_file_format == ".json":
                # JSON文件
                df.to_json(file_path, orient='records', lines=True)
            elif data_file_format in [".tsv", ".tsv.gz"]:
                # TSV文件
                if data_file_format.endswith(".gz"):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as temp_file:
                        df.to_csv(temp_file.name, sep='\t', index=False)
                        
                        with open(temp_file.name, 'rb') as f_in:
                            with gzip.open(file_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        os.unlink(temp_file.name)
                else:
                    df.to_csv(file_path, sep='\t', index=False)
            else:
                # 默认作为CSV写入
                df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"写入文件 {file_path} 时出错: {e}")
    
    def _is_node_file(self, file_path: str, filename: str) -> bool:
        """
        判断文件是否为节点文件（而非边文件）
        通用方法，基于常见的图数据命名规范
        
        Args:
            file_path: 文件路径
            filename: 文件名
            
        Returns:
            bool: 是否为节点文件
        """
        # 获取父目录名称
        parent_dir = os.path.basename(os.path.dirname(file_path))
        
        # 移除文件扩展名
        base_filename = filename
        for ext in ['.csv.gz', '.csv', '.tsv.gz', '.tsv', '.parquet', '.pq', '.json']:
            if base_filename.endswith(ext):
                base_filename = base_filename[:-len(ext)]
                break
        
        # 检查驼峰命名法的边文件（如 PersonApplyLoan）
        # 驼峰命名法通常表示：实体A + 动作 + 实体B 的关系模式
        if self._is_camel_case_edge(base_filename):
            return False
        
        # 边文件的常见特征
        edge_indicators = [
            '_',  # 包含下划线通常表示关系，如 Person_knows_Person
            'edge', 'edges',
            'relation', 'relations', 'relationship', 'relationships',
            'link', 'links',
            'has', 'knows', 'likes', 'follows', 'contains',
            'replyof', 'islocatedin', 'workat', 'studyat',
            'hascreator', 'hasmember', 'hasmoderator', 'hastag', 'hastype',
            'containerof', 'ispartof', 'issubclassof'
        ]
        
        # 检查文件名和父目录名
        filename_lower = base_filename.lower()
        parent_dir_lower = parent_dir.lower()
        
        # 如果文件名或目录名包含边的特征词
        for indicator in edge_indicators:
            if indicator in filename_lower or indicator in parent_dir_lower:
                # 排除一些特殊的目录名（这些不是边）
                excluded_dirs = ['initial_snapshot', 'composite-projected-fk', 'static', 'dynamic']
                if parent_dir_lower not in [d.lower() for d in excluded_dirs]:
                    # 如果包含下划线，进一步检查是否是关系命名模式
                    if '_' in base_filename or '_' in parent_dir:
                        # 典型的关系命名: EntityA_relationship_EntityB
                        # 或者目录名包含下划线
                        return False
                    elif indicator != '_':  # 其他关键词
                        return False
        
        # 节点文件的常见特征
        node_indicators = [
            'node', 'nodes',
            'entity', 'entities',
            'vertex', 'vertices',
            'person', 'comment', 'post', 'forum', 'tag', 'place', 
            'organisation', 'organization', 'company', 'university',
            'message', 'country', 'city', 'continent'
        ]
        
        # 如果明确包含节点特征词，返回True
        for indicator in node_indicators:
            if indicator in filename_lower or indicator in parent_dir_lower:
                return True
        
        # 默认认为是节点（保守策略）
        return True
    
    def _is_camel_case_edge(self, filename: str) -> bool:
        """
        检测文件名是否为驼峰命名法的边文件
        例如: PersonApplyLoan, CompanyOwnAccount
        
        驼峰命名法边文件的特征：
        1. 至少包含3个大写字母开头的单词（实体A + 动作 + 实体B）
        2. 不是全大写（排除缩写）
        
        Args:
            filename: 文件名（不含扩展名）
            
        Returns:
            bool: 是否为驼峰命名法的边文件
        """
        import re
        
        # 如果是全大写或全小写，不是驼峰命名
        if filename.isupper() or filename.islower():
            return False
        
        # 查找所有大写字母开头的单词
        # 匹配模式：大写字母后跟小写字母
        capital_words = re.findall(r'[A-Z][a-z]*', filename)
        
        # 驼峰命名的边文件通常至少有3个单词（实体 + 动作 + 实体）
        # 例如: PersonApplyLoan = Person + Apply + Loan
        if len(capital_words) >= 3:
            return True
        
        return False
    
    @abstractmethod
    def _process_file(self, file_path: str, filename: str, data_file_format: str, perturb_type: str) -> List[Dict]:
        """
        处理单个文件（子类需要实现）
        
        Args:
            file_path: 文件路径
            filename: 文件名
            perturbation_config: 扰动配置
            data_file_format: 文件格式
            perturb_type: 扰动类型 ('incompleteness' 或 'noise')
            
        Returns:
            List[Dict]: 操作记录列表
        """
        pass
        