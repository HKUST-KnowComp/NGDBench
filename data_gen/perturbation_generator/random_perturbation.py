"""
随机扰动生成器实现
"""

from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np
import random
import os
import shutil
import gzip
import pandas as pd
import tempfile
from .base import BasePerturbationGenerator


class RandomPerturbationGenerator(BasePerturbationGenerator):
    
    def apply_perturbation(self):
        
        perturbation_config = self.config.get('perturbation', {})
        perturb_type_ls = perturbation_config.get('type')
        perturbed_data_path = self._copy_dataset()

        for perturb_type in perturb_type_ls:
            if perturb_type == 'incompleteness':
                perturbation_info = self.incomplete_perturb(perturbed_data_path)
            elif perturb_type == 'noise':
                perturbation_info = self.noise_perturb(perturbed_data_path)
            elif perturb_type == 'mixture':
                perturbation_info = self.incomplete_perturb(perturbed_data_path)
                perturbation_info = self.noise_perturb(perturbed_data_path)
            else:
                raise ValueError(f"Unsupported: {perturb_type}")
        return perturbation_info
    
    
    def incomplete_perturb(self, perturbed_data_path: str) -> tuple:
        """
        Randomly apply incomplete perturbation to the dataset;
        Randomly delete some nodes or edges from the dataset.
        """
        perturbation_info = {
            "method": "random_incomplete",
            "operations": [],
            "perturbed_data_path": perturbed_data_path
        }
        
        data_file_format = self.data_config.get("data_file_format", ".csv.gz")
        operations = self._process_directory(perturbed_data_path, data_file_format, 'incompleteness')
        perturbation_info["operations"] = operations
        
        return  perturbation_info
    
    def _process_file(self, file_path: str, filename: str, data_file_format: str, perturb_type: str) -> List[Dict]:
        """
        process every single file (support incompleteness and noise two types of perturbation)
        
        Args:
            file_path: file path
            filename: file name
            data_file_format: data file format
            perturb_type: perturbation type ('incompleteness' or 'noise')
            
        Returns:
            List[Dict]: operation record list
        """
        operations = []
        
        try:
            df = self._read_file(file_path, data_file_format)
            
            if df is None or len(df) == 0:
                return operations
            
            original_rows = len(df)
            
            # judge node or edge
            is_node = self._is_node_file(file_path, filename)
            
            if perturb_type == 'incompleteness':
                # get incompleteness_config
                remove_nodes = self.perturbation_config.get('remove_nodes', False)
                node_removal_ratio = self.perturbation_config.get('node_removal_ratio', 0.1)
                remove_edges = self.perturbation_config.get('remove_edges', False)
                edge_removal_ratio = self.perturbation_config.get('edge_removal_ratio', 0.15)
                incomplete_attribute = self.perturbation_config.get('incomplete_attribute', False)
                incomplete_attribute_ratio = self.perturbation_config.get('incomplete_attribute_ratio', 0.05)
                
                if is_node and remove_nodes:
                    df, remove_ops = self._remove_rows(df, node_removal_ratio, file_path, filename, "node")
                    operations.extend(remove_ops)
                elif not is_node and remove_edges:
                    df, remove_ops = self._remove_rows(df, edge_removal_ratio, file_path, filename, "edge")
                    operations.extend(remove_ops)
                
                # apply attribute modification perturbation (incompleteness type)
                if incomplete_attribute and incomplete_attribute_ratio > 0:
                    df, modify_ops = self._modify_attributes_incomplete(df, incomplete_attribute_ratio, 
                                                                        file_path, filename)
                    operations.extend(modify_ops)
            
            elif perturb_type == 'noise':
                # get noise_config
                # To be modified
                add_noise = self.perturbation_config.get('add_noise', False)
                noise_ratio = self.perturbation_config.get('noise_ratio', 0.05)
                numeric_noise_factor = self.perturbation_config.get('numeric_noise_factor', 0.1)
                string_noise_factor = self.perturbation_config.get('string_noise_factor', 0.1)
                
                if add_noise and noise_ratio is not None and noise_ratio > 0:
                    df, noise_ops = self._add_noise_to_attributes(df, noise_ratio, numeric_noise_factor, 
                                                                  string_noise_factor, file_path, filename)
                    operations.extend(noise_ops)
            
            
            if len(operations) > 0:  # 只有在有修改时才写回
                self._write_file(df, file_path, data_file_format)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
        
        return operations
    
    def _remove_rows(self, df: pd.DataFrame, removal_ratio: float, file_path: str, 
                     filename: str, entity_type: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        随机删除行Ramdonly remove some rows from the dataset according to the removal_ratio
        
        Args:
            df: dataframe
            removal_ratio: removal ratio
            file_path: file path
            filename: file name
            entity_type: entity type (node or edge)
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: modified dataframe and operation record
        """
        operations = []
        original_rows = len(df)
        
        if original_rows == 0 or removal_ratio <= 0:
            return df, operations
        
        # Calculate the number of rows to remove
        num_to_remove = int(original_rows * removal_ratio)
        
        if num_to_remove > 0 and num_to_remove < original_rows:
            # Randomly select the rows to remove
            rows_to_remove = random.sample(range(original_rows), num_to_remove)
            df = df.drop(rows_to_remove).reset_index(drop=True)
            
            # Record the operation
            operations.append({
                "operation": "remove_rows",
                "entity_type": entity_type,
                "file": filename,
                "file_path": file_path,
                "original_rows": original_rows,
                "removed_rows": num_to_remove,
                "remaining_rows": len(df),
                "removal_ratio": removal_ratio
            })
        
        return df, operations
    
    def _modify_attributes_incomplete(self, df: pd.DataFrame, modification_ratio: float, 
                                     file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        随机修改属性值（不完整性扰动 - 将某些属性设为空值或删除）
        
        Args:
            df: 数据框
            modification_ratio: 修改比例
            file_path: 文件路径
            filename: 文件名
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: 修改后的数据框和操作记录
        """
        operations = []
        
        if len(df) == 0 or modification_ratio <= 0:
            return df, operations
        
        # 计算要修改的行数
        num_to_modify = max(1, int(len(df) * modification_ratio))
        
        # 随机选择要修改的行
        rows_to_modify = random.sample(range(len(df)), min(num_to_modify, len(df)))
        
        # 选择可以修改的列（排除ID列）
        modifiable_columns = [col for col in df.columns 
                             if 'id' not in col.lower()]
        
        if len(modifiable_columns) == 0:
            return df, operations
        
        modified_count = 0
        for row_idx in rows_to_modify:
            # 随机选择一列进行修改
            col = random.choice(modifiable_columns)
            original_value = df.at[row_idx, col]
            
            # 不完整性扰动：将值设为空值（NaN）
            if not pd.isna(original_value):
                df.at[row_idx, col] = np.nan
                modified_count += 1
        
        if modified_count > 0:
            operations.append({
                "operation": "modify_attributes_incomplete",
                "file": filename,
                "file_path": file_path,
                "modified_rows": modified_count,
                "modification_ratio": modification_ratio,
                "description": "将属性设为空值"
            })
        
        return df, operations
    
    def _add_noise_to_attributes(self, df: pd.DataFrame, noise_ratio: float, 
                                numeric_noise_factor: float, string_noise_factor: float,
                                file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        向属性添加噪声（噪声扰动）
        
        Args:
            df: 数据框
            noise_ratio: 噪声比例
            numeric_noise_factor: 数值噪声因子
            string_noise_factor: 字符串噪声因子
            file_path: 文件路径
            filename: 文件名
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: 修改后的数据框和操作记录
        """
        operations = []
        
        if len(df) == 0 or noise_ratio <= 0:
            return df, operations
        
        # 计算要添加噪声的行数
        num_to_modify = max(1, int(len(df) * noise_ratio))
        
        # 随机选择要添加噪声的行
        rows_to_modify = random.sample(range(len(df)), min(num_to_modify, len(df)))
        
        # 选择可以修改的列（排除ID列）
        modifiable_columns = [col for col in df.columns 
                             if 'id' not in col.lower() and df[col].dtype in ['int64', 'float64', 'object']]
        
        if len(modifiable_columns) == 0:
            return df, operations
        
        modified_count = 0
        for row_idx in rows_to_modify:
            # 随机选择一列进行修改
            col = random.choice(modifiable_columns)
            original_value = df.at[row_idx, col]
            
            # 根据数据类型添加噪声
            if pd.isna(original_value):
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # 数值类型：添加高斯噪声
                noise = np.random.normal(0, abs(float(original_value)) * numeric_noise_factor)
                new_value = float(original_value) + noise
                # 保持原始数据类型
                if df[col].dtype == 'int64':
                    new_value = int(round(new_value))
                df.at[row_idx, col] = new_value
                modified_count += 1
            elif df[col].dtype == 'object':
                # 字符串类型：添加噪声后缀或前缀
                if isinstance(original_value, str):
                    if random.random() < 0.5:
                        new_value = original_value + "_noise"
                    else:
                        new_value = "noise_" + original_value
                    df.at[row_idx, col] = new_value
                    modified_count += 1
        
        if modified_count > 0:
            operations.append({
                "operation": "add_noise_to_attributes",
                "file": filename,
                "file_path": file_path,
                "modified_rows": modified_count,
                "noise_ratio": noise_ratio,
                "numeric_noise_factor": numeric_noise_factor,
                "description": "向属性添加噪声"
            })
        
        return df, operations
        
    
    def noise_perturb(self, perturbed_data_path: str) -> tuple:
        """
        Randomly add some noise to the dataset;
        Randomly modify some attributes of the nodes or edges in the dataset.
        """
        perturbation_info = {
            "method": "random_noise",
            "operations": [],
            "perturbed_data_path": perturbed_data_path
        }
        
        # 获取文件格式
        data_file_format = self.data_config.get("data_file_format", ".csv.gz")
        
        # 递归处理所有文件，使用基类的通用方法
        operations = self._process_directory(perturbed_data_path, data_file_format, 'noise')
        perturbation_info["operations"] = operations
        
        return perturbed_data_path, perturbation_info
        
        
    def _modify_random_attributes(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """随机修改节点和边属性"""
        modification_ratio = self.config.get('attribute_modification_ratio', 0.1)
        operations = []
        
        # 修改节点属性
        nodes_to_modify = random.sample(list(graph.nodes()), 
                                      int(graph.number_of_nodes() * modification_ratio))
        
        for node in nodes_to_modify:
            if graph.nodes[node]:
                attr_name = random.choice(list(graph.nodes[node].keys()))
                original_value = graph.nodes[node][attr_name]
                
                # 根据属性类型生成新值
                new_value = self._generate_noisy_value(original_value)
                
                operations.append({
                    "operation": "modify_node_attribute",
                    "target": node,
                    "attribute": attr_name,
                    "original_value": original_value,
                    "new_value": new_value
                })
                
                graph.nodes[node][attr_name] = new_value
        
        # 修改边属性
        edges_to_modify = random.sample(list[Incomplete](graph.edges()), 
                                      int(graph.number_of_edges() * modification_ratio))
        
        for edge in edges_to_modify:
            if graph.edges[edge]:
                attr_name = random.choice(list(graph.edges[edge].keys()))
                original_value = graph.edges[edge][attr_name]
                
                new_value = self._generate_noisy_value(original_value)
                
                operations.append({
                    "operation": "modify_edge_attribute",
                    "target": edge,
                    "attribute": attr_name,
                    "original_value": original_value,
                    "new_value": new_value
                })
                
                graph.edges[edge][attr_name] = new_value
        
        return graph, operations
    
    def _add_random_noise(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """添加随机噪声数据"""
        noise_ratio = self.config.get('noise_ratio', 0.05)
        operations = []
        
        # 添加噪声节点
        num_noise_nodes = int(graph.number_of_nodes() * noise_ratio)
        max_node_id = max(graph.nodes()) if graph.nodes() else 0
        
        for i in range(num_noise_nodes):
            noise_node = max_node_id + i + 1
            noise_attrs = self._generate_noise_attributes()
            
            graph.add_node(noise_node, **noise_attrs)
            operations.append({
                "operation": "add_noise_node",
                "target": noise_node,
                "attributes": noise_attrs
            })
        
        # 添加噪声边
        num_noise_edges = int(graph.number_of_edges() * noise_ratio)
        nodes = list(graph.nodes())
        
        for _ in range(num_noise_edges):
            if len(nodes) >= 2:
                source, target = random.sample(nodes, 2)
                if not graph.has_edge(source, target):
                    noise_attrs = self._generate_noise_attributes()
                    graph.add_edge(source, target, **noise_attrs)
                    operations.append({
                        "operation": "add_noise_edge",
                        "target": (source, target),
                        "attributes": noise_attrs
                    })
        
        return graph, operations
    
    def _generate_noisy_value(self, original_value: Any) -> Any:
        """为给定值生成噪声版本"""
        if isinstance(original_value, (int, float)):
            noise_factor = self.config.get('numeric_noise_factor', 0.1)
            noise = np.random.normal(0, abs(original_value) * noise_factor)
            return type(original_value)(original_value + noise)
        elif isinstance(original_value, str):
            # 字符串噪声：随机替换字符或添加后缀
            if random.random() < 0.5:
                return original_value + "_noise"
            else:
                return "noise_" + original_value
        else:
            return original_value
    
    def _generate_noise_attributes(self) -> Dict[str, Any]:
        """生成噪声属性"""
        return {
            "noise_value": np.random.randint(1, 1000),
            "noise_category": random.choice(["noise_A", "noise_B", "noise_C"]),
            "noise_weight": np.random.uniform(0.0, 1.0),
            "is_noise": True
        }
    
    
