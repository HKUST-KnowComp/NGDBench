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
from . import perturb_tool


class RandomPerturbationGenerator(BasePerturbationGenerator):
    
    def apply_perturbation(self):
        
        perturbation_config = self.config.get('perturbation', {})
        perturb_type_ls = perturbation_config.get('type')
        perturbed_data_path = self._copy_dataset()
        # 如果要混合扰动，这里逻辑要改，perturbation_info得存下来
        for perturb_type in perturb_type_ls:
            if perturb_type == 'incompleteness':
                perturbation_info = self.incomplete_perturb(perturbed_data_path)
            elif perturb_type == 'noise':
                perturbation_info = self.noise_perturb(perturbed_data_path)
            else:
                raise ValueError(f"Unsupported: {perturb_type}")
        return perturbation_info
    
    def incomplete_perturb(self, perturbed_data_path: str) -> tuple:
        """
        Randomly apply incomplete perturbation to the dataset;
        Randomly delete some nodes or edges from the dataset.
        """
        # 初始化已删除节点ID的集合，用于防止悬空边
        self.removed_node_ids = set()
        
        perturbation_info = {
            "method": "random_incomplete",
            "operations": [],
            "perturbed_data_path": perturbed_data_path
        }
        
        data_file_format = self.data_config.get("data_file_format", ".csv.gz")
        operations = self._process_directory(perturbed_data_path, data_file_format, 'incompleteness')
        perturbation_info["operations"] = operations
        
        return  perturbation_info
    
    def _remove_rows(self, df: pd.DataFrame, removal_ratio: float, file_path: str, 
                     filename: str, entity_type: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Ramdonly remove some rows from the dataset according to the removal_ratio
        
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
            rows_to_remove = random.sample(range(original_rows), num_to_remove)
            
            # 如果是节点类型，记录被删除的节点ID
            if entity_type == "node" and hasattr(self, 'removed_node_ids'):
                removed_ids = self._extract_node_ids(df, rows_to_remove)
                self.removed_node_ids.update(removed_ids)
            
            df = df.drop(rows_to_remove).reset_index(drop=True)
            
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
        
        num_to_modify = max(1, int(len(df) * modification_ratio))
        
        rows_to_modify = random.sample(range(len(df)), min(num_to_modify, len(df)))
        
        modifiable_columns = [col for col in df.columns 
                             if 'id' not in col.lower()]
        
        if len(modifiable_columns) == 0:
            return df, operations
        
        modified_count = 0
        for row_idx in rows_to_modify:
            col = random.choice(modifiable_columns)
            original_value = df.at[row_idx, col]
            
            # Set the value to NaN（NaN）
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
        
        # get data file format
        data_file_format = self.data_config.get("data_file_format", ".csv.gz")
        
        # process all files in the directory
        operations = self._process_directory(perturbed_data_path, data_file_format, 'noise')
        perturbation_info["operations"] = operations
        
        return perturbation_info

    def _add_noise_attributes(self, df: pd.DataFrame, noise_ratio: float, 
                                numeric_noise_factor: float, string_noise_factor: float,
                                file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        向属性添加噪声（噪声扰动）
        对每个待加噪的行，采样若干列进行加噪：
        - 字符串：大小写互换、插入干扰字符
        - 时间戳/数值：转为字符串后插入干扰字符
        
        Args:
            df: 数据框
            noise_ratio: 噪声比例（要加噪的行的比例）
            numeric_noise_factor: 数值噪声因子（未使用，保留接口兼容性）
            string_noise_factor: 字符串噪声因子（未使用，保留接口兼容性）
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
                             if 'id' not in col.lower()]
        
        if len(modifiable_columns) == 0:
            return df, operations
        
        # 获取每行要修改的列的比例
        columns_ratio = self.perturbation_config.get('noise_columns_ratio', 0.3)
        
        modified_count = 0
        total_columns_modified = 0
        
        for row_idx in rows_to_modify:
            # 为当前行随机采样若干列进行加噪（根据比例计算列数）
            num_cols_to_modify = max(1, int(len(modifiable_columns) * columns_ratio))
            cols_to_modify = random.sample(modifiable_columns, min(num_cols_to_modify, len(modifiable_columns)))
            
            for col in cols_to_modify:
                original_value = df.at[row_idx, col]
                
                # 跳过空值
                if pd.isna(original_value):
                    continue
                
                # 根据数据类型添加不同的噪声（调用外部工具函数）
                new_value = perturb_tool.add_noise_to_value(original_value, df[col].dtype)
                
                if new_value != original_value:
                    df.at[row_idx, col] = new_value
                    total_columns_modified += 1
            
            modified_count += 1
        
        if modified_count > 0:
            operations.append({
                "operation": "add_noise_to_attributes",
                "file": filename,
                "file_path": file_path,
                "modified_rows": modified_count,
                "total_columns_modified": total_columns_modified,
                "noise_ratio": noise_ratio,
                "columns_ratio": columns_ratio,
                "description": "向属性值添加噪声（大小写互换、插入干扰字符等）"
            })
        
        return df, operations
        
    def _add_noise_rows(self, df: pd.DataFrame, noise_ratio: float, 
                                file_path: str, filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        添加噪声行（错误行）
        通过复制已有的行并修改其中某些列的值来生成噪声行
        例如：同样的ID和companyName，但是不一样的createTime或者country等
        
        Args:
            df: 数据框
            noise_ratio: 噪声比例（要添加的噪声行占总行数的比例）
            file_path: 文件路径
            filename: 文件名
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: 添加噪声行后的数据框和操作记录
        """
        operations = []
        
        if len(df) == 0 or noise_ratio <= 0:
            return df, operations
        
        # 计算要添加的噪声行数
        num_noise_rows = max(1, int(len(df) * noise_ratio))
        
        # 随机选择要复制的行
        rows_to_copy = random.sample(range(len(df)), min(num_noise_rows, len(df)))
        
        # 获取每个噪声行要修改的列的比例
        columns_ratio = self.perturbation_config.get('noise_row_columns_ratio', 0.3)
        
        # 选择可以修改的列（排除ID列，因为ID通常需要保持一致或作为识别标志）
        # 但根据需求，有时候我们需要保留某些列（如ID、Name）而修改其他列
        all_columns = df.columns.tolist()
        
        noise_rows = []
        for row_idx in rows_to_copy:
            # 复制该行
            noise_row = df.iloc[row_idx].copy()
            
            # 根据比例计算要修改的列数
            num_cols_to_change = max(1, int(len(all_columns) * columns_ratio))
            cols_to_modify = random.sample(all_columns, min(num_cols_to_change, len(all_columns)))
            
            modified_columns = {}
            for col in cols_to_modify:
                original_value = noise_row[col]
                
                # 跳过空值
                if pd.isna(original_value):
                    continue
                
                # 生成不同的值（调用外部工具函数）
                new_value = perturb_tool.generate_different_value(original_value, df[col])
                noise_row[col] = new_value
                modified_columns[col] = {
                    'original': str(original_value),
                    'new': str(new_value)
                }
            
            noise_rows.append(noise_row)
            operations.append({
                "operation": "add_noise_row",
                "file": filename,
                "source_row_index": int(row_idx),
                "modified_columns": modified_columns,
                "description": f"复制第{row_idx}行并修改{len(modified_columns)}列"
            })
        
        # 将噪声行添加到原数据框
        if noise_rows:
            noise_df = pd.DataFrame(noise_rows)
            df = pd.concat([df, noise_df], ignore_index=True)
        
        return df, operations
    
    def _extract_node_ids(self, df: pd.DataFrame, row_indices: List[int]) -> set:
        """
        从数据框中提取指定行的节点ID
        
        Args:
            df: 数据框
            row_indices: 要提取ID的行索引列表
            
        Returns:
            set: 节点ID集合
        """
        node_ids = set()
        
        # 尝试识别ID列（通常是第一列或包含'id'的列）
        id_column = None
        
        # 首先查找明确包含'id'的列名（不区分大小写）
        for col in df.columns:
            if col.lower() == 'id' or col.lower().endswith('.id'):
                id_column = col
                break
        
        # 如果没找到，使用第一列
        if id_column is None and len(df.columns) > 0:
            id_column = df.columns[0]
        
        # 提取ID
        if id_column is not None:
            for idx in row_indices:
                if idx < len(df):
                    node_id = df.iloc[idx][id_column]
                    # 转换为字符串以统一处理不同类型的ID
                    node_ids.add(str(node_id))
        
        return node_ids
    
    def _remove_dangling_edges(self, df: pd.DataFrame, file_path: str, 
                               filename: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        删除连接到已删除节点的悬空边
        
        Args:
            df: 边文件的数据框
            file_path: 文件路径
            filename: 文件名
            
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: 修改后的数据框和操作记录
        """
        operations = []
        original_rows = len(df)
        
        if original_rows == 0 or not hasattr(self, 'removed_node_ids') or len(self.removed_node_ids) == 0:
            return df, operations
        
        # 尝试识别边文件中的源节点和目标节点列
        # 通常边文件的前两列是源节点和目标节点ID
        if len(df.columns) < 2:
            return df, operations
        
        source_col = df.columns[0]
        target_col = df.columns[1]
        
        # 找出需要删除的边（源节点或目标节点在已删除节点集合中）
        rows_to_keep = []
        rows_to_remove = []
        
        for idx, row in df.iterrows():
            source_id = str(row[source_col])
            target_id = str(row[target_col])
            
            # 如果源节点或目标节点被删除了，则这条边也需要删除
            if source_id in self.removed_node_ids or target_id in self.removed_node_ids:
                rows_to_remove.append(idx)
            else:
                rows_to_keep.append(idx)
        
        # 删除悬空边
        if len(rows_to_remove) > 0:
            df = df.loc[rows_to_keep].reset_index(drop=True)
            
            operations.append({
                "operation": "remove_dangling_edges",
                "file": filename,
                "file_path": file_path,
                "original_rows": original_rows,
                "removed_rows": len(rows_to_remove),
                "remaining_rows": len(df),
                "description": f"删除了{len(rows_to_remove)}条连接到已删除节点的悬空边"
            })
        
        return df, operations
    
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
                elif not is_node:
                    # 对于边文件，首先删除悬空边（连接到已删除节点的边）
                    if hasattr(self, 'removed_node_ids') and len(self.removed_node_ids) > 0:
                        df, dangling_ops = self._remove_dangling_edges(df, file_path, filename)
                        operations.extend(dangling_ops)
                    
                    # 然后再按比例删除边
                    if remove_edges:
                        df, remove_ops = self._remove_rows(df, edge_removal_ratio, file_path, filename, "edge")
                        operations.extend(remove_ops)
                
                # apply attribute modification perturbation (incompleteness type)
                if incomplete_attribute and incomplete_attribute_ratio > 0:
                    df, modify_ops = self._modify_attributes_incomplete(df, incomplete_attribute_ratio, 
                                                                        file_path, filename)
                    operations.extend(modify_ops)
            
            elif perturb_type == 'noise':
                # 获取噪声配置
                add_noise = self.perturbation_config.get('add_noise', True)
                
                # 属性值加噪
                if self.perturbation_config.get('noise_attributes', False):
                    noise_attributes_ratio = self.perturbation_config.get('noise_attributes_ratio', 0.05)
                    if noise_attributes_ratio > 0:
                        # 传入所有必需的参数（numeric_noise_factor 和 string_noise_factor 保留用于接口兼容性）
                        df, noise_attributes_ops = self._add_noise_attributes(
                            df, noise_attributes_ratio, 0.1, 0.1, file_path, filename)
                        operations.extend(noise_attributes_ops)
                
                # 添加噪声行
                if self.perturbation_config.get('noise_rows', False):
                    noise_rows_ratio = self.perturbation_config.get('noise_rows_ratio', 0.05)
                    if noise_rows_ratio > 0:
                        df, noise_rows_ops = self._add_noise_rows(
                            df, noise_rows_ratio, file_path, filename)
                        operations.extend(noise_rows_ops)
                
            
            
            if len(operations) > 0:  
                self._write_file(df, file_path, data_file_format)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
        
        return operations