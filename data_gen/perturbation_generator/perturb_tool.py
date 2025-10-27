"""
扰动工具函数
"""

import os
import shutil
import gzip
import pandas as pd
import tempfile
import random
import numpy as np
from typing import Dict, List


def apply_incomplete_perturbation(perturbed_data_path: str, dataset_name: str, perturbation_config: Dict) -> tuple:
    """
    针对不同数据集应用不完整性扰动
    
    Args:
        perturbed_data_path: 已复制的扰动数据集路径
        dataset_name: 数据集名称
        perturbation_config: 扰动配置
        
    Returns:
        tuple: (perturbed_data_path, perturbation_info)
    """
    if not os.path.exists(perturbed_data_path):
        raise ValueError(f"扰动数据集路径不存在: {perturbed_data_path}")
    
    perturbation_info = {
        "method": "random",
        "operations": [],
        "perturbed_data_path": perturbed_data_path
    }
    
    if dataset_name == 'ldbc_snb_bi':
        # 定位initial_snapshot文件夹
        initial_snapshot_path = os.path.join(
            perturbed_data_path, 
            "graphs", "csv", "bi", "composite-projected-fk", "initial_snapshot"
        )
        
        if not os.path.exists(initial_snapshot_path):
            raise ValueError(f"Initial snapshot路径不存在: {initial_snapshot_path}")
        
        operations = []
        
        # 处理dynamic数据
        dynamic_path = os.path.join(initial_snapshot_path, "dynamic")
        if os.path.exists(dynamic_path):
            dynamic_ops = perturb_csv_incomplete(
                dynamic_path, 
                perturbation_config,
                "dynamic"
            )
            operations.extend(dynamic_ops)
        
        # 处理static数据
        static_path = os.path.join(initial_snapshot_path, "static")
        if os.path.exists(static_path):
            static_ops = perturb_csv_incomplete(
                static_path, 
                perturbation_config,
                "static"
            )
            operations.extend(static_ops)
        
        perturbation_info["operations"] = operations
        
    elif dataset_name == 'ldbc_finbench':
        # TODO: 实现ldbc_finbench的扰动逻辑
        pass
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return perturbed_data_path, perturbation_info


def perturb_csv_incomplete(data_path: str, perturbation_config: Dict, data_type: str) -> List[Dict]:
    """
    对CSV文件进行随机删除扰动
    
    Args:
        data_path: 数据路径
        perturbation_config: 扰动配置
        data_type: 数据类型 (dynamic/static)
        
    Returns:
        List[Dict]: 操作记录列表
    """
    operations = []
    
    # 获取扰动参数
    remove_nodes = perturbation_config.get('remove_nodes', False)
    node_removal_ratio = perturbation_config.get('node_removal_ratio', 0.1)
    remove_edges = perturbation_config.get('remove_edges', False)
    edge_removal_ratio = perturbation_config.get('edge_removal_ratio', 0.1)
    seed = perturbation_config.get('seed')
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 遍历所有实体文件夹
    for entity_name in os.listdir(data_path):
        entity_path = os.path.join(data_path, entity_name)
        if not os.path.isdir(entity_path):
            continue
        
        # 处理该实体的所有CSV文件
        csv_files = [f for f in os.listdir(entity_path) if f.endswith('.csv.gz')]
        
        for csv_file in csv_files:
            csv_path = os.path.join(entity_path, csv_file)
            
            try:
                # 解压并读取CSV文件
                with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
                
                original_rows = len(df)
                if original_rows == 0:
                    continue
                
                # 根据实体类型决定删除策略
                if is_node_entity(entity_name):
                    if remove_nodes:
                        removal_ratio = node_removal_ratio
                    else:
                        removal_ratio = 0
                else:
                    if remove_edges:
                        removal_ratio = edge_removal_ratio
                    else:
                        removal_ratio = 0
                
                if removal_ratio > 0:
                    # 随机选择要删除的行
                    num_to_remove = int(original_rows * removal_ratio)
                    if num_to_remove > 0 and num_to_remove < original_rows:
                        rows_to_remove = random.sample(range(original_rows), num_to_remove)
                        df = df.drop(rows_to_remove).reset_index(drop=True)
                        
                        # 记录操作
                        operations.append({
                            "operation": "remove_rows",
                            "entity": entity_name,
                            "file": csv_file,
                            "data_type": data_type,
                            "original_rows": original_rows,
                            "removed_rows": num_to_remove,
                            "remaining_rows": len(df),
                            "removal_ratio": removal_ratio
                        })
                        
                        # 重新压缩并写回文件
                        write_compressed_csv(df, csv_path)
                
            except Exception as e:
                print(f"处理文件 {csv_path} 时出错: {e}")
                continue
    
    return operations


def is_node_entity(entity_name: str) -> bool:
    """
    判断实体是否为节点实体
    基于目录名称模式判断：简单名称表示实体，包含下划线的表示关系
    
    Args:
        entity_name: 实体名称
        
    Returns:
        bool: 是否为节点实体
    """
    # 如果包含下划线，通常是关系（如Comment_hasCreator_Person）
    if '_' in entity_name:
        return False
    
    # 简单名称通常是实体（如Comment, Person等）
    return True


def write_compressed_csv(df: pd.DataFrame, output_path: str):
    """
    将DataFrame写入压缩的CSV文件
    
    Args:
        df: DataFrame对象
        output_path: 输出文件路径
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
        
        # 压缩文件
        with open(temp_file.name, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 删除临时文件
        os.unlink(temp_file.name)