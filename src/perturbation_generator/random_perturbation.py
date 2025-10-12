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
from .perturb_tool import apply_incomplete_perturbation


class RandomPerturbationGenerator(BasePerturbationGenerator):
    
    def apply_perturbation(self):
        """主要的扰动应用方法"""
        gnd_path = self.config.get('data_source', {}).get('data_path', None)
        dataset_name = self.config.get('data_source', {}).get('data_set_name', 'ldbc_snb_bi')
        perturbation_config = self.config.get('perturbation', {})
        
        if not os.path.exists(gnd_path):
            raise ValueError(f"Groundtruth数据集路径不存在: {gnd_path}")
        
        perturb_type = perturbation_config.get('perturb_type')
        
        if perturb_type == 'incompleteness':
            return self.incomplete_perturb(gnd_path, dataset_name, perturbation_config)
        elif perturb_type == 'noise':
            return self.noise_perturb(gnd_path, dataset_name, perturbation_config)
        else:
            raise ValueError(f"不支持的扰动类型: {perturb_type}")
    
    def _copy_dataset(self, gnd_path: str) -> str:
        """复制数据集的公共方法，避免代码冗余"""
        # 获取当前代码文件的父目录的父目录（项目根目录）
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        perturb_dataset_dir = os.path.join(project_root, "perturb_dataset")
        perturbed_data_path = os.path.join(perturb_dataset_dir, os.path.basename(gnd_path))
        
        os.makedirs(perturb_dataset_dir, exist_ok=True)
        
        # 复制整个数据集
        if os.path.exists(perturbed_data_path):
            shutil.rmtree(perturbed_data_path)
        shutil.copytree(gnd_path, perturbed_data_path)
        
        return perturbed_data_path
    
    def incomplete_perturb(self, gnd_path: str, dataset_name: str, perturbation_config: Dict) -> tuple:
        """不完整性扰动"""
        # 复制数据集
        perturbed_data_path = self._copy_dataset(gnd_path)
        
        # 使用外部函数处理数据集扰动
        perturbed_data_path, perturbation_info = apply_incomplete_perturbation(
            perturbed_data_path, dataset_name, perturbation_config
        )
        
        # 对于incompleteness扰动，返回None作为图对象，因为这是文件级别的扰动
        return None, perturbation_info
    
    def noise_perturb(self, gnd_path: str, dataset_name: str, perturbation_config: Dict) -> tuple:
        """噪声扰动（暂时不实现具体逻辑）"""
        # 复制数据集
        perturbed_data_path = self._copy_dataset(gnd_path)
        
        perturbation_info = {
            "method": "random",
            "operations": [],
            "perturbed_data_path": perturbed_data_path
        }
        
        # TODO: 实现噪声扰动的具体逻辑
        print("噪声扰动功能暂未实现")
        
        return None, perturbation_info
        
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
        edges_to_modify = random.sample(list(graph.edges()), 
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
    
    
