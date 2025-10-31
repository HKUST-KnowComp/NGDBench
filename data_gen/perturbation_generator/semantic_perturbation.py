from typing import Any, Dict, List, Tuple, Callable
import networkx as nx
import numpy as np
import random
from .base import BasePerturbationGenerator


class SemanticPerturbationGenerator(BasePerturbationGenerator):
    """Semantic perturbation generator - based on semantic logic to delete or modify data"""
    
    def apply_perturbation(self, graph: nx.Graph, perturb_type) -> Tuple[nx.Graph, Dict[str, Any]]:
        
        perturbed_graph = graph.copy()
        perturbation_info = {
            "type": "semantic",
            "operations": []
        }
        if perturb_type == 'incompleteness':
        # 基于实体关系的语义删除
            if self.config.get('semantic_node_removal', False):
                perturbed_graph, node_ops = self._semantic_node_removal(perturbed_graph)
                perturbation_info["operations"].extend(node_ops)
            
            # 基于关系语义的边删除
            if self.config.get('semantic_edge_removal', False):
                perturbed_graph, edge_ops = self._semantic_edge_removal(perturbed_graph)
                perturbation_info["operations"].extend(edge_ops)
            
            # 语义属性修改
            if self.config.get('semantic_attribute_modification', False):
                perturbed_graph, attr_ops = self._semantic_attribute_modification(perturbed_graph)
                perturbation_info["operations"].extend(attr_ops)
        elif perturb_type == 'noise':    
            # 添加语义噪声
            if self.config.get('semantic_noise', False):
                perturbed_graph, noise_ops = self._add_semantic_noise(perturbed_graph)
                perturbation_info["operations"].extend(noise_ops)
        else:
            raise ValueError(f"不支持的扰动类型: {perturb_type}")
        
        if not self.validate_perturbation(graph, perturbed_graph):
            raise ValueError("语义扰动验证失败")
        
        return perturbed_graph, perturbation_info
    
    def _semantic_node_removal(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """基于语义规则删除节点"""
        removal_rules = self.config.get('node_removal_rules', [])
        operations = []
        
        for rule in removal_rules:
            nodes_to_remove = self._apply_node_rule(graph, rule)
            
            for node in nodes_to_remove:
                if node in graph.nodes():
                    operations.append({
                        "operation": "semantic_remove_node",
                        "target": node,
                        "rule": rule,
                        "original_data": dict(graph.nodes[node])
                    })
                    graph.remove_node(node)
        
        return graph, operations
    
    def _semantic_edge_removal(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """基于语义规则删除边"""
        removal_rules = self.config.get('edge_removal_rules', [])
        operations = []
        
        for rule in removal_rules:
            edges_to_remove = self._apply_edge_rule(graph, rule)
            
            for edge in edges_to_remove:
                if graph.has_edge(*edge):
                    operations.append({
                        "operation": "semantic_remove_edge",
                        "target": edge,
                        "rule": rule,
                        "original_data": dict(graph.edges[edge])
                    })
                    graph.remove_edge(*edge)
        
        return graph, operations
    
    def _semantic_attribute_modification(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """基于语义规则修改属性"""
        modification_rules = self.config.get('attribute_modification_rules', [])
        operations = []
        
        for rule in modification_rules:
            if rule['target_type'] == 'node':
                ops = self._modify_node_attributes_by_rule(graph, rule)
            else:
                ops = self._modify_edge_attributes_by_rule(graph, rule)
            operations.extend(ops)
        
        return graph, operations
    
    def _add_semantic_noise(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Dict]]:
        """添加语义相关的噪声"""
        noise_rules = self.config.get('semantic_noise_rules', [])
        operations = []
        
        for rule in noise_rules:
            if rule['type'] == 'inconsistent_attributes':
                ops = self._add_inconsistent_attributes(graph, rule)
            elif rule['type'] == 'temporal_inconsistency':
                ops = self._add_temporal_inconsistency(graph, rule)
            elif rule['type'] == 'relationship_inconsistency':
                ops = self._add_relationship_inconsistency(graph, rule)
            else:
                ops = []
            
            operations.extend(ops)
        
        return graph, operations
    
    def _apply_node_rule(self, graph: nx.Graph, rule: Dict[str, Any]) -> List:
        """应用节点删除规则"""
        nodes_to_remove = []
        
        if rule['type'] == 'attribute_based':
            # 基于属性值删除节点
            attr_name = rule['attribute']
            condition = rule['condition']
            
            for node, data in graph.nodes(data=True):
                if attr_name in data:
                    if self._evaluate_condition(data[attr_name], condition):
                        nodes_to_remove.append(node)
        
        elif rule['type'] == 'degree_based':
            # 基于度数删除节点
            threshold = rule['threshold']
            comparison = rule.get('comparison', 'less_than')
            
            for node in graph.nodes():
                degree = graph.degree(node)
                if comparison == 'less_than' and degree < threshold:
                    nodes_to_remove.append(node)
                elif comparison == 'greater_than' and degree > threshold:
                    nodes_to_remove.append(node)
        
        elif rule['type'] == 'category_based':
            # 基于类别删除节点
            category_attr = rule['category_attribute']
            target_categories = rule['target_categories']
            
            for node, data in graph.nodes(data=True):
                if category_attr in data and data[category_attr] in target_categories:
                    nodes_to_remove.append(node)
        
        # 限制删除数量
        max_removal = rule.get('max_removal_ratio', 0.3)
        max_count = int(graph.number_of_nodes() * max_removal)
        
        return nodes_to_remove[:max_count]
    
    def _apply_edge_rule(self, graph: nx.Graph, rule: Dict[str, Any]) -> List[Tuple]:
        """应用边删除规则"""
        edges_to_remove = []
        
        if rule['type'] == 'attribute_based':
            attr_name = rule['attribute']
            condition = rule['condition']
            
            for edge in graph.edges(data=True):
                if attr_name in edge[2]:
                    if self._evaluate_condition(edge[2][attr_name], condition):
                        edges_to_remove.append((edge[0], edge[1]))
        
        elif rule['type'] == 'relationship_type':
            # 基于关系类型删除边
            target_types = rule['target_types']
            type_attr = rule.get('type_attribute', 'type')
            
            for edge in graph.edges(data=True):
                if type_attr in edge[2] and edge[2][type_attr] in target_types:
                    edges_to_remove.append((edge[0], edge[1]))
        
        # 限制删除数量
        max_removal = rule.get('max_removal_ratio', 0.3)
        max_count = int(graph.number_of_edges() * max_removal)
        
        return edges_to_remove[:max_count]
    
    def _modify_node_attributes_by_rule(self, graph: nx.Graph, rule: Dict[str, Any]) -> List[Dict]:
        """基于规则修改节点属性"""
        operations = []
        target_attr = rule['target_attribute']
        modification_type = rule['modification_type']
        
        for node, data in graph.nodes(data=True):
            if target_attr in data:
                original_value = data[target_attr]
                new_value = self._apply_semantic_modification(original_value, modification_type, rule)
                
                if new_value != original_value:
                    graph.nodes[node][target_attr] = new_value
                    operations.append({
                        "operation": "semantic_modify_node_attribute",
                        "target": node,
                        "attribute": target_attr,
                        "original_value": original_value,
                        "new_value": new_value,
                        "rule": rule
                    })
        
        return operations
    
    def _modify_edge_attributes_by_rule(self, graph: nx.Graph, rule: Dict[str, Any]) -> List[Dict]:
        """基于规则修改边属性"""
        operations = []
        target_attr = rule['target_attribute']
        modification_type = rule['modification_type']
        
        for edge in graph.edges(data=True):
            if target_attr in edge[2]:
                original_value = edge[2][target_attr]
                new_value = self._apply_semantic_modification(original_value, modification_type, rule)
                
                if new_value != original_value:
                    graph.edges[edge[0], edge[1]][target_attr] = new_value
                    operations.append({
                        "operation": "semantic_modify_edge_attribute",
                        "target": (edge[0], edge[1]),
                        "attribute": target_attr,
                        "original_value": original_value,
                        "new_value": new_value,
                        "rule": rule
                    })
        
        return operations
    
    def _add_inconsistent_attributes(self, graph: nx.Graph, rule: Dict[str, Any]) -> List[Dict]:
        """添加不一致的属性值"""
        operations = []
        inconsistency_ratio = rule.get('inconsistency_ratio', 0.1)
        target_nodes = random.sample(list(graph.nodes()), 
                                   int(graph.number_of_nodes() * inconsistency_ratio))
        
        for node in target_nodes:
            # 添加与现有属性冲突的新属性
            conflicting_attr = f"conflicting_{random.choice(['age', 'status', 'type'])}"
            conflicting_value = self._generate_conflicting_value(graph.nodes[node])
            
            graph.nodes[node][conflicting_attr] = conflicting_value
            operations.append({
                "operation": "add_inconsistent_attribute",
                "target": node,
                "attribute": conflicting_attr,
                "value": conflicting_value
            })
        
        return operations
    
    def _add_temporal_inconsistency(self, graph: nx.Graph, rule: Dict[str, Any]) -> List[Dict]:
        """添加时间不一致性"""
        operations = []
        # 实现时间相关的语义噪声
        # 例如：创建时间晚于修改时间等
        return operations
    
    def _add_relationship_inconsistency(self, graph: nx.Graph, rule: Dict[str, Any]) -> List[Dict]:
        """添加关系不一致性"""
        operations = []
        # 实现关系不一致的语义噪声
        # 例如：同时存在冲突的关系类型
        return operations
    
    def _evaluate_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        op = condition['operator']
        target = condition['value']
        
        if op == 'equals':
            return value == target
        elif op == 'not_equals':
            return value != target
        elif op == 'greater_than':
            return value > target
        elif op == 'less_than':
            return value < target
        elif op == 'in':
            return value in target
        elif op == 'not_in':
            return value not in target
        else:
            return False
    
    def _apply_semantic_modification(self, value: Any, modification_type: str, rule: Dict[str, Any]) -> Any:
        """应用语义修改"""
        if modification_type == 'age_inconsistency':
            # 年龄不一致：设置不合理的年龄值
            if isinstance(value, (int, float)) and 0 < value < 150:
                return random.choice([-10, 200, 999])
        elif modification_type == 'status_conflict':
            # 状态冲突：设置冲突的状态
            status_conflicts = rule.get('status_conflicts', {})
            return status_conflicts.get(value, value)
        elif modification_type == 'category_mismatch':
            # 类别不匹配
            if isinstance(value, str):
                return f"invalid_{value}"
        
        return value
    
    def _generate_conflicting_value(self, node_data: Dict[str, Any]) -> Any:
        """生成与现有数据冲突的值"""
        # 基于现有属性生成冲突值
        if 'age' in node_data:
            return -abs(node_data['age'])  # 负年龄
        elif 'category' in node_data:
            return f"not_{node_data['category']}"  # 否定类别
        else:
            return "conflicting_value"
