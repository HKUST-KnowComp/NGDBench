"""
更新查询模块实现
"""

from typing import Any, Dict, List
import networkx as nx
import random
import numpy as np
from .base import BaseQueryModule


class UpdateQueryModule(BaseQueryModule):
    """更新查询模块 - 生成各种写入和修改查询"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化更新查询模块
        
        Args:
            config: 查询配置参数
        """
        super().__init__(config)
        self.num_queries = config.get('num_queries', 15)
        self.query_distribution = config.get('query_distribution', {
            'node_operations': 0.4,
            'edge_operations': 0.4,
            'attribute_operations': 0.2
        })
        
    def generate_queries(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成更新查询集合"""
        queries = []
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        
        # 根据分布生成不同类型的查询
        for query_type, ratio in self.query_distribution.items():
            num_queries = int(self.num_queries * ratio)
            
            if query_type == 'node_operations':
                queries.extend(self._generate_node_operations(graph, nodes, num_queries))
            elif query_type == 'edge_operations':
                queries.extend(self._generate_edge_operations(graph, nodes, edges, num_queries))
            elif query_type == 'attribute_operations':
                queries.extend(self._generate_attribute_operations(graph, nodes, edges, num_queries))
        
        # 随机打乱查询顺序
        random.shuffle(queries)
        
        # 验证查询
        validated_queries = [q for q in queries if self.validate_query(q)]
        
        self._generated_queries = validated_queries
        return validated_queries
    
    def validate_query(self, query: Dict[str, Any]) -> bool:
        """验证查询的有效性"""
        required_fields = ['type', 'parameters', 'operation']
        
        for field in required_fields:
            if field not in query:
                return False
        
        # 检查查询类型是否支持
        if query['type'] not in self.get_supported_query_types():
            return False
        
        # 检查操作类型
        valid_operations = ['add', 'remove', 'update']
        if query['operation'] not in valid_operations:
            return False
        
        return True
    
    def get_supported_query_types(self) -> List[str]:
        """获取支持的查询类型"""
        return [
            'node_addition', 'node_removal', 'node_update',
            'edge_addition', 'edge_removal', 'edge_update',
            'attribute_update', 'batch_update', 'conditional_update'
        ]
    
    def _generate_node_operations(self, graph: nx.Graph, nodes: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成节点操作查询"""
        queries = []
        
        for _ in range(num_queries):
            operation = random.choice(['add', 'remove', 'update'])
            
            if operation == 'add':
                # 生成新节点ID
                new_node_id = self._generate_new_node_id(graph)
                attributes = self._generate_random_attributes()
                
                queries.append({
                    "type": "node_addition",
                    "operation": "add",
                    "parameters": {
                        "node": new_node_id,
                        "attributes": attributes
                    },
                    "description": f"添加新节点 {new_node_id}",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "reversible": True,
                    "reverse_operation": {
                        "type": "node_removal",
                        "operation": "remove",
                        "parameters": {"node": new_node_id}
                    }
                })
            
            elif operation == 'remove' and nodes:
                node_to_remove = random.choice(nodes)
                # 保存原始数据以便恢复
                original_data = dict(graph.nodes[node_to_remove]) if node_to_remove in graph.nodes else {}
                original_edges = [(u, v, graph.edges[u, v]) for u, v in graph.edges(node_to_remove)]
                
                queries.append({
                    "type": "node_removal",
                    "operation": "remove",
                    "parameters": {"node": node_to_remove},
                    "description": f"删除节点 {node_to_remove}",
                    "complexity": "medium",
                    "expected_result_type": "bool",
                    "reversible": True,
                    "reverse_operation": {
                        "type": "node_addition",
                        "operation": "add",
                        "parameters": {
                            "node": node_to_remove,
                            "attributes": original_data,
                            "edges": original_edges
                        }
                    }
                })
            
            elif operation == 'update' and nodes:
                node_to_update = random.choice(nodes)
                new_attributes = self._generate_random_attributes()
                
                queries.append({
                    "type": "node_update",
                    "operation": "update",
                    "parameters": {
                        "node": node_to_update,
                        "attributes": new_attributes
                    },
                    "description": f"更新节点 {node_to_update} 的属性",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "reversible": True
                })
        
        return queries
    
    def _generate_edge_operations(self, graph: nx.Graph, nodes: List, edges: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成边操作查询"""
        queries = []
        
        for _ in range(num_queries):
            operation = random.choice(['add', 'remove', 'update'])
            
            if operation == 'add' and len(nodes) >= 2:
                # 选择两个节点
                source, target = random.sample(nodes, 2)
                
                # 确保边不存在
                if not graph.has_edge(source, target):
                    attributes = self._generate_random_edge_attributes()
                    
                    queries.append({
                        "type": "edge_addition",
                        "operation": "add",
                        "parameters": {
                            "source": source,
                            "target": target,
                            "attributes": attributes
                        },
                        "description": f"添加边 {source}-{target}",
                        "complexity": "low",
                        "expected_result_type": "bool",
                        "reversible": True,
                        "reverse_operation": {
                            "type": "edge_removal",
                            "operation": "remove",
                            "parameters": {"source": source, "target": target}
                        }
                    })
            
            elif operation == 'remove' and edges:
                edge_to_remove = random.choice(edges)
                source, target = edge_to_remove[0], edge_to_remove[1]
                original_data = dict(graph.edges[edge_to_remove]) if graph.has_edge(source, target) else {}
                
                queries.append({
                    "type": "edge_removal",
                    "operation": "remove",
                    "parameters": {"source": source, "target": target},
                    "description": f"删除边 {source}-{target}",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "reversible": True,
                    "reverse_operation": {
                        "type": "edge_addition",
                        "operation": "add",
                        "parameters": {
                            "source": source,
                            "target": target,
                            "attributes": original_data
                        }
                    }
                })
            
            elif operation == 'update' and edges:
                edge_to_update = random.choice(edges)
                source, target = edge_to_update[0], edge_to_update[1]
                new_attributes = self._generate_random_edge_attributes()
                
                queries.append({
                    "type": "edge_update",
                    "operation": "update",
                    "parameters": {
                        "source": source,
                        "target": target,
                        "attributes": new_attributes
                    },
                    "description": f"更新边 {source}-{target} 的属性",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "reversible": True
                })
        
        return queries
    
    def _generate_attribute_operations(self, graph: nx.Graph, nodes: List, edges: List, num_queries: int) -> List[Dict[str, Any]]:
        """生成属性操作查询"""
        queries = []
        
        for _ in range(num_queries):
            target_type = random.choice(['node', 'edge'])
            operation_type = random.choice(['attribute_update', 'batch_update', 'conditional_update'])
            
            if operation_type == 'attribute_update':
                if target_type == 'node' and nodes:
                    node = random.choice(nodes)
                    attribute_name = random.choice(['value', 'category', 'weight', 'label'])
                    new_value = self._generate_attribute_value(attribute_name)
                    
                    queries.append({
                        "type": "attribute_update",
                        "operation": "update",
                        "parameters": {
                            "target_type": "node",
                            "target": node,
                            "attribute": attribute_name,
                            "value": new_value
                        },
                        "description": f"更新节点 {node} 的属性 {attribute_name}",
                        "complexity": "low",
                        "expected_result_type": "bool",
                        "reversible": True
                    })
                
                elif target_type == 'edge' and edges:
                    edge = random.choice(edges)
                    attribute_name = random.choice(['weight', 'type', 'strength'])
                    new_value = self._generate_attribute_value(attribute_name)
                    
                    queries.append({
                        "type": "attribute_update",
                        "operation": "update",
                        "parameters": {
                            "target_type": "edge",
                            "target": edge,
                            "attribute": attribute_name,
                            "value": new_value
                        },
                        "description": f"更新边 {edge[0]}-{edge[1]} 的属性 {attribute_name}",
                        "complexity": "low",
                        "expected_result_type": "bool",
                        "reversible": True
                    })
            
            elif operation_type == 'batch_update':
                if target_type == 'node' and nodes:
                    # 批量更新节点属性
                    sample_size = min(random.randint(3, 10), len(nodes))
                    selected_nodes = random.sample(nodes, sample_size)
                    attribute_name = random.choice(['category', 'status', 'group'])
                    new_value = self._generate_attribute_value(attribute_name)
                    
                    queries.append({
                        "type": "batch_update",
                        "operation": "update",
                        "parameters": {
                            "target_type": "node",
                            "targets": selected_nodes,
                            "attribute": attribute_name,
                            "value": new_value
                        },
                        "description": f"批量更新 {len(selected_nodes)} 个节点的属性 {attribute_name}",
                        "complexity": "medium",
                        "expected_result_type": "int",
                        "reversible": True
                    })
            
            elif operation_type == 'conditional_update':
                if target_type == 'node' and nodes:
                    # 条件更新节点属性
                    condition_attr = random.choice(['value', 'category'])
                    condition_value = self._generate_attribute_value(condition_attr)
                    update_attr = random.choice(['status', 'flag'])
                    update_value = self._generate_attribute_value(update_attr)
                    
                    queries.append({
                        "type": "conditional_update",
                        "operation": "update",
                        "parameters": {
                            "target_type": "node",
                            "condition": {
                                "attribute": condition_attr,
                                "operator": "equals",
                                "value": condition_value
                            },
                            "update": {
                                "attribute": update_attr,
                                "value": update_value
                            }
                        },
                        "description": f"条件更新：当 {condition_attr}={condition_value} 时，设置 {update_attr}={update_value}",
                        "complexity": "medium",
                        "expected_result_type": "int",
                        "reversible": False
                    })
        
        return queries
    
    def _generate_new_node_id(self, graph: nx.Graph) -> Any:
        """生成新的节点ID"""
        existing_nodes = set(graph.nodes())
        
        # 尝试生成数字ID
        for i in range(max(existing_nodes) + 1 if existing_nodes and all(isinstance(n, int) for n in existing_nodes) else 1000, 10000):
            if i not in existing_nodes:
                return i
        
        # 如果数字ID不可用，生成字符串ID
        for i in range(1000):
            new_id = f"new_node_{i}"
            if new_id not in existing_nodes:
                return new_id
        
        return f"new_node_{random.randint(10000, 99999)}"
    
    def _generate_random_attributes(self) -> Dict[str, Any]:
        """生成随机节点属性"""
        return {
            "value": random.randint(1, 100),
            "category": random.choice(["A", "B", "C", "D"]),
            "weight": round(random.uniform(0.1, 1.0), 2),
            "status": random.choice(["active", "inactive", "pending"]),
            "created_by": "update_query",
            "timestamp": random.randint(1000000000, 2000000000)
        }
    
    def _generate_random_edge_attributes(self) -> Dict[str, Any]:
        """生成随机边属性"""
        return {
            "weight": round(random.uniform(0.1, 1.0), 2),
            "type": random.choice(["friend", "colleague", "family", "business"]),
            "strength": random.choice(["weak", "medium", "strong"]),
            "created_by": "update_query",
            "timestamp": random.randint(1000000000, 2000000000)
        }
    
    def _generate_attribute_value(self, attribute_name: str) -> Any:
        """根据属性名生成合适的值"""
        if attribute_name in ["value", "score", "rank"]:
            return random.randint(1, 100)
        elif attribute_name in ["weight", "strength", "probability"]:
            return round(random.uniform(0.0, 1.0), 2)
        elif attribute_name in ["category", "type", "group"]:
            return random.choice(["A", "B", "C", "D", "E"])
        elif attribute_name in ["status", "state"]:
            return random.choice(["active", "inactive", "pending", "completed"])
        elif attribute_name in ["label", "name"]:
            return f"label_{random.randint(1, 1000)}"
        elif attribute_name == "flag":
            return random.choice([True, False])
        else:
            return f"value_{random.randint(1, 1000)}"
    
    def generate_stress_test_queries(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成压力测试查询"""
        stress_queries = []
        nodes = list(graph.nodes())
        
        # 大批量节点添加
        num_new_nodes = min(100, graph.number_of_nodes())
        new_node_ids = [self._generate_new_node_id(graph) + i for i in range(num_new_nodes)]
        
        stress_queries.append({
            "type": "batch_update",
            "operation": "add",
            "parameters": {
                "target_type": "node",
                "targets": new_node_ids,
                "attributes": self._generate_random_attributes()
            },
            "description": f"压力测试：批量添加 {num_new_nodes} 个节点",
            "complexity": "high",
            "expected_result_type": "int",
            "is_stress_test": True
        })
        
        # 大批量边添加
        if len(nodes) >= 2:
            num_new_edges = min(200, len(nodes) * (len(nodes) - 1) // 4)
            edge_pairs = []
            
            for _ in range(num_new_edges):
                source, target = random.sample(nodes, 2)
                if not graph.has_edge(source, target):
                    edge_pairs.append((source, target))
            
            stress_queries.append({
                "type": "batch_update",
                "operation": "add",
                "parameters": {
                    "target_type": "edge",
                    "targets": edge_pairs,
                    "attributes": self._generate_random_edge_attributes()
                },
                "description": f"压力测试：批量添加 {len(edge_pairs)} 条边",
                "complexity": "high",
                "expected_result_type": "int",
                "is_stress_test": True
            })
        
        return stress_queries
    
    def generate_consistency_test_queries(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """生成一致性测试查询"""
        consistency_queries = []
        nodes = list(graph.nodes())
        
        if len(nodes) >= 2:
            node1, node2 = random.sample(nodes, 2)
            
            # 测试事务一致性的查询序列
            consistency_queries.extend([
                {
                    "type": "node_update",
                    "operation": "update",
                    "parameters": {
                        "node": node1,
                        "attributes": {"status": "processing"}
                    },
                    "description": "一致性测试：开始处理",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "is_consistency_test": True,
                    "sequence_id": 1
                },
                {
                    "type": "edge_addition",
                    "operation": "add",
                    "parameters": {
                        "source": node1,
                        "target": node2,
                        "attributes": {"type": "processing_link"}
                    },
                    "description": "一致性测试：添加处理链接",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "is_consistency_test": True,
                    "sequence_id": 2
                },
                {
                    "type": "node_update",
                    "operation": "update",
                    "parameters": {
                        "node": node1,
                        "attributes": {"status": "completed"}
                    },
                    "description": "一致性测试：完成处理",
                    "complexity": "low",
                    "expected_result_type": "bool",
                    "is_consistency_test": True,
                    "sequence_id": 3
                }
            ])
        
        return consistency_queries
