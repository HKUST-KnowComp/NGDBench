"""
GraphRAG算法执行器实现
"""

from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np
from .base import BaseMethodology


class GraphRAGMethodology(BaseMethodology):
    """GraphRAG算法执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化GraphRAG执行器
        
        Args:
            config: 算法配置参数
        """
        super().__init__(config)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.retrieval_method = config.get('retrieval_method', 'similarity')
        self.generation_method = config.get('generation_method', 'template')
        
    def execute(self, graph: nx.Graph, queries: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """执行GraphRAG算法"""
        if not self.validate_input(graph):
            raise ValueError("输入图数据验证失败")
        
        # 预处理图数据
        processed_graph = self.preprocess_graph(graph)
        
        # 构建知识图谱表示
        kg_representation = self._build_knowledge_graph(processed_graph)
        
        # 生成节点嵌入
        node_embeddings = self._generate_node_embeddings(processed_graph)
        
        # 处理查询
        query_results = {}
        if queries:
            query_results = self._process_rag_queries(processed_graph, queries, 
                                                    kg_representation, node_embeddings)
        
        results = {
            "kg_representation": kg_representation,
            "node_embeddings": node_embeddings,
            "query_results": query_results,
            "graph_stats": self._get_graph_statistics(processed_graph)
        }
        
        return self.postprocess_results(results)
    
    def validate_input(self, graph: nx.Graph) -> bool:
        """验证输入图数据"""
        if graph is None or graph.number_of_nodes() == 0:
            return False
        return True
    
    def get_supported_query_types(self) -> List[str]:
        """获取支持的查询类型"""
        return [
            'entity_retrieval', 'relation_query', 'path_reasoning',
            'subgraph_qa', 'similarity_search', 'knowledge_completion'
        ]
    
    def _build_knowledge_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """构建知识图谱表示"""
        # 提取实体和关系
        entities = {}
        relations = {}
        
        # 处理节点（实体）
        for node, data in graph.nodes(data=True):
            entities[node] = {
                "id": node,
                "type": data.get("type", "unknown"),
                "attributes": data,
                "description": self._generate_entity_description(node, data)
            }
        
        # 处理边（关系）
        for source, target, data in graph.edges(data=True):
            relation_id = f"{source}-{target}"
            relations[relation_id] = {
                "id": relation_id,
                "source": source,
                "target": target,
                "type": data.get("type", "related_to"),
                "attributes": data,
                "description": self._generate_relation_description(source, target, data)
            }
        
        return {
            "entities": entities,
            "relations": relations,
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
    
    def _generate_entity_description(self, node: Any, data: Dict[str, Any]) -> str:
        """生成实体描述"""
        entity_type = data.get("type", "entity")
        description = f"This is a {entity_type} with ID {node}."
        
        # 添加属性信息
        for key, value in data.items():
            if key != "type":
                description += f" {key}: {value}."
        
        return description
    
    def _generate_relation_description(self, source: Any, target: Any, data: Dict[str, Any]) -> str:
        """生成关系描述"""
        relation_type = data.get("type", "is related to")
        description = f"Entity {source} {relation_type} entity {target}."
        
        # 添加关系属性
        for key, value in data.items():
            if key != "type":
                description += f" {key}: {value}."
        
        return description
    
    def _generate_node_embeddings(self, graph: nx.Graph) -> Dict[str, np.ndarray]:
        """生成节点嵌入向量"""
        embeddings = {}
        
        # 使用简单的结构特征生成嵌入
        for node in graph.nodes():
            # 计算结构特征
            degree = graph.degree(node)
            neighbors = list(graph.neighbors(node))
            
            # 生成基础嵌入向量
            embedding = np.random.normal(0, 1, self.embedding_dim)
            
            # 基于度数调整嵌入
            embedding[0] = degree / max(dict(graph.degree()).values())
            
            # 基于邻居数量调整
            embedding[1] = len(neighbors) / graph.number_of_nodes()
            
            # 基于节点属性调整（如果有）
            node_data = graph.nodes[node]
            if 'value' in node_data:
                embedding[2] = float(node_data['value']) / 100.0
            
            embeddings[node] = embedding
        
        return embeddings
    
    def _process_rag_queries(self, graph: nx.Graph, queries: List[Dict],
                           kg_representation: Dict[str, Any], 
                           node_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """处理RAG查询"""
        query_results = {}
        
        for i, query in enumerate(queries):
            if not self.is_query_supported(query):
                query_results[f"query_{i}"] = {"error": f"不支持的查询类型: {query.get('type')}"}
                continue
            
            try:
                result = self._execute_rag_query(graph, query, kg_representation, node_embeddings)
                query_results[f"query_{i}"] = result
            except Exception as e:
                query_results[f"query_{i}"] = {"error": str(e)}
        
        return query_results
    
    def _execute_rag_query(self, graph: nx.Graph, query: Dict[str, Any],
                          kg_representation: Dict[str, Any],
                          node_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """执行单个RAG查询"""
        query_type = query['type']
        
        if query_type == 'entity_retrieval':
            return self._query_entity_retrieval(kg_representation, query)
        elif query_type == 'relation_query':
            return self._query_relation(kg_representation, query)
        elif query_type == 'path_reasoning':
            return self._query_path_reasoning(graph, kg_representation, query)
        elif query_type == 'subgraph_qa':
            return self._query_subgraph_qa(graph, kg_representation, query)
        elif query_type == 'similarity_search':
            return self._query_similarity_search(node_embeddings, query)
        elif query_type == 'knowledge_completion':
            return self._query_knowledge_completion(graph, kg_representation, query)
        else:
            raise ValueError(f"未实现的查询类型: {query_type}")
    
    def _query_entity_retrieval(self, kg_representation: Dict[str, Any], 
                               query: Dict[str, Any]) -> Dict[str, Any]:
        """实体检索查询"""
        entity_type = query.get('entity_type')
        attributes = query.get('attributes', {})
        
        entities = kg_representation['entities']
        matching_entities = []
        
        for entity_id, entity_data in entities.items():
            # 检查类型匹配
            if entity_type and entity_data.get('type') != entity_type:
                continue
            
            # 检查属性匹配
            match = True
            for attr_name, attr_value in attributes.items():
                if entity_data['attributes'].get(attr_name) != attr_value:
                    match = False
                    break
            
            if match:
                matching_entities.append({
                    "entity_id": entity_id,
                    "entity_data": entity_data,
                    "relevance_score": self._calculate_entity_relevance(entity_data, query)
                })
        
        # 按相关性排序
        matching_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            "type": "entity_retrieval",
            "query": query,
            "matching_entities": matching_entities,
            "count": len(matching_entities)
        }
    
    def _query_relation(self, kg_representation: Dict[str, Any], 
                       query: Dict[str, Any]) -> Dict[str, Any]:
        """关系查询"""
        source_entity = query.get('source_entity')
        target_entity = query.get('target_entity')
        relation_type = query.get('relation_type')
        
        relations = kg_representation['relations']
        matching_relations = []
        
        for relation_id, relation_data in relations.items():
            # 检查源实体匹配
            if source_entity and relation_data['source'] != source_entity:
                continue
            
            # 检查目标实体匹配
            if target_entity and relation_data['target'] != target_entity:
                continue
            
            # 检查关系类型匹配
            if relation_type and relation_data.get('type') != relation_type:
                continue
            
            matching_relations.append({
                "relation_id": relation_id,
                "relation_data": relation_data,
                "confidence": self._calculate_relation_confidence(relation_data, query)
            })
        
        return {
            "type": "relation_query",
            "query": query,
            "matching_relations": matching_relations,
            "count": len(matching_relations)
        }
    
    def _query_path_reasoning(self, graph: nx.Graph, kg_representation: Dict[str, Any],
                             query: Dict[str, Any]) -> Dict[str, Any]:
        """路径推理查询"""
        source = query.get('source')
        target = query.get('target')
        max_length = query.get('max_length', 3)
        
        if not source or not target:
            return {"error": "缺少源节点或目标节点"}
        
        # 找到所有可能的路径
        try:
            all_paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
            
            # 为每条路径生成推理解释
            reasoning_paths = []
            for path in all_paths:
                reasoning = self._generate_path_reasoning(path, kg_representation)
                reasoning_paths.append({
                    "path": path,
                    "length": len(path) - 1,
                    "reasoning": reasoning,
                    "confidence": self._calculate_path_confidence(path, kg_representation)
                })
            
            # 按置信度排序
            reasoning_paths.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "type": "path_reasoning",
                "source": source,
                "target": target,
                "reasoning_paths": reasoning_paths,
                "count": len(reasoning_paths)
            }
        
        except nx.NetworkXNoPath:
            return {
                "type": "path_reasoning",
                "source": source,
                "target": target,
                "reasoning_paths": [],
                "count": 0,
                "message": "No path found"
            }
    
    def _query_subgraph_qa(self, graph: nx.Graph, kg_representation: Dict[str, Any],
                          query: Dict[str, Any]) -> Dict[str, Any]:
        """子图问答查询"""
        center_nodes = query.get('center_nodes', [])
        radius = query.get('radius', 1)
        question = query.get('question', '')
        
        if not center_nodes:
            return {"error": "缺少中心节点"}
        
        # 构建子图
        subgraph_nodes = set(center_nodes)
        for node in center_nodes:
            if node in graph:
                # 添加邻域节点
                neighbors = list(graph.neighbors(node))
                for _ in range(radius):
                    new_neighbors = []
                    for n in neighbors:
                        new_neighbors.extend(list(graph.neighbors(n)))
                    neighbors.extend(new_neighbors)
                    neighbors = list(set(neighbors))
                
                subgraph_nodes.update(neighbors[:20])  # 限制子图大小
        
        subgraph = graph.subgraph(subgraph_nodes)
        
        # 生成答案
        answer = self._generate_subgraph_answer(subgraph, kg_representation, question)
        
        return {
            "type": "subgraph_qa",
            "center_nodes": center_nodes,
            "subgraph_size": subgraph.number_of_nodes(),
            "question": question,
            "answer": answer,
            "confidence": 0.8  # 简化的置信度
        }
    
    def _query_similarity_search(self, node_embeddings: Dict[str, np.ndarray],
                                query: Dict[str, Any]) -> Dict[str, Any]:
        """相似性搜索查询"""
        target_node = query.get('target_node')
        top_k = query.get('k', 5)
        
        if not target_node or target_node not in node_embeddings:
            return {"error": "目标节点不存在或没有嵌入"}
        
        target_embedding = node_embeddings[target_node]
        similarities = []
        
        for node, embedding in node_embeddings.items():
            if node != target_node:
                # 计算余弦相似度
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((node, float(similarity)))
        
        # 排序并取前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]
        
        return {
            "type": "similarity_search",
            "target_node": target_node,
            "similar_nodes": top_similar,
            "k": top_k
        }
    
    def _query_knowledge_completion(self, graph: nx.Graph, kg_representation: Dict[str, Any],
                                   query: Dict[str, Any]) -> Dict[str, Any]:
        """知识补全查询"""
        incomplete_triple = query.get('incomplete_triple', {})
        completion_type = query.get('completion_type', 'object')  # 'subject', 'predicate', 'object'
        
        subject = incomplete_triple.get('subject')
        predicate = incomplete_triple.get('predicate')
        obj = incomplete_triple.get('object')
        
        candidates = []
        
        if completion_type == 'object' and subject:
            # 预测对象
            if subject in graph:
                neighbors = list(graph.neighbors(subject))
                for neighbor in neighbors:
                    edge_data = graph.edges[subject, neighbor] if graph.has_edge(subject, neighbor) else {}
                    if not predicate or edge_data.get('type') == predicate:
                        confidence = self._calculate_completion_confidence(subject, predicate, neighbor, graph)
                        candidates.append({
                            "completion": neighbor,
                            "confidence": confidence,
                            "explanation": f"Entity {subject} is connected to {neighbor}"
                        })
        
        elif completion_type == 'subject' and obj:
            # 预测主体
            if obj in graph:
                predecessors = list(graph.predecessors(obj)) if graph.is_directed() else list(graph.neighbors(obj))
                for pred in predecessors:
                    edge_data = graph.edges[pred, obj] if graph.has_edge(pred, obj) else {}
                    if not predicate or edge_data.get('type') == predicate:
                        confidence = self._calculate_completion_confidence(pred, predicate, obj, graph)
                        candidates.append({
                            "completion": pred,
                            "confidence": confidence,
                            "explanation": f"Entity {pred} is connected to {obj}"
                        })
        
        # 按置信度排序
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "type": "knowledge_completion",
            "incomplete_triple": incomplete_triple,
            "completion_type": completion_type,
            "candidates": candidates[:10],  # 返回前10个候选
            "count": len(candidates)
        }
    
    def _calculate_entity_relevance(self, entity_data: Dict[str, Any], query: Dict[str, Any]) -> float:
        """计算实体相关性分数"""
        # 简化的相关性计算
        base_score = 0.5
        
        # 基于类型匹配
        if query.get('entity_type') == entity_data.get('type'):
            base_score += 0.3
        
        # 基于属性匹配数量
        query_attrs = query.get('attributes', {})
        entity_attrs = entity_data.get('attributes', {})
        
        matching_attrs = sum(1 for k, v in query_attrs.items() if entity_attrs.get(k) == v)
        if query_attrs:
            base_score += 0.2 * (matching_attrs / len(query_attrs))
        
        return min(base_score, 1.0)
    
    def _calculate_relation_confidence(self, relation_data: Dict[str, Any], query: Dict[str, Any]) -> float:
        """计算关系置信度"""
        # 简化的置信度计算
        return 0.8
    
    def _generate_path_reasoning(self, path: List, kg_representation: Dict[str, Any]) -> str:
        """生成路径推理解释"""
        if len(path) < 2:
            return "Empty path"
        
        reasoning = f"Starting from {path[0]}"
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            relation_id = f"{source}-{target}"
            
            if relation_id in kg_representation['relations']:
                relation_type = kg_representation['relations'][relation_id].get('type', 'connected to')
                reasoning += f", then {relation_type} {target}"
            else:
                reasoning += f", then connected to {target}"
        
        return reasoning
    
    def _calculate_path_confidence(self, path: List, kg_representation: Dict[str, Any]) -> float:
        """计算路径置信度"""
        if len(path) < 2:
            return 0.0
        
        # 基于路径长度的置信度（短路径置信度更高）
        length_penalty = 1.0 / len(path)
        
        # 基于关系存在性的置信度
        relation_confidence = 0.0
        for i in range(len(path) - 1):
            relation_id = f"{path[i]}-{path[i + 1]}"
            if relation_id in kg_representation['relations']:
                relation_confidence += 1.0
        
        relation_confidence /= (len(path) - 1)
        
        return (length_penalty + relation_confidence) / 2.0
    
    def _generate_subgraph_answer(self, subgraph: nx.Graph, kg_representation: Dict[str, Any], 
                                 question: str) -> str:
        """生成子图问答答案"""
        # 简化的答案生成
        entities = [kg_representation['entities'][node] for node in subgraph.nodes() 
                   if node in kg_representation['entities']]
        
        if not entities:
            return "No relevant information found in the subgraph."
        
        # 基于问题类型生成不同的答案
        if "how many" in question.lower():
            return f"There are {len(entities)} entities in the relevant subgraph."
        elif "what" in question.lower():
            entity_types = [e.get('type', 'unknown') for e in entities]
            unique_types = list(set(entity_types))
            return f"The subgraph contains entities of types: {', '.join(unique_types)}."
        else:
            return f"The subgraph contains {len(entities)} entities and {subgraph.number_of_edges()} relationships."
    
    def _calculate_completion_confidence(self, subject: Any, predicate: Any, obj: Any, 
                                       graph: nx.Graph) -> float:
        """计算知识补全置信度"""
        # 基于图结构的简化置信度计算
        if graph.has_edge(subject, obj):
            return 0.9
        else:
            # 基于共同邻居计算置信度
            subject_neighbors = set(graph.neighbors(subject)) if subject in graph else set()
            obj_neighbors = set(graph.neighbors(obj)) if obj in graph else set()
            common_neighbors = len(subject_neighbors.intersection(obj_neighbors))
            
            if subject_neighbors and obj_neighbors:
                return min(0.8, common_neighbors / min(len(subject_neighbors), len(obj_neighbors)))
            else:
                return 0.1
    
    def _get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed()
        }
