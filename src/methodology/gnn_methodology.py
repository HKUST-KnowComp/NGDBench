"""
图神经网络(GNN)算法执行器实现
"""

from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np
from .base import BaseMethodology


class GNNMethodology(BaseMethodology):
    """图神经网络算法执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化GNN执行器
        
        Args:
            config: 算法配置参数
        """
        super().__init__(config)
        self.model_type = config.get('model_type', 'gcn')  # 'gcn', 'gat', 'sage', 'gin'
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.task_type = config.get('task_type', 'node_classification')  # 'node_classification', 'link_prediction', 'graph_classification'
        
        # 支持的模型类型
        self.supported_models = {'gcn', 'gat', 'sage', 'gin', 'simple_gnn'}
        
        if self.model_type not in self.supported_models:
            raise ValueError(f"不支持的GNN模型类型: {self.model_type}")
    
    def execute(self, graph: nx.Graph, queries: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """执行GNN算法"""
        if not self.validate_input(graph):
            raise ValueError("输入图数据验证失败")
        
        # 预处理图数据
        processed_graph = self.preprocess_graph(graph)
        
        # 准备特征和标签
        node_features, edge_features = self._prepare_features(processed_graph)
        
        # 构建邻接矩阵
        adjacency_matrix = self._build_adjacency_matrix(processed_graph)
        
        # 执行GNN模型
        model_results = self._execute_gnn_model(processed_graph, node_features, 
                                              edge_features, adjacency_matrix)
        
        # 处理查询
        query_results = {}
        if queries:
            query_results = self._process_gnn_queries(processed_graph, queries, model_results)
        
        results = {
            "model_results": model_results,
            "node_features": node_features,
            "query_results": query_results,
            "graph_stats": self._get_graph_statistics(processed_graph)
        }
        
        return self.postprocess_results(results)
    
    def validate_input(self, graph: nx.Graph) -> bool:
        """验证输入图数据"""
        if graph is None or graph.number_of_nodes() == 0:
            return False
        
        # 检查节点数量是否足够
        if graph.number_of_nodes() < 3:
            return False
        
        return True
    
    def get_supported_query_types(self) -> List[str]:
        """获取支持的查询类型"""
        return [
            'node_embedding', 'node_prediction', 'link_prediction',
            'graph_embedding', 'similarity_search', 'anomaly_detection'
        ]
    
    def _prepare_features(self, graph: nx.Graph) -> tuple:
        """准备节点和边特征"""
        # 准备节点特征
        node_features = {}
        feature_dim = self.config.get('feature_dim', 10)
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            # 提取结构特征
            degree = graph.degree(node)
            clustering_coeff = nx.clustering(graph, node)
            
            # 提取属性特征
            feature_vector = np.zeros(feature_dim)
            feature_vector[0] = degree / max(dict(graph.degree()).values()) if graph.degree() else 0
            feature_vector[1] = clustering_coeff
            
            # 如果有数值属性，使用它们
            if 'value' in node_data:
                feature_vector[2] = float(node_data['value']) / 100.0
            if 'weight' in node_data:
                feature_vector[3] = float(node_data['weight'])
            
            # 如果有类别属性，进行one-hot编码
            if 'category' in node_data:
                category_map = {'A': 4, 'B': 5, 'C': 6}
                if node_data['category'] in category_map:
                    feature_vector[category_map[node_data['category']]] = 1.0
            
            # 填充剩余维度
            for i in range(7, feature_dim):
                feature_vector[i] = np.random.normal(0, 0.1)
            
            node_features[node] = feature_vector
        
        # 准备边特征
        edge_features = {}
        edge_feature_dim = self.config.get('edge_feature_dim', 5)
        
        for edge in graph.edges():
            edge_data = graph.edges[edge]
            
            feature_vector = np.zeros(edge_feature_dim)
            
            # 边权重
            if 'weight' in edge_data:
                feature_vector[0] = float(edge_data['weight'])
            else:
                feature_vector[0] = 1.0
            
            # 边类型
            if 'type' in edge_data:
                type_map = {'friend': 1, 'colleague': 2, 'family': 3}
                if edge_data['type'] in type_map:
                    feature_vector[type_map[edge_data['type']]] = 1.0
            
            edge_features[edge] = feature_vector
        
        return node_features, edge_features
    
    def _build_adjacency_matrix(self, graph: nx.Graph) -> np.ndarray:
        """构建邻接矩阵"""
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        # 创建邻接矩阵
        adj_matrix = np.zeros((n, n))
        
        for edge in graph.edges():
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            weight = graph.edges[edge].get('weight', 1.0)
            adj_matrix[i, j] = weight
            
            if not graph.is_directed():
                adj_matrix[j, i] = weight
        
        return adj_matrix
    
    def _execute_gnn_model(self, graph: nx.Graph, node_features: Dict, 
                          edge_features: Dict, adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """执行GNN模型"""
        if self.model_type == 'simple_gnn':
            return self._execute_simple_gnn(graph, node_features, adjacency_matrix)
        elif self.model_type == 'gcn':
            return self._execute_gcn(graph, node_features, adjacency_matrix)
        elif self.model_type == 'gat':
            return self._execute_gat(graph, node_features, adjacency_matrix)
        elif self.model_type == 'sage':
            return self._execute_sage(graph, node_features, adjacency_matrix)
        elif self.model_type == 'gin':
            return self._execute_gin(graph, node_features, adjacency_matrix)
        else:
            raise ValueError(f"未实现的GNN模型: {self.model_type}")
    
    def _execute_simple_gnn(self, graph: nx.Graph, node_features: Dict, 
                           adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """执行简单的GNN模型（基于特征聚合）"""
        nodes = list(graph.nodes())
        node_embeddings = {}
        
        # 特征矩阵
        feature_matrix = np.array([node_features[node] for node in nodes])
        
        # 简单的消息传递：聚合邻居特征
        for layer in range(self.num_layers):
            new_features = np.zeros_like(feature_matrix)
            
            for i, node in enumerate(nodes):
                # 自身特征
                self_feature = feature_matrix[i]
                
                # 邻居特征聚合
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    neighbor_indices = [nodes.index(n) for n in neighbors if n in nodes]
                    neighbor_features = feature_matrix[neighbor_indices]
                    aggregated = np.mean(neighbor_features, axis=0)
                    
                    # 简单的更新规则
                    new_features[i] = 0.5 * self_feature + 0.5 * aggregated
                else:
                    new_features[i] = self_feature
            
            feature_matrix = new_features
        
        # 生成最终嵌入
        for i, node in enumerate(nodes):
            node_embeddings[node] = feature_matrix[i]
        
        # 执行任务特定的预测
        task_results = self._execute_task_prediction(graph, node_embeddings)
        
        return {
            "model_type": "simple_gnn",
            "node_embeddings": node_embeddings,
            "task_results": task_results,
            "num_layers": self.num_layers
        }
    
    def _execute_gcn(self, graph: nx.Graph, node_features: Dict, 
                    adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """执行图卷积网络(GCN)"""
        nodes = list(graph.nodes())
        
        # 特征矩阵
        X = np.array([node_features[node] for node in nodes])
        A = adjacency_matrix
        
        # 添加自环
        A_hat = A + np.eye(A.shape[0])
        
        # 度矩阵
        D = np.diag(np.sum(A_hat, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        
        # 归一化邻接矩阵
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        # 简化的GCN前向传播
        H = X
        for layer in range(self.num_layers):
            # H = σ(A_norm * H * W)
            # 这里使用随机权重矩阵作为简化
            W = np.random.normal(0, 0.1, (H.shape[1], self.hidden_dim))
            H = np.tanh(A_norm @ H @ W)
        
        # 生成节点嵌入
        node_embeddings = {nodes[i]: H[i] for i in range(len(nodes))}
        
        # 执行任务预测
        task_results = self._execute_task_prediction(graph, node_embeddings)
        
        return {
            "model_type": "gcn",
            "node_embeddings": node_embeddings,
            "task_results": task_results,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim
        }
    
    def _execute_gat(self, graph: nx.Graph, node_features: Dict, 
                    adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """执行图注意力网络(GAT)"""
        # 简化的GAT实现
        nodes = list(graph.nodes())
        X = np.array([node_features[node] for node in nodes])
        
        # 注意力机制的简化实现
        node_embeddings = {}
        
        for i, node in enumerate(nodes):
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                node_embeddings[node] = X[i]
                continue
            
            # 计算注意力权重（简化版）
            attention_weights = []
            neighbor_features = []
            
            for neighbor in neighbors:
                if neighbor in nodes:
                    j = nodes.index(neighbor)
                    # 简化的注意力计算
                    attention = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                    attention_weights.append(attention)
                    neighbor_features.append(X[j])
            
            if attention_weights:
                # 归一化注意力权重
                attention_weights = np.array(attention_weights)
                attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
                
                # 加权聚合
                aggregated = np.sum([w * f for w, f in zip(attention_weights, neighbor_features)], axis=0)
                node_embeddings[node] = 0.5 * X[i] + 0.5 * aggregated
            else:
                node_embeddings[node] = X[i]
        
        task_results = self._execute_task_prediction(graph, node_embeddings)
        
        return {
            "model_type": "gat",
            "node_embeddings": node_embeddings,
            "task_results": task_results,
            "attention_mechanism": "simplified"
        }
    
    def _execute_sage(self, graph: nx.Graph, node_features: Dict, 
                     adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """执行GraphSAGE"""
        nodes = list(graph.nodes())
        X = np.array([node_features[node] for node in nodes])
        
        node_embeddings = {}
        
        # GraphSAGE的采样和聚合
        for i, node in enumerate(nodes):
            neighbors = list(graph.neighbors(node))
            
            # 邻居采样（简化版：取所有邻居或随机采样）
            sample_size = min(len(neighbors), self.config.get('sample_size', 10))
            if len(neighbors) > sample_size:
                sampled_neighbors = np.random.choice(neighbors, sample_size, replace=False)
            else:
                sampled_neighbors = neighbors
            
            if sampled_neighbors:
                # 聚合邻居特征
                neighbor_indices = [nodes.index(n) for n in sampled_neighbors if n in nodes]
                if neighbor_indices:
                    neighbor_features = X[neighbor_indices]
                    # 使用均值聚合
                    aggregated = np.mean(neighbor_features, axis=0)
                    
                    # 连接自身特征和聚合特征
                    combined = np.concatenate([X[i], aggregated])
                    
                    # 简化的非线性变换
                    node_embeddings[node] = np.tanh(combined[:len(X[i])])
                else:
                    node_embeddings[node] = X[i]
            else:
                node_embeddings[node] = X[i]
        
        task_results = self._execute_task_prediction(graph, node_embeddings)
        
        return {
            "model_type": "sage",
            "node_embeddings": node_embeddings,
            "task_results": task_results,
            "sampling": True
        }
    
    def _execute_gin(self, graph: nx.Graph, node_features: Dict, 
                    adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """执行图同构网络(GIN)"""
        nodes = list(graph.nodes())
        X = np.array([node_features[node] for node in nodes])
        
        # GIN的更新规则：h_v^(k+1) = MLP((1 + ε) * h_v^(k) + Σ h_u^(k))
        epsilon = self.config.get('epsilon', 0.1)
        
        H = X
        for layer in range(self.num_layers):
            new_H = np.zeros_like(H)
            
            for i, node in enumerate(nodes):
                neighbors = list(graph.neighbors(node))
                
                # 聚合邻居特征
                if neighbors:
                    neighbor_indices = [nodes.index(n) for n in neighbors if n in nodes]
                    if neighbor_indices:
                        neighbor_sum = np.sum(H[neighbor_indices], axis=0)
                    else:
                        neighbor_sum = np.zeros_like(H[i])
                else:
                    neighbor_sum = np.zeros_like(H[i])
                
                # GIN更新规则
                updated = (1 + epsilon) * H[i] + neighbor_sum
                
                # 简化的MLP（单层）
                W = np.random.normal(0, 0.1, (len(updated), len(updated)))
                new_H[i] = np.tanh(updated @ W)
            
            H = new_H
        
        node_embeddings = {nodes[i]: H[i] for i in range(len(nodes))}
        task_results = self._execute_task_prediction(graph, node_embeddings)
        
        return {
            "model_type": "gin",
            "node_embeddings": node_embeddings,
            "task_results": task_results,
            "epsilon": epsilon
        }
    
    def _execute_task_prediction(self, graph: nx.Graph, node_embeddings: Dict) -> Dict[str, Any]:
        """执行任务特定的预测"""
        if self.task_type == 'node_classification':
            return self._node_classification_task(graph, node_embeddings)
        elif self.task_type == 'link_prediction':
            return self._link_prediction_task(graph, node_embeddings)
        elif self.task_type == 'graph_classification':
            return self._graph_classification_task(graph, node_embeddings)
        else:
            return {"task_type": self.task_type, "status": "not_implemented"}
    
    def _node_classification_task(self, graph: nx.Graph, node_embeddings: Dict) -> Dict[str, Any]:
        """节点分类任务"""
        # 基于嵌入的简单分类
        classifications = {}
        embedding_values = list(node_embeddings.values())
        
        if embedding_values:
            # 使用嵌入的第一个维度进行分类
            first_dim_values = [emb[0] for emb in embedding_values]
            threshold_low = np.percentile(first_dim_values, 33)
            threshold_high = np.percentile(first_dim_values, 67)
            
            for node, embedding in node_embeddings.items():
                if embedding[0] <= threshold_low:
                    classifications[node] = "class_0"
                elif embedding[0] >= threshold_high:
                    classifications[node] = "class_2"
                else:
                    classifications[node] = "class_1"
        
        return {
            "task": "node_classification",
            "predictions": classifications,
            "num_classes": 3,
            "method": "embedding_based"
        }
    
    def _link_prediction_task(self, graph: nx.Graph, node_embeddings: Dict) -> Dict[str, Any]:
        """链接预测任务"""
        predictions = []
        nodes = list(node_embeddings.keys())
        
        # 预测不存在的边
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if not graph.has_edge(node1, node2):
                    # 计算节点嵌入的相似度
                    emb1, emb2 = node_embeddings[node1], node_embeddings[node2]
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    predictions.append({
                        "node_pair": (node1, node2),
                        "link_probability": float(similarity),
                        "predicted": similarity > 0.5
                    })
        
        # 按概率排序
        predictions.sort(key=lambda x: x['link_probability'], reverse=True)
        
        return {
            "task": "link_prediction",
            "predictions": predictions[:20],  # 返回前20个预测
            "total_predictions": len(predictions),
            "method": "embedding_similarity"
        }
    
    def _graph_classification_task(self, graph: nx.Graph, node_embeddings: Dict) -> Dict[str, Any]:
        """图分类任务"""
        # 图级别的表示：节点嵌入的聚合
        if node_embeddings:
            embeddings = list(node_embeddings.values())
            
            # 不同的聚合方法
            graph_embedding_mean = np.mean(embeddings, axis=0)
            graph_embedding_max = np.max(embeddings, axis=0)
            graph_embedding_sum = np.sum(embeddings, axis=0)
            
            # 基于图嵌入的简单分类
            classification_score = np.mean(graph_embedding_mean)
            
            if classification_score > 0.1:
                graph_class = "positive"
            elif classification_score < -0.1:
                graph_class = "negative"
            else:
                graph_class = "neutral"
            
            return {
                "task": "graph_classification",
                "graph_embedding": {
                    "mean": graph_embedding_mean.tolist(),
                    "max": graph_embedding_max.tolist(),
                    "sum": graph_embedding_sum.tolist()
                },
                "predicted_class": graph_class,
                "classification_score": float(classification_score)
            }
        else:
            return {
                "task": "graph_classification",
                "error": "No node embeddings available"
            }
    
    def _process_gnn_queries(self, graph: nx.Graph, queries: List[Dict],
                            model_results: Dict[str, Any]) -> Dict[str, Any]:
        """处理GNN查询"""
        query_results = {}
        
        for i, query in enumerate(queries):
            if not self.is_query_supported(query):
                query_results[f"query_{i}"] = {"error": f"不支持的查询类型: {query.get('type')}"}
                continue
            
            try:
                result = self._execute_gnn_query(graph, query, model_results)
                query_results[f"query_{i}"] = result
            except Exception as e:
                query_results[f"query_{i}"] = {"error": str(e)}
        
        return query_results
    
    def _execute_gnn_query(self, graph: nx.Graph, query: Dict[str, Any],
                          model_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个GNN查询"""
        query_type = query['type']
        
        if query_type == 'node_embedding':
            return self._query_node_embedding(model_results, query)
        elif query_type == 'node_prediction':
            return self._query_node_prediction(model_results, query)
        elif query_type == 'link_prediction':
            return self._query_link_prediction(model_results, query)
        elif query_type == 'graph_embedding':
            return self._query_graph_embedding(model_results, query)
        elif query_type == 'similarity_search':
            return self._query_embedding_similarity(model_results, query)
        elif query_type == 'anomaly_detection':
            return self._query_anomaly_detection(model_results, query)
        else:
            raise ValueError(f"未实现的查询类型: {query_type}")
    
    def _query_node_embedding(self, model_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """节点嵌入查询"""
        node = query.get('node')
        node_embeddings = model_results.get('node_embeddings', {})
        
        if node and node in node_embeddings:
            return {
                "type": "node_embedding",
                "node": node,
                "embedding": node_embeddings[node].tolist(),
                "embedding_dim": len(node_embeddings[node])
            }
        else:
            return {"error": f"节点 {node} 的嵌入不存在"}
    
    def _query_node_prediction(self, model_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """节点预测查询"""
        node = query.get('node')
        task_results = model_results.get('task_results', {})
        
        if 'predictions' in task_results:
            predictions = task_results['predictions']
            if node in predictions:
                return {
                    "type": "node_prediction",
                    "node": node,
                    "prediction": predictions[node],
                    "task": task_results.get('task', 'unknown')
                }
        
        return {"error": f"节点 {node} 的预测结果不存在"}
    
    def _query_link_prediction(self, model_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """链接预测查询"""
        source = query.get('source')
        target = query.get('target')
        task_results = model_results.get('task_results', {})
        
        if task_results.get('task') == 'link_prediction':
            predictions = task_results.get('predictions', [])
            
            # 查找特定节点对的预测
            for pred in predictions:
                node_pair = pred['node_pair']
                if (node_pair[0] == source and node_pair[1] == target) or \
                   (node_pair[0] == target and node_pair[1] == source):
                    return {
                        "type": "link_prediction",
                        "source": source,
                        "target": target,
                        "link_probability": pred['link_probability'],
                        "predicted": pred['predicted']
                    }
            
            return {
                "type": "link_prediction",
                "source": source,
                "target": target,
                "error": "No prediction found for this node pair"
            }
        else:
            return {"error": "链接预测任务未执行"}
    
    def _query_graph_embedding(self, model_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """图嵌入查询"""
        task_results = model_results.get('task_results', {})
        
        if 'graph_embedding' in task_results:
            return {
                "type": "graph_embedding",
                "graph_embedding": task_results['graph_embedding']
            }
        else:
            # 如果没有图级别嵌入，计算节点嵌入的聚合
            node_embeddings = model_results.get('node_embeddings', {})
            if node_embeddings:
                embeddings = list(node_embeddings.values())
                graph_emb = np.mean(embeddings, axis=0)
                return {
                    "type": "graph_embedding",
                    "graph_embedding": graph_emb.tolist(),
                    "method": "node_embedding_aggregation"
                }
            else:
                return {"error": "无法计算图嵌入"}
    
    def _query_embedding_similarity(self, model_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """嵌入相似性查询"""
        target_node = query.get('target_node')
        top_k = query.get('k', 5)
        node_embeddings = model_results.get('node_embeddings', {})
        
        if target_node not in node_embeddings:
            return {"error": f"目标节点 {target_node} 的嵌入不存在"}
        
        target_embedding = node_embeddings[target_node]
        similarities = []
        
        for node, embedding in node_embeddings.items():
            if node != target_node:
                similarity = np.dot(target_embedding, embedding) / \
                           (np.linalg.norm(target_embedding) * np.linalg.norm(embedding))
                similarities.append((node, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "type": "similarity_search",
            "target_node": target_node,
            "similar_nodes": similarities[:top_k],
            "k": top_k
        }
    
    def _query_anomaly_detection(self, model_results: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        """异常检测查询"""
        threshold = query.get('threshold', 0.1)
        node_embeddings = model_results.get('node_embeddings', {})
        
        if not node_embeddings:
            return {"error": "没有节点嵌入用于异常检测"}
        
        # 计算每个节点嵌入与平均嵌入的距离
        embeddings = list(node_embeddings.values())
        mean_embedding = np.mean(embeddings, axis=0)
        
        anomalies = []
        for node, embedding in node_embeddings.items():
            distance = np.linalg.norm(embedding - mean_embedding)
            if distance > threshold:
                anomalies.append({
                    "node": node,
                    "anomaly_score": float(distance),
                    "is_anomaly": True
                })
        
        # 按异常分数排序
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return {
            "type": "anomaly_detection",
            "threshold": threshold,
            "anomalies": anomalies,
            "num_anomalies": len(anomalies)
        }
    
    def _get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed()
        }
