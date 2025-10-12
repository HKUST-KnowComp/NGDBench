"""
准确性评估器实现
"""

from typing import Any, Dict, List
import numpy as np
from .base import BaseEvaluationHarness


class AccuracyEvaluator(BaseEvaluationHarness):
    """准确性评估器 - 评估算法结果的准确性"""
    
    def evaluate(self, ground_truth: Dict[str, Any], 
                algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """执行准确性评估"""
        evaluation_result = {
            "evaluation_type": "accuracy",
            "comparisons": [],
            "metrics": {},
            "summary": {}
        }
        
        # 比较不同类型的结果
        comparisons = []
        
        # 比较算法结果
        if 'algorithm_results' in ground_truth and 'algorithm_results' in algorithm_output:
            algo_comparison = self._evaluate_algorithm_results(
                ground_truth['algorithm_results'],
                algorithm_output['algorithm_results']
            )
            comparisons.extend(algo_comparison)
        
        # 比较查询结果
        if 'query_results' in ground_truth and 'query_results' in algorithm_output:
            query_comparison = self._evaluate_query_results(
                ground_truth['query_results'],
                algorithm_output['query_results']
            )
            comparisons.extend(query_comparison)
        
        evaluation_result['comparisons'] = comparisons
        evaluation_result['num_comparisons'] = len(comparisons)
        
        # 计算指标
        if comparisons:
            evaluation_result['metrics'] = self.calculate_metrics(comparisons)
        
        # 生成摘要
        evaluation_result['summary'] = self._generate_accuracy_summary(evaluation_result)
        
        # 记录评估
        self.record_evaluation(evaluation_result)
        
        return evaluation_result
    
    def _evaluate_algorithm_results(self, ground_truth: Dict[str, Any], 
                                   algorithm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估算法结果的准确性"""
        comparisons = []
        
        # 比较不同类型的算法结果
        result_types = ['scores', 'rankings', 'classifications', 'embeddings']
        
        for result_type in result_types:
            if result_type in ground_truth and result_type in algorithm_output:
                comparison = self._compare_algorithm_result_type(
                    ground_truth[result_type],
                    algorithm_output[result_type],
                    result_type
                )
                comparison['result_type'] = result_type
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_algorithm_result_type(self, gt_result: Any, algo_result: Any, 
                                     result_type: str) -> Dict[str, Any]:
        """比较特定类型的算法结果"""
        if result_type == 'scores':
            return self._compare_scores(gt_result, algo_result)
        elif result_type == 'rankings':
            return self._compare_rankings(gt_result, algo_result)
        elif result_type == 'classifications':
            return self._compare_classifications(gt_result, algo_result)
        elif result_type == 'embeddings':
            return self._compare_embeddings(gt_result, algo_result)
        else:
            return self.compare_results(gt_result, algo_result, 'approximate')
    
    def _compare_scores(self, gt_scores: Dict, algo_scores: Dict) -> Dict[str, Any]:
        """比较分数结果"""
        common_keys = set(gt_scores.keys()) & set(algo_scores.keys())
        
        if not common_keys:
            return {
                "match": False,
                "accuracy": 0.0,
                "error": "No common keys found"
            }
        
        # 计算数值准确性
        absolute_errors = []
        relative_errors = []
        matches = 0
        
        for key in common_keys:
            gt_val = float(gt_scores[key])
            algo_val = float(algo_scores[key])
            
            abs_error = abs(gt_val - algo_val)
            rel_error = abs_error / max(abs(gt_val), 1e-10)
            
            absolute_errors.append(abs_error)
            relative_errors.append(rel_error)
            
            if abs_error <= self.tolerance:
                matches += 1
        
        # 计算排名相关性
        gt_ranking = sorted(common_keys, key=lambda x: gt_scores[x], reverse=True)
        algo_ranking = sorted(common_keys, key=lambda x: algo_scores[x], reverse=True)
        rank_correlation = self._calculate_rank_correlation(gt_ranking, algo_ranking)
        
        return {
            "match": matches == len(common_keys),
            "accuracy": matches / len(common_keys),
            "mean_absolute_error": np.mean(absolute_errors),
            "mean_relative_error": np.mean(relative_errors),
            "rank_correlation": rank_correlation,
            "num_compared": len(common_keys)
        }
    
    def _compare_rankings(self, gt_ranking: List, algo_ranking: List) -> Dict[str, Any]:
        """比较排名结果"""
        if isinstance(gt_ranking[0], tuple):
            # 排名是 (item, score) 的形式
            gt_items = [item for item, _ in gt_ranking]
            algo_items = [item for item, _ in algo_ranking]
        else:
            gt_items = gt_ranking
            algo_items = algo_ranking
        
        # 计算排名相关性
        rank_correlation = self._calculate_rank_correlation(gt_items, algo_items)
        
        # 计算Top-K准确性
        top_k_accuracies = {}
        for k in [1, 5, 10, min(len(gt_items), len(algo_items))]:
            if k <= len(gt_items) and k <= len(algo_items):
                gt_top_k = set(gt_items[:k])
                algo_top_k = set(algo_items[:k])
                top_k_accuracies[f'top_{k}_accuracy'] = len(gt_top_k & algo_top_k) / k
        
        return {
            "match": gt_items == algo_items,
            "rank_correlation": rank_correlation,
            **top_k_accuracies,
            "length_match": len(gt_items) == len(algo_items)
        }
    
    def _compare_classifications(self, gt_classes: Dict, algo_classes: Dict) -> Dict[str, Any]:
        """比较分类结果"""
        common_keys = set(gt_classes.keys()) & set(algo_classes.keys())
        
        if not common_keys:
            return {
                "match": False,
                "accuracy": 0.0,
                "error": "No common keys found"
            }
        
        # 计算分类准确性
        correct = sum(1 for key in common_keys if gt_classes[key] == algo_classes[key])
        accuracy = correct / len(common_keys)
        
        # 计算混淆矩阵相关指标
        gt_labels = [gt_classes[key] for key in common_keys]
        algo_labels = [algo_classes[key] for key in common_keys]
        
        unique_labels = list(set(gt_labels + algo_labels))
        confusion_matrix = self._calculate_confusion_matrix(gt_labels, algo_labels, unique_labels)
        
        # 计算精确率、召回率、F1分数
        precision, recall, f1 = self._calculate_classification_metrics(confusion_matrix, unique_labels)
        
        return {
            "match": correct == len(common_keys),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix,
            "num_compared": len(common_keys)
        }
    
    def _compare_embeddings(self, gt_embeddings: Dict, algo_embeddings: Dict) -> Dict[str, Any]:
        """比较嵌入向量结果"""
        common_keys = set(gt_embeddings.keys()) & set(algo_embeddings.keys())
        
        if not common_keys:
            return {
                "match": False,
                "accuracy": 0.0,
                "error": "No common keys found"
            }
        
        # 计算余弦相似度
        cosine_similarities = []
        euclidean_distances = []
        
        for key in common_keys:
            gt_emb = np.array(gt_embeddings[key])
            algo_emb = np.array(algo_embeddings[key])
            
            # 余弦相似度
            cos_sim = np.dot(gt_emb, algo_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(algo_emb))
            cosine_similarities.append(cos_sim)
            
            # 欧几里得距离
            eucl_dist = np.linalg.norm(gt_emb - algo_emb)
            euclidean_distances.append(eucl_dist)
        
        return {
            "mean_cosine_similarity": np.mean(cosine_similarities),
            "mean_euclidean_distance": np.mean(euclidean_distances),
            "embedding_correlation": np.mean(cosine_similarities),
            "num_compared": len(common_keys)
        }
    
    def _evaluate_query_results(self, ground_truth: Dict[str, Any], 
                               algorithm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估查询结果的准确性"""
        comparisons = []
        
        # 比较每个查询的结果
        common_queries = set(ground_truth.keys()) & set(algorithm_output.keys())
        
        for query_id in common_queries:
            gt_result = ground_truth[query_id]
            algo_result = algorithm_output[query_id]
            
            comparison = self.compare_results(gt_result, algo_result, 'approximate')
            comparison['query_id'] = query_id
            comparison['query_type'] = gt_result.get('type', 'unknown') if isinstance(gt_result, dict) else 'unknown'
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _calculate_rank_correlation(self, ranking1: List, ranking2: List) -> float:
        """计算排名相关性（Spearman相关系数）"""
        if not ranking1 or not ranking2:
            return 0.0
        
        # 找到共同元素
        common_items = list(set(ranking1) & set(ranking2))
        if len(common_items) < 2:
            return 0.0
        
        # 获取排名
        rank1 = {item: i for i, item in enumerate(ranking1)}
        rank2 = {item: i for i, item in enumerate(ranking2)}
        
        # 计算Spearman相关系数
        ranks1 = [rank1[item] for item in common_items]
        ranks2 = [rank2[item] for item in common_items]
        
        return np.corrcoef(ranks1, ranks2)[0, 1] if len(set(ranks1)) > 1 and len(set(ranks2)) > 1 else 0.0
    
    def _calculate_confusion_matrix(self, true_labels: List, pred_labels: List, 
                                   labels: List) -> Dict[str, Dict[str, int]]:
        """计算混淆矩阵"""
        matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            matrix[true_label][pred_label] += 1
        
        return matrix
    
    def _calculate_classification_metrics(self, confusion_matrix: Dict, 
                                        labels: List) -> tuple:
        """计算分类指标"""
        precisions = []
        recalls = []
        f1_scores = []
        
        for label in labels:
            # True Positive
            tp = confusion_matrix[label][label]
            
            # False Positive
            fp = sum(confusion_matrix[other_label][label] 
                    for other_label in labels if other_label != label)
            
            # False Negative
            fn = sum(confusion_matrix[label][other_label] 
                    for other_label in labels if other_label != label)
            
            # 计算精确率、召回率、F1分数
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
    
    def _generate_accuracy_summary(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成准确性评估摘要"""
        metrics = evaluation_result.get('metrics', {})
        comparisons = evaluation_result.get('comparisons', [])
        
        summary = {
            "overall_accuracy": metrics.get('accuracy', 0.0),
            "match_rate": metrics.get('match_rate', 0.0),
            "error_rate": metrics.get('error_rate', 0.0),
            "total_comparisons": len(comparisons)
        }
        
        # 按结果类型分组统计
        result_type_stats = {}
        for comp in comparisons:
            result_type = comp.get('result_type', comp.get('query_type', 'unknown'))
            if result_type not in result_type_stats:
                result_type_stats[result_type] = {
                    'count': 0,
                    'accuracy_sum': 0.0,
                    'match_count': 0
                }
            
            result_type_stats[result_type]['count'] += 1
            result_type_stats[result_type]['accuracy_sum'] += comp.get('accuracy', 0.0)
            result_type_stats[result_type]['match_count'] += 1 if comp.get('match', False) else 0
        
        # 计算每种类型的平均准确性
        for result_type, stats in result_type_stats.items():
            stats['average_accuracy'] = stats['accuracy_sum'] / stats['count']
            stats['match_rate'] = stats['match_count'] / stats['count']
        
        summary['result_type_performance'] = result_type_stats
        
        return summary
