"""
鲁棒性评估器实现
"""

from typing import Any, Dict, List
import numpy as np
from .base import BaseEvaluationHarness


class RobustnessEvaluator(BaseEvaluationHarness):
    """鲁棒性评估器 - 评估算法在扰动数据下的鲁棒性"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化鲁棒性评估器
        
        Args:
            config: 评估配置参数
        """
        super().__init__(config)
        self.perturbation_types = config.get('perturbation_types', ['random', 'semantic', 'topology'])
        self.robustness_threshold = config.get('robustness_threshold', 0.8)
        
    def evaluate(self, ground_truth: Dict[str, Any], 
                algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """执行鲁棒性评估"""
        evaluation_result = {
            "evaluation_type": "robustness",
            "perturbation_analysis": {},
            "robustness_metrics": {},
            "summary": {}
        }
        
        # 分析不同扰动类型的影响
        for perturbation_type in self.perturbation_types:
            perturbation_analysis = self._analyze_perturbation_impact(
                ground_truth, algorithm_output, perturbation_type
            )
            evaluation_result['perturbation_analysis'][perturbation_type] = perturbation_analysis
        
        # 计算鲁棒性指标
        evaluation_result['robustness_metrics'] = self._calculate_robustness_metrics(
            evaluation_result['perturbation_analysis']
        )
        
        # 生成摘要
        evaluation_result['summary'] = self._generate_robustness_summary(evaluation_result)
        
        # 记录评估
        self.record_evaluation(evaluation_result)
        
        return evaluation_result
    
    def _analyze_perturbation_impact(self, ground_truth: Dict[str, Any], 
                                   algorithm_output: Dict[str, Any], 
                                   perturbation_type: str) -> Dict[str, Any]:
        """分析特定扰动类型的影响"""
        analysis = {
            "perturbation_type": perturbation_type,
            "performance_degradation": {},
            "stability_metrics": {},
            "error_patterns": {}
        }
        
        # 比较扰动前后的性能
        if 'algorithm_results' in ground_truth and 'algorithm_results' in algorithm_output:
            performance_analysis = self._analyze_performance_degradation(
                ground_truth['algorithm_results'],
                algorithm_output['algorithm_results'],
                perturbation_type
            )
            analysis['performance_degradation'] = performance_analysis
        
        # 分析稳定性
        stability_analysis = self._analyze_stability(
            ground_truth, algorithm_output, perturbation_type
        )
        analysis['stability_metrics'] = stability_analysis
        
        # 分析错误模式
        error_analysis = self._analyze_error_patterns(
            ground_truth, algorithm_output, perturbation_type
        )
        analysis['error_patterns'] = error_analysis
        
        return analysis
    
    def _analyze_performance_degradation(self, gt_results: Dict[str, Any], 
                                       algo_results: Dict[str, Any], 
                                       perturbation_type: str) -> Dict[str, Any]:
        """分析性能退化"""
        degradation_analysis = {
            "score_degradation": {},
            "ranking_degradation": {},
            "classification_degradation": {}
        }
        
        # 分析分数退化
        if 'scores' in gt_results and 'scores' in algo_results:
            score_degradation = self._calculate_score_degradation(
                gt_results['scores'], algo_results['scores']
            )
            degradation_analysis['score_degradation'] = score_degradation
        
        # 分析排名退化
        if 'top_nodes' in gt_results and 'top_nodes' in algo_results:
            ranking_degradation = self._calculate_ranking_degradation(
                gt_results['top_nodes'], algo_results['top_nodes']
            )
            degradation_analysis['ranking_degradation'] = ranking_degradation
        
        # 分析分类退化
        if 'classifications' in gt_results and 'classifications' in algo_results:
            classification_degradation = self._calculate_classification_degradation(
                gt_results['classifications'], algo_results['classifications']
            )
            degradation_analysis['classification_degradation'] = classification_degradation
        
        return degradation_analysis
    
    def _calculate_score_degradation(self, gt_scores: Dict, algo_scores: Dict) -> Dict[str, Any]:
        """计算分数退化"""
        common_keys = set(gt_scores.keys()) & set(algo_scores.keys())
        
        if not common_keys:
            return {"error": "No common keys for score comparison"}
        
        # 计算分数变化
        score_changes = []
        relative_changes = []
        
        for key in common_keys:
            gt_score = float(gt_scores[key])
            algo_score = float(algo_scores[key])
            
            absolute_change = abs(gt_score - algo_score)
            relative_change = absolute_change / max(abs(gt_score), 1e-10)
            
            score_changes.append(absolute_change)
            relative_changes.append(relative_change)
        
        return {
            "mean_absolute_change": np.mean(score_changes),
            "std_absolute_change": np.std(score_changes),
            "mean_relative_change": np.mean(relative_changes),
            "std_relative_change": np.std(relative_changes),
            "max_change": np.max(score_changes),
            "num_compared": len(common_keys)
        }
    
    def _calculate_ranking_degradation(self, gt_ranking: List, algo_ranking: List) -> Dict[str, Any]:
        """计算排名退化"""
        if isinstance(gt_ranking[0], tuple):
            gt_items = [item for item, _ in gt_ranking]
            algo_items = [item for item, _ in algo_ranking]
        else:
            gt_items = gt_ranking
            algo_items = algo_ranking
        
        # 计算排名变化
        rank_changes = []
        common_items = set(gt_items) & set(algo_items)
        
        for item in common_items:
            gt_rank = gt_items.index(item) if item in gt_items else len(gt_items)
            algo_rank = algo_items.index(item) if item in algo_items else len(algo_items)
            rank_changes.append(abs(gt_rank - algo_rank))
        
        # 计算Top-K稳定性
        top_k_stability = {}
        for k in [1, 5, 10]:
            if k <= len(gt_items) and k <= len(algo_items):
                gt_top_k = set(gt_items[:k])
                algo_top_k = set(algo_items[:k])
                stability = len(gt_top_k & algo_top_k) / k
                top_k_stability[f'top_{k}_stability'] = stability
        
        return {
            "mean_rank_change": np.mean(rank_changes) if rank_changes else 0.0,
            "std_rank_change": np.std(rank_changes) if rank_changes else 0.0,
            "max_rank_change": np.max(rank_changes) if rank_changes else 0.0,
            **top_k_stability
        }
    
    def _calculate_classification_degradation(self, gt_classes: Dict, algo_classes: Dict) -> Dict[str, Any]:
        """计算分类退化"""
        common_keys = set(gt_classes.keys()) & set(algo_classes.keys())
        
        if not common_keys:
            return {"error": "No common keys for classification comparison"}
        
        # 计算分类变化
        class_changes = sum(1 for key in common_keys if gt_classes[key] != algo_classes[key])
        stability = 1.0 - (class_changes / len(common_keys))
        
        # 分析类别分布变化
        gt_class_dist = {}
        algo_class_dist = {}
        
        for key in common_keys:
            gt_class = gt_classes[key]
            algo_class = algo_classes[key]
            
            gt_class_dist[gt_class] = gt_class_dist.get(gt_class, 0) + 1
            algo_class_dist[algo_class] = algo_class_dist.get(algo_class, 0) + 1
        
        # 计算分布差异
        all_classes = set(gt_class_dist.keys()) | set(algo_class_dist.keys())
        distribution_changes = []
        
        for cls in all_classes:
            gt_count = gt_class_dist.get(cls, 0)
            algo_count = algo_class_dist.get(cls, 0)
            distribution_changes.append(abs(gt_count - algo_count))
        
        return {
            "classification_stability": stability,
            "num_class_changes": class_changes,
            "distribution_change": np.sum(distribution_changes),
            "num_compared": len(common_keys)
        }
    
    def _analyze_stability(self, ground_truth: Dict[str, Any], 
                          algorithm_output: Dict[str, Any], 
                          perturbation_type: str) -> Dict[str, Any]:
        """分析算法稳定性"""
        stability_metrics = {
            "output_consistency": 0.0,
            "structure_preservation": 0.0,
            "relative_order_preservation": 0.0
        }
        
        # 输出一致性
        if 'algorithm_results' in ground_truth and 'algorithm_results' in algorithm_output:
            consistency = self._calculate_output_consistency(
                ground_truth['algorithm_results'],
                algorithm_output['algorithm_results']
            )
            stability_metrics['output_consistency'] = consistency
        
        # 结构保持性
        structure_preservation = self._calculate_structure_preservation(
            ground_truth, algorithm_output
        )
        stability_metrics['structure_preservation'] = structure_preservation
        
        # 相对顺序保持性
        if 'algorithm_results' in ground_truth and 'algorithm_results' in algorithm_output:
            order_preservation = self._calculate_relative_order_preservation(
                ground_truth['algorithm_results'],
                algorithm_output['algorithm_results']
            )
            stability_metrics['relative_order_preservation'] = order_preservation
        
        return stability_metrics
    
    def _calculate_output_consistency(self, gt_results: Dict[str, Any], 
                                    algo_results: Dict[str, Any]) -> float:
        """计算输出一致性"""
        if 'scores' in gt_results and 'scores' in algo_results:
            gt_scores = gt_results['scores']
            algo_scores = algo_results['scores']
            
            common_keys = set(gt_scores.keys()) & set(algo_scores.keys())
            if not common_keys:
                return 0.0
            
            # 计算相关系数
            gt_values = [gt_scores[key] for key in common_keys]
            algo_values = [algo_scores[key] for key in common_keys]
            
            if len(set(gt_values)) > 1 and len(set(algo_values)) > 1:
                correlation = np.corrcoef(gt_values, algo_values)[0, 1]
                return max(0.0, correlation)  # 确保非负
            else:
                return 0.0
        
        return 0.0
    
    def _calculate_structure_preservation(self, ground_truth: Dict[str, Any], 
                                        algorithm_output: Dict[str, Any]) -> float:
        """计算结构保持性"""
        # 比较图统计信息的保持程度
        gt_stats = ground_truth.get('graph_stats', {})
        algo_stats = algorithm_output.get('graph_stats', {})
        
        if not gt_stats or not algo_stats:
            return 0.0
        
        # 比较关键统计指标
        key_metrics = ['num_nodes', 'num_edges', 'density']
        preservation_scores = []
        
        for metric in key_metrics:
            if metric in gt_stats and metric in algo_stats:
                gt_val = gt_stats[metric]
                algo_val = algo_stats[metric]
                
                if gt_val == 0:
                    score = 1.0 if algo_val == 0 else 0.0
                else:
                    score = 1.0 - abs(gt_val - algo_val) / gt_val
                
                preservation_scores.append(max(0.0, score))
        
        return np.mean(preservation_scores) if preservation_scores else 0.0
    
    def _calculate_relative_order_preservation(self, gt_results: Dict[str, Any], 
                                             algo_results: Dict[str, Any]) -> float:
        """计算相对顺序保持性"""
        if 'scores' in gt_results and 'scores' in algo_results:
            gt_scores = gt_results['scores']
            algo_scores = algo_results['scores']
            
            common_keys = list(set(gt_scores.keys()) & set(algo_scores.keys()))
            if len(common_keys) < 2:
                return 0.0
            
            # 计算所有节点对的顺序保持性
            preserved_orders = 0
            total_pairs = 0
            
            for i in range(len(common_keys)):
                for j in range(i + 1, len(common_keys)):
                    key1, key2 = common_keys[i], common_keys[j]
                    
                    gt_order = gt_scores[key1] > gt_scores[key2]
                    algo_order = algo_scores[key1] > algo_scores[key2]
                    
                    if gt_order == algo_order:
                        preserved_orders += 1
                    total_pairs += 1
            
            return preserved_orders / total_pairs if total_pairs > 0 else 0.0
        
        return 0.0
    
    def _analyze_error_patterns(self, ground_truth: Dict[str, Any], 
                               algorithm_output: Dict[str, Any], 
                               perturbation_type: str) -> Dict[str, Any]:
        """分析错误模式"""
        error_patterns = {
            "systematic_errors": [],
            "random_errors": [],
            "error_distribution": {},
            "failure_modes": []
        }
        
        # 分析查询结果中的错误模式
        if 'query_results' in ground_truth and 'query_results' in algorithm_output:
            query_errors = self._analyze_query_errors(
                ground_truth['query_results'],
                algorithm_output['query_results']
            )
            error_patterns.update(query_errors)
        
        return error_patterns
    
    def _analyze_query_errors(self, gt_queries: Dict[str, Any], 
                             algo_queries: Dict[str, Any]) -> Dict[str, Any]:
        """分析查询错误"""
        error_analysis = {
            "query_error_rates": {},
            "error_types": {},
            "systematic_failures": []
        }
        
        common_queries = set(gt_queries.keys()) & set(algo_queries.keys())
        
        for query_id in common_queries:
            gt_result = gt_queries[query_id]
            algo_result = algo_queries[query_id]
            
            # 检查是否有错误
            if isinstance(gt_result, dict) and isinstance(algo_result, dict):
                if 'error' in algo_result:
                    error_type = algo_result.get('error', 'unknown_error')
                    query_type = gt_result.get('type', 'unknown')
                    
                    if query_type not in error_analysis['query_error_rates']:
                        error_analysis['query_error_rates'][query_type] = 0
                    error_analysis['query_error_rates'][query_type] += 1
                    
                    if error_type not in error_analysis['error_types']:
                        error_analysis['error_types'][error_type] = 0
                    error_analysis['error_types'][error_type] += 1
        
        return error_analysis
    
    def _calculate_robustness_metrics(self, perturbation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算鲁棒性指标"""
        robustness_metrics = {
            "overall_robustness": 0.0,
            "perturbation_robustness": {},
            "stability_score": 0.0,
            "degradation_score": 0.0
        }
        
        stability_scores = []
        degradation_scores = []
        
        for perturbation_type, analysis in perturbation_analysis.items():
            # 计算该扰动类型的鲁棒性
            stability_metrics = analysis.get('stability_metrics', {})
            performance_degradation = analysis.get('performance_degradation', {})
            
            # 稳定性分数
            stability_score = np.mean([
                stability_metrics.get('output_consistency', 0.0),
                stability_metrics.get('structure_preservation', 0.0),
                stability_metrics.get('relative_order_preservation', 0.0)
            ])
            
            # 性能退化分数（越低越好，转换为鲁棒性分数）
            degradation_values = []
            for deg_type, deg_data in performance_degradation.items():
                if isinstance(deg_data, dict):
                    if 'mean_relative_change' in deg_data:
                        degradation_values.append(deg_data['mean_relative_change'])
                    elif 'classification_stability' in deg_data:
                        degradation_values.append(1.0 - deg_data['classification_stability'])
            
            degradation_score = 1.0 - np.mean(degradation_values) if degradation_values else 1.0
            
            # 综合鲁棒性分数
            perturbation_robustness = (stability_score + degradation_score) / 2.0
            
            robustness_metrics['perturbation_robustness'][perturbation_type] = {
                'robustness_score': perturbation_robustness,
                'stability_score': stability_score,
                'degradation_score': degradation_score
            }
            
            stability_scores.append(stability_score)
            degradation_scores.append(degradation_score)
        
        # 总体鲁棒性
        if stability_scores and degradation_scores:
            robustness_metrics['overall_robustness'] = (
                np.mean(stability_scores) + np.mean(degradation_scores)
            ) / 2.0
            robustness_metrics['stability_score'] = np.mean(stability_scores)
            robustness_metrics['degradation_score'] = np.mean(degradation_scores)
        
        return robustness_metrics
    
    def _generate_robustness_summary(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成鲁棒性评估摘要"""
        robustness_metrics = evaluation_result.get('robustness_metrics', {})
        perturbation_analysis = evaluation_result.get('perturbation_analysis', {})
        
        summary = {
            "overall_robustness": robustness_metrics.get('overall_robustness', 0.0),
            "is_robust": robustness_metrics.get('overall_robustness', 0.0) >= self.robustness_threshold,
            "robustness_threshold": self.robustness_threshold,
            "perturbation_impact": {}
        }
        
        # 按扰动类型总结影响
        for perturbation_type, analysis in perturbation_analysis.items():
            perturbation_robustness = robustness_metrics.get('perturbation_robustness', {}).get(perturbation_type, {})
            
            summary['perturbation_impact'][perturbation_type] = {
                "robustness_score": perturbation_robustness.get('robustness_score', 0.0),
                "is_robust_to_perturbation": perturbation_robustness.get('robustness_score', 0.0) >= self.robustness_threshold,
                "main_impact": self._summarize_perturbation_impact(analysis)
            }
        
        return summary
    
    def _summarize_perturbation_impact(self, analysis: Dict[str, Any]) -> str:
        """总结扰动影响"""
        performance_degradation = analysis.get('performance_degradation', {})
        stability_metrics = analysis.get('stability_metrics', {})
        
        impacts = []
        
        # 检查性能退化
        for deg_type, deg_data in performance_degradation.items():
            if isinstance(deg_data, dict):
                if 'mean_relative_change' in deg_data and deg_data['mean_relative_change'] > 0.1:
                    impacts.append(f"显著的{deg_type}退化")
                elif 'classification_stability' in deg_data and deg_data['classification_stability'] < 0.8:
                    impacts.append(f"分类稳定性下降")
        
        # 检查稳定性问题
        if stability_metrics.get('output_consistency', 0.0) < 0.7:
            impacts.append("输出一致性差")
        if stability_metrics.get('structure_preservation', 0.0) < 0.8:
            impacts.append("结构保持性差")
        
        return "; ".join(impacts) if impacts else "影响较小"
