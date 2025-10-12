"""
性能评估器实现
"""

from typing import Any, Dict, List
import numpy as np
import time
from .base import BaseEvaluationHarness


class PerformanceEvaluator(BaseEvaluationHarness):
    """性能评估器 - 评估算法的执行性能和效率"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化性能评估器
        
        Args:
            config: 评估配置参数
        """
        super().__init__(config)
        self.performance_metrics = config.get('performance_metrics', [
            'execution_time', 'memory_usage', 'throughput', 'scalability'
        ])
        self.baseline_performance = config.get('baseline_performance', {})
        
    def evaluate(self, ground_truth: Dict[str, Any], 
                algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """执行性能评估"""
        evaluation_result = {
            "evaluation_type": "performance",
            "execution_metrics": {},
            "efficiency_metrics": {},
            "scalability_metrics": {},
            "summary": {}
        }
        
        # 执行性能指标
        execution_metrics = self._evaluate_execution_performance(algorithm_output)
        evaluation_result['execution_metrics'] = execution_metrics
        
        # 效率指标
        efficiency_metrics = self._evaluate_efficiency(ground_truth, algorithm_output)
        evaluation_result['efficiency_metrics'] = efficiency_metrics
        
        # 可扩展性指标
        scalability_metrics = self._evaluate_scalability(algorithm_output)
        evaluation_result['scalability_metrics'] = scalability_metrics
        
        # 生成摘要
        evaluation_result['summary'] = self._generate_performance_summary(evaluation_result)
        
        # 记录评估
        self.record_evaluation(evaluation_result)
        
        return evaluation_result
    
    def _evaluate_execution_performance(self, algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """评估执行性能"""
        execution_metrics = {
            "total_execution_time": 0.0,
            "algorithm_execution_time": 0.0,
            "query_execution_times": {},
            "average_query_time": 0.0,
            "execution_efficiency": 0.0
        }
        
        # 提取执行时间信息
        if 'execution_metadata' in algorithm_output:
            metadata = algorithm_output['execution_metadata']
            execution_metrics['total_execution_time'] = metadata.get('total_time', 0.0)
            execution_metrics['algorithm_execution_time'] = metadata.get('algorithm_time', 0.0)
        
        # 分析查询执行时间
        if 'query_results' in algorithm_output:
            query_times = []
            for query_id, result in algorithm_output['query_results'].items():
                if isinstance(result, dict) and 'execution_time' in result:
                    query_time = result['execution_time']
                    execution_metrics['query_execution_times'][query_id] = query_time
                    query_times.append(query_time)
            
            if query_times:
                execution_metrics['average_query_time'] = np.mean(query_times)
                execution_metrics['query_time_std'] = np.std(query_times)
                execution_metrics['max_query_time'] = np.max(query_times)
                execution_metrics['min_query_time'] = np.min(query_times)
        
        # 计算执行效率
        execution_metrics['execution_efficiency'] = self._calculate_execution_efficiency(
            execution_metrics
        )
        
        return execution_metrics
    
    def _evaluate_efficiency(self, ground_truth: Dict[str, Any], 
                           algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """评估算法效率"""
        efficiency_metrics = {
            "time_efficiency": 0.0,
            "space_efficiency": 0.0,
            "computational_complexity": "unknown",
            "resource_utilization": {}
        }
        
        # 时间效率
        time_efficiency = self._calculate_time_efficiency(algorithm_output)
        efficiency_metrics['time_efficiency'] = time_efficiency
        
        # 空间效率
        space_efficiency = self._calculate_space_efficiency(algorithm_output)
        efficiency_metrics['space_efficiency'] = space_efficiency
        
        # 资源利用率
        resource_utilization = self._analyze_resource_utilization(algorithm_output)
        efficiency_metrics['resource_utilization'] = resource_utilization
        
        # 与基准性能比较
        if self.baseline_performance:
            baseline_comparison = self._compare_with_baseline(algorithm_output)
            efficiency_metrics['baseline_comparison'] = baseline_comparison
        
        return efficiency_metrics
    
    def _evaluate_scalability(self, algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """评估可扩展性"""
        scalability_metrics = {
            "node_scalability": {},
            "edge_scalability": {},
            "query_scalability": {},
            "overall_scalability_score": 0.0
        }
        
        # 分析图大小对性能的影响
        graph_stats = algorithm_output.get('graph_stats', {})
        execution_time = algorithm_output.get('execution_metadata', {}).get('total_time', 0.0)
        
        if graph_stats and execution_time > 0:
            num_nodes = graph_stats.get('num_nodes', 0)
            num_edges = graph_stats.get('num_edges', 0)
            
            # 节点可扩展性（时间复杂度估算）
            if num_nodes > 0:
                node_time_ratio = execution_time / num_nodes
                scalability_metrics['node_scalability'] = {
                    'time_per_node': node_time_ratio,
                    'estimated_complexity': self._estimate_node_complexity(node_time_ratio, num_nodes)
                }
            
            # 边可扩展性
            if num_edges > 0:
                edge_time_ratio = execution_time / num_edges
                scalability_metrics['edge_scalability'] = {
                    'time_per_edge': edge_time_ratio,
                    'estimated_complexity': self._estimate_edge_complexity(edge_time_ratio, num_edges)
                }
        
        # 查询可扩展性
        query_scalability = self._analyze_query_scalability(algorithm_output)
        scalability_metrics['query_scalability'] = query_scalability
        
        # 总体可扩展性评分
        scalability_metrics['overall_scalability_score'] = self._calculate_scalability_score(
            scalability_metrics
        )
        
        return scalability_metrics
    
    def _calculate_execution_efficiency(self, execution_metrics: Dict[str, Any]) -> float:
        """计算执行效率"""
        total_time = execution_metrics.get('total_execution_time', 0.0)
        algorithm_time = execution_metrics.get('algorithm_execution_time', 0.0)
        
        if total_time == 0:
            return 0.0
        
        # 算法执行时间占总时间的比例（越高越好）
        algorithm_ratio = algorithm_time / total_time if total_time > 0 else 0.0
        
        # 查询执行效率
        avg_query_time = execution_metrics.get('average_query_time', 0.0)
        query_efficiency = 1.0 / (1.0 + avg_query_time) if avg_query_time > 0 else 1.0
        
        # 综合效率分数
        return (algorithm_ratio + query_efficiency) / 2.0
    
    def _calculate_time_efficiency(self, algorithm_output: Dict[str, Any]) -> float:
        """计算时间效率"""
        execution_metadata = algorithm_output.get('execution_metadata', {})
        total_time = execution_metadata.get('total_time', 0.0)
        
        if total_time == 0:
            return 1.0
        
        # 基于图大小的时间效率
        graph_stats = algorithm_output.get('graph_stats', {})
        num_nodes = graph_stats.get('num_nodes', 1)
        num_edges = graph_stats.get('num_edges', 1)
        
        # 简单的效率估算：时间应该与图大小成合理比例
        expected_time = (num_nodes + num_edges) * 1e-6  # 假设每个元素1微秒
        efficiency = expected_time / max(total_time, 1e-10)
        
        return min(1.0, efficiency)  # 限制在[0, 1]范围内
    
    def _calculate_space_efficiency(self, algorithm_output: Dict[str, Any]) -> float:
        """计算空间效率"""
        # 简化的空间效率计算
        # 实际实现中可以通过内存监控获取真实内存使用情况
        
        graph_stats = algorithm_output.get('graph_stats', {})
        num_nodes = graph_stats.get('num_nodes', 0)
        num_edges = graph_stats.get('num_edges', 0)
        
        # 估算理论最小内存需求
        min_memory = (num_nodes + num_edges) * 8  # 假设每个元素8字节
        
        # 假设实际使用内存（实际应用中需要真实监控）
        estimated_memory = min_memory * 2  # 假设使用了2倍的理论最小内存
        
        space_efficiency = min_memory / max(estimated_memory, 1)
        return min(1.0, space_efficiency)
    
    def _analyze_resource_utilization(self, algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """分析资源利用率"""
        # 简化的资源利用率分析
        # 实际实现中需要系统监控工具
        
        return {
            "cpu_utilization": 0.8,  # 假设值
            "memory_utilization": 0.6,  # 假设值
            "io_utilization": 0.3,  # 假设值
            "overall_utilization": 0.6
        }
    
    def _compare_with_baseline(self, algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """与基准性能比较"""
        comparison = {
            "time_improvement": 0.0,
            "space_improvement": 0.0,
            "overall_improvement": 0.0
        }
        
        execution_metadata = algorithm_output.get('execution_metadata', {})
        current_time = execution_metadata.get('total_time', 0.0)
        baseline_time = self.baseline_performance.get('execution_time', current_time)
        
        if baseline_time > 0:
            time_improvement = (baseline_time - current_time) / baseline_time
            comparison['time_improvement'] = time_improvement
        
        # 空间改进（简化）
        current_memory = 1000  # 假设值
        baseline_memory = self.baseline_performance.get('memory_usage', current_memory)
        
        if baseline_memory > 0:
            space_improvement = (baseline_memory - current_memory) / baseline_memory
            comparison['space_improvement'] = space_improvement
        
        # 总体改进
        comparison['overall_improvement'] = (
            comparison['time_improvement'] + comparison['space_improvement']
        ) / 2.0
        
        return comparison
    
    def _estimate_node_complexity(self, time_per_node: float, num_nodes: int) -> str:
        """估算节点复杂度"""
        if num_nodes <= 1:
            return "O(1)"
        
        # 基于时间增长率估算复杂度
        if time_per_node < 1e-6:
            return "O(1)"
        elif time_per_node < 1e-5:
            return "O(log n)"
        elif time_per_node < 1e-4:
            return "O(n)"
        elif time_per_node < 1e-3:
            return "O(n log n)"
        else:
            return "O(n²) or higher"
    
    def _estimate_edge_complexity(self, time_per_edge: float, num_edges: int) -> str:
        """估算边复杂度"""
        if num_edges <= 1:
            return "O(1)"
        
        # 基于时间增长率估算复杂度
        if time_per_edge < 1e-6:
            return "O(1)"
        elif time_per_edge < 1e-5:
            return "O(log m)"
        elif time_per_edge < 1e-4:
            return "O(m)"
        else:
            return "O(m log m) or higher"
    
    def _analyze_query_scalability(self, algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """分析查询可扩展性"""
        query_scalability = {
            "query_time_distribution": {},
            "scalability_by_query_type": {},
            "bottleneck_queries": []
        }
        
        query_results = algorithm_output.get('query_results', {})
        if not query_results:
            return query_scalability
        
        # 分析查询时间分布
        query_times = []
        query_types = {}
        
        for query_id, result in query_results.items():
            if isinstance(result, dict):
                query_time = result.get('execution_time', 0.0)
                query_type = result.get('type', 'unknown')
                
                query_times.append(query_time)
                
                if query_type not in query_types:
                    query_types[query_type] = []
                query_types[query_type].append(query_time)
        
        if query_times:
            query_scalability['query_time_distribution'] = {
                'mean': np.mean(query_times),
                'std': np.std(query_times),
                'min': np.min(query_times),
                'max': np.max(query_times),
                'percentiles': {
                    '50': np.percentile(query_times, 50),
                    '90': np.percentile(query_times, 90),
                    '95': np.percentile(query_times, 95)
                }
            }
        
        # 按查询类型分析可扩展性
        for query_type, times in query_types.items():
            query_scalability['scalability_by_query_type'][query_type] = {
                'average_time': np.mean(times),
                'time_variance': np.var(times),
                'scalability_rating': self._rate_query_scalability(np.mean(times))
            }
        
        # 识别瓶颈查询
        if query_times:
            threshold = np.percentile(query_times, 90)
            for query_id, result in query_results.items():
                if isinstance(result, dict):
                    query_time = result.get('execution_time', 0.0)
                    if query_time >= threshold:
                        query_scalability['bottleneck_queries'].append({
                            'query_id': query_id,
                            'execution_time': query_time,
                            'query_type': result.get('type', 'unknown')
                        })
        
        return query_scalability
    
    def _rate_query_scalability(self, average_time: float) -> str:
        """评估查询可扩展性等级"""
        if average_time < 0.001:  # 1ms
            return "excellent"
        elif average_time < 0.01:  # 10ms
            return "good"
        elif average_time < 0.1:  # 100ms
            return "fair"
        else:
            return "poor"
    
    def _calculate_scalability_score(self, scalability_metrics: Dict[str, Any]) -> float:
        """计算总体可扩展性评分"""
        scores = []
        
        # 节点可扩展性评分
        node_scalability = scalability_metrics.get('node_scalability', {})
        if 'time_per_node' in node_scalability:
            time_per_node = node_scalability['time_per_node']
            node_score = 1.0 / (1.0 + time_per_node * 1000000)  # 转换为合理范围
            scores.append(node_score)
        
        # 边可扩展性评分
        edge_scalability = scalability_metrics.get('edge_scalability', {})
        if 'time_per_edge' in edge_scalability:
            time_per_edge = edge_scalability['time_per_edge']
            edge_score = 1.0 / (1.0 + time_per_edge * 1000000)
            scores.append(edge_score)
        
        # 查询可扩展性评分
        query_scalability = scalability_metrics.get('query_scalability', {})
        query_dist = query_scalability.get('query_time_distribution', {})
        if 'mean' in query_dist:
            mean_query_time = query_dist['mean']
            query_score = 1.0 / (1.0 + mean_query_time * 100)
            scores.append(query_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_performance_summary(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能评估摘要"""
        execution_metrics = evaluation_result.get('execution_metrics', {})
        efficiency_metrics = evaluation_result.get('efficiency_metrics', {})
        scalability_metrics = evaluation_result.get('scalability_metrics', {})
        
        summary = {
            "overall_performance_score": 0.0,
            "execution_summary": {},
            "efficiency_summary": {},
            "scalability_summary": {},
            "performance_bottlenecks": [],
            "recommendations": []
        }
        
        # 执行摘要
        summary['execution_summary'] = {
            "total_execution_time": execution_metrics.get('total_execution_time', 0.0),
            "average_query_time": execution_metrics.get('average_query_time', 0.0),
            "execution_efficiency": execution_metrics.get('execution_efficiency', 0.0)
        }
        
        # 效率摘要
        summary['efficiency_summary'] = {
            "time_efficiency": efficiency_metrics.get('time_efficiency', 0.0),
            "space_efficiency": efficiency_metrics.get('space_efficiency', 0.0),
            "resource_utilization": efficiency_metrics.get('resource_utilization', {}).get('overall_utilization', 0.0)
        }
        
        # 可扩展性摘要
        summary['scalability_summary'] = {
            "overall_scalability_score": scalability_metrics.get('overall_scalability_score', 0.0),
            "node_complexity": scalability_metrics.get('node_scalability', {}).get('estimated_complexity', 'unknown'),
            "edge_complexity": scalability_metrics.get('edge_scalability', {}).get('estimated_complexity', 'unknown')
        }
        
        # 计算总体性能分数
        performance_scores = [
            execution_metrics.get('execution_efficiency', 0.0),
            efficiency_metrics.get('time_efficiency', 0.0),
            efficiency_metrics.get('space_efficiency', 0.0),
            scalability_metrics.get('overall_scalability_score', 0.0)
        ]
        summary['overall_performance_score'] = np.mean([s for s in performance_scores if s > 0])
        
        # 识别性能瓶颈
        bottlenecks = []
        if execution_metrics.get('execution_efficiency', 0.0) < 0.5:
            bottlenecks.append("执行效率低")
        if efficiency_metrics.get('time_efficiency', 0.0) < 0.5:
            bottlenecks.append("时间效率低")
        if scalability_metrics.get('overall_scalability_score', 0.0) < 0.5:
            bottlenecks.append("可扩展性差")
        
        summary['performance_bottlenecks'] = bottlenecks
        
        # 生成建议
        recommendations = []
        if execution_metrics.get('average_query_time', 0.0) > 0.1:
            recommendations.append("优化查询执行时间")
        if efficiency_metrics.get('space_efficiency', 0.0) < 0.6:
            recommendations.append("优化内存使用")
        if scalability_metrics.get('overall_scalability_score', 0.0) < 0.6:
            recommendations.append("改进算法可扩展性")
        
        summary['recommendations'] = recommendations
        
        return summary
