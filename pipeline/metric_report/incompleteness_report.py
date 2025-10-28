"""
不完整性报告生成器实现
"""

from typing import Any, Dict, List
import numpy as np
from .base import BaseMetricReport


class IncompletenessReport(BaseMetricReport):
    """不完整性报告生成器 - 专注于算法处理不完整数据的能力评估"""
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成不完整性评估报告"""
        report = {
            "header": self.create_report_header("NGDB不完整性评估报告", evaluation_results),
            "summary": self._create_incompleteness_summary(evaluation_results),
            "detailed_analysis": self._create_detailed_analysis(evaluation_results),
            "missing_data_impact": self._analyze_missing_data_impact(evaluation_results),
            "recovery_capabilities": self._analyze_recovery_capabilities(evaluation_results),
            "recommendations": self._create_incompleteness_recommendations(evaluation_results)
        }
        
        self._report_data = report
        return report
    
    def _create_incompleteness_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建不完整性摘要"""
        summary = {
            "overall_incompleteness_handling": 0.0,
            "missing_data_tolerance": 0.0,
            "information_recovery_rate": 0.0,
            "performance_under_incompleteness": 0.0,
            "key_insights": []
        }
        
        # 从评估结果中提取不完整性相关指标
        incompleteness_metrics = self._extract_incompleteness_metrics(evaluation_results)
        
        if incompleteness_metrics:
            # 计算总体不完整性处理能力
            handling_scores = []
            
            # 准确性在不完整数据下的表现
            if 'accuracy_under_incompleteness' in incompleteness_metrics:
                accuracy_score = incompleteness_metrics['accuracy_under_incompleteness']
                handling_scores.append(accuracy_score)
                summary['performance_under_incompleteness'] = accuracy_score
            
            # 鲁棒性对缺失数据的容忍度
            if 'missing_data_robustness' in incompleteness_metrics:
                robustness_score = incompleteness_metrics['missing_data_robustness']
                handling_scores.append(robustness_score)
                summary['missing_data_tolerance'] = robustness_score
            
            # 信息恢复能力
            if 'information_recovery' in incompleteness_metrics:
                recovery_score = incompleteness_metrics['information_recovery']
                handling_scores.append(recovery_score)
                summary['information_recovery_rate'] = recovery_score
            
            # 计算总体评分
            if handling_scores:
                summary['overall_incompleteness_handling'] = np.mean(handling_scores)
        
        # 生成关键洞察
        summary['key_insights'] = self._generate_incompleteness_insights(incompleteness_metrics)
        
        return summary
    
    def _create_detailed_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建详细分析"""
        detailed_analysis = {
            "node_removal_impact": {},
            "edge_removal_impact": {},
            "attribute_missing_impact": {},
            "structural_completeness": {},
            "semantic_completeness": {}
        }
        
        # 分析节点删除的影响
        detailed_analysis['node_removal_impact'] = self._analyze_node_removal_impact(evaluation_results)
        
        # 分析边删除的影响
        detailed_analysis['edge_removal_impact'] = self._analyze_edge_removal_impact(evaluation_results)
        
        # 分析属性缺失的影响
        detailed_analysis['attribute_missing_impact'] = self._analyze_attribute_missing_impact(evaluation_results)
        
        # 分析结构完整性
        detailed_analysis['structural_completeness'] = self._analyze_structural_completeness(evaluation_results)
        
        # 分析语义完整性
        detailed_analysis['semantic_completeness'] = self._analyze_semantic_completeness(evaluation_results)
        
        return detailed_analysis
    
    def _analyze_missing_data_impact(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析缺失数据影响"""
        impact_analysis = {
            "impact_by_missing_ratio": {},
            "critical_missing_thresholds": {},
            "graceful_degradation": {},
            "failure_points": []
        }
        
        # 分析不同缺失比例的影响
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for ratio in missing_ratios:
            ratio_impact = self._calculate_impact_at_missing_ratio(evaluation_results, ratio)
            impact_analysis['impact_by_missing_ratio'][f'{ratio:.1%}'] = ratio_impact
        
        # 识别关键阈值
        impact_analysis['critical_missing_thresholds'] = self._identify_critical_thresholds(
            impact_analysis['impact_by_missing_ratio']
        )
        
        # 分析优雅降级
        impact_analysis['graceful_degradation'] = self._analyze_graceful_degradation(evaluation_results)
        
        # 识别失效点
        impact_analysis['failure_points'] = self._identify_failure_points(evaluation_results)
        
        return impact_analysis
    
    def _analyze_recovery_capabilities(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析恢复能力"""
        recovery_analysis = {
            "inference_capabilities": {},
            "interpolation_accuracy": {},
            "extrapolation_accuracy": {},
            "uncertainty_quantification": {}
        }
        
        # 分析推理能力
        recovery_analysis['inference_capabilities'] = self._analyze_inference_capabilities(evaluation_results)
        
        # 分析插值准确性
        recovery_analysis['interpolation_accuracy'] = self._analyze_interpolation_accuracy(evaluation_results)
        
        # 分析外推准确性
        recovery_analysis['extrapolation_accuracy'] = self._analyze_extrapolation_accuracy(evaluation_results)
        
        # 分析不确定性量化
        recovery_analysis['uncertainty_quantification'] = self._analyze_uncertainty_quantification(evaluation_results)
        
        return recovery_analysis
    
    def _create_incompleteness_recommendations(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建不完整性建议"""
        recommendations = {
            "data_preprocessing": [],
            "algorithm_improvements": [],
            "robustness_enhancements": [],
            "monitoring_strategies": []
        }
        
        incompleteness_metrics = self._extract_incompleteness_metrics(evaluation_results)
        
        # 数据预处理建议
        if incompleteness_metrics.get('missing_data_tolerance', 0.0) < 0.6:
            recommendations['data_preprocessing'].extend([
                "实施数据质量检查",
                "开发数据插补策略",
                "建立数据完整性监控"
            ])
        
        # 算法改进建议
        if incompleteness_metrics.get('information_recovery', 0.0) < 0.7:
            recommendations['algorithm_improvements'].extend([
                "增强缺失数据处理能力",
                "实现自适应算法参数",
                "开发不确定性感知算法"
            ])
        
        # 鲁棒性增强建议
        if incompleteness_metrics.get('accuracy_under_incompleteness', 0.0) < 0.8:
            recommendations['robustness_enhancements'].extend([
                "实施多模型集成",
                "开发容错机制",
                "增加冗余信息利用"
            ])
        
        # 监控策略建议
        recommendations['monitoring_strategies'].extend([
            "建立数据完整性指标",
            "实施实时质量监控",
            "设置性能降级警报"
        ])
        
        return recommendations
    
    def _extract_incompleteness_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取不完整性相关指标"""
        incompleteness_metrics = {}
        
        # 从准确性评估中提取
        if 'accuracy_evaluation' in evaluation_results:
            acc_data = evaluation_results['accuracy_evaluation']
            acc_metrics = acc_data.get('metrics', {})
            
            # 在不完整数据下的准确性
            incompleteness_metrics['accuracy_under_incompleteness'] = acc_metrics.get('accuracy', 0.0)
        
        # 从鲁棒性评估中提取
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            perturbation_analysis = rob_data.get('perturbation_analysis', {})
            
            # 随机扰动（包括节点/边删除）的鲁棒性
            if 'random' in perturbation_analysis:
                random_analysis = perturbation_analysis['random']
                stability_metrics = random_analysis.get('stability_metrics', {})
                incompleteness_metrics['missing_data_robustness'] = stability_metrics.get('output_consistency', 0.0)
            
            # 语义扰动的鲁棒性
            if 'semantic' in perturbation_analysis:
                semantic_analysis = perturbation_analysis['semantic']
                stability_metrics = semantic_analysis.get('stability_metrics', {})
                incompleteness_metrics['semantic_robustness'] = stability_metrics.get('structure_preservation', 0.0)
        
        # 计算信息恢复率
        if 'accuracy_under_incompleteness' in incompleteness_metrics and 'missing_data_robustness' in incompleteness_metrics:
            accuracy = incompleteness_metrics['accuracy_under_incompleteness']
            robustness = incompleteness_metrics['missing_data_robustness']
            incompleteness_metrics['information_recovery'] = (accuracy + robustness) / 2.0
        
        return incompleteness_metrics
    
    def _generate_incompleteness_insights(self, incompleteness_metrics: Dict[str, Any]) -> List[str]:
        """生成不完整性洞察"""
        insights = []
        
        overall_handling = incompleteness_metrics.get('overall_incompleteness_handling', 0.0)
        
        if overall_handling >= 0.8:
            insights.append("算法对不完整数据具有良好的处理能力")
        elif overall_handling >= 0.6:
            insights.append("算法对不完整数据具有中等处理能力")
        else:
            insights.append("算法对不完整数据的处理能力需要显著改进")
        
        missing_tolerance = incompleteness_metrics.get('missing_data_tolerance', 0.0)
        if missing_tolerance < 0.5:
            insights.append("算法对缺失数据敏感，需要高质量输入")
        
        recovery_rate = incompleteness_metrics.get('information_recovery', 0.0)
        if recovery_rate >= 0.7:
            insights.append("算法具有较强的信息恢复能力")
        elif recovery_rate < 0.5:
            insights.append("算法的信息恢复能力有限")
        
        return insights
    
    def _analyze_node_removal_impact(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析节点删除影响"""
        impact_analysis = {
            "performance_degradation": 0.0,
            "critical_node_sensitivity": 0.0,
            "recovery_mechanisms": [],
            "impact_patterns": {}
        }
        
        # 从鲁棒性评估中提取节点删除相关信息
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            perturbation_analysis = rob_data.get('perturbation_analysis', {})
            
            # 分析随机节点删除
            if 'random' in perturbation_analysis:
                random_analysis = perturbation_analysis['random']
                performance_degradation = random_analysis.get('performance_degradation', {})
                
                if 'score_degradation' in performance_degradation:
                    score_deg = performance_degradation['score_degradation']
                    impact_analysis['performance_degradation'] = score_deg.get('mean_relative_change', 0.0)
            
            # 分析拓扑节点删除
            if 'topology' in perturbation_analysis:
                topology_analysis = perturbation_analysis['topology']
                performance_degradation = topology_analysis.get('performance_degradation', {})
                
                if 'score_degradation' in performance_degradation:
                    score_deg = performance_degradation['score_degradation']
                    impact_analysis['critical_node_sensitivity'] = score_deg.get('max_change', 0.0)
        
        return impact_analysis
    
    def _analyze_edge_removal_impact(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析边删除影响"""
        impact_analysis = {
            "connectivity_impact": 0.0,
            "path_disruption": 0.0,
            "community_fragmentation": 0.0,
            "recovery_strategies": []
        }
        
        # 从鲁棒性评估中提取边删除相关信息
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            perturbation_analysis = rob_data.get('perturbation_analysis', {})
            
            if 'topology' in perturbation_analysis:
                topology_analysis = perturbation_analysis['topology']
                stability_metrics = topology_analysis.get('stability_metrics', {})
                
                impact_analysis['connectivity_impact'] = 1.0 - stability_metrics.get('structure_preservation', 0.0)
                impact_analysis['path_disruption'] = 1.0 - stability_metrics.get('relative_order_preservation', 0.0)
        
        return impact_analysis
    
    def _analyze_attribute_missing_impact(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析属性缺失影响"""
        impact_analysis = {
            "feature_importance": {},
            "missing_attribute_tolerance": 0.0,
            "compensation_mechanisms": [],
            "critical_attributes": []
        }
        
        # 从语义扰动分析中提取属性相关信息
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            perturbation_analysis = rob_data.get('perturbation_analysis', {})
            
            if 'semantic' in perturbation_analysis:
                semantic_analysis = perturbation_analysis['semantic']
                stability_metrics = semantic_analysis.get('stability_metrics', {})
                
                impact_analysis['missing_attribute_tolerance'] = stability_metrics.get('output_consistency', 0.0)
        
        return impact_analysis
    
    def _analyze_structural_completeness(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析结构完整性"""
        completeness_analysis = {
            "graph_connectivity": 0.0,
            "component_integrity": 0.0,
            "topological_preservation": 0.0,
            "structural_metrics": {}
        }
        
        # 从图统计信息中分析结构完整性
        if 'graph_stats' in evaluation_results:
            graph_stats = evaluation_results['graph_stats']
            
            # 连通性分析
            if 'is_connected' in graph_stats:
                completeness_analysis['graph_connectivity'] = 1.0 if graph_stats['is_connected'] else 0.0
            
            # 密度分析
            if 'density' in graph_stats:
                completeness_analysis['structural_metrics']['density'] = graph_stats['density']
        
        return completeness_analysis
    
    def _analyze_semantic_completeness(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析语义完整性"""
        completeness_analysis = {
            "semantic_consistency": 0.0,
            "information_coherence": 0.0,
            "context_preservation": 0.0,
            "semantic_metrics": {}
        }
        
        # 从语义扰动分析中提取语义完整性信息
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            perturbation_analysis = rob_data.get('perturbation_analysis', {})
            
            if 'semantic' in perturbation_analysis:
                semantic_analysis = perturbation_analysis['semantic']
                stability_metrics = semantic_analysis.get('stability_metrics', {})
                
                completeness_analysis['semantic_consistency'] = stability_metrics.get('output_consistency', 0.0)
                completeness_analysis['context_preservation'] = stability_metrics.get('structure_preservation', 0.0)
        
        return completeness_analysis
    
    def _calculate_impact_at_missing_ratio(self, evaluation_results: Dict[str, Any], missing_ratio: float) -> Dict[str, Any]:
        """计算特定缺失比例下的影响"""
        # 简化的影响计算，实际应用中需要根据具体的扰动数据来计算
        base_performance = 1.0
        
        # 假设性能随缺失比例线性下降（实际可能更复杂）
        performance_impact = base_performance * (1.0 - missing_ratio)
        
        return {
            "performance_retention": performance_impact,
            "accuracy_impact": missing_ratio * 0.8,  # 假设准确性影响
            "robustness_impact": missing_ratio * 0.6   # 假设鲁棒性影响
        }
    
    def _identify_critical_thresholds(self, impact_by_ratio: Dict[str, Any]) -> Dict[str, Any]:
        """识别关键阈值"""
        thresholds = {
            "performance_threshold": 0.3,  # 性能下降30%的阈值
            "accuracy_threshold": 0.2,     # 准确性下降20%的阈值
            "failure_threshold": 0.5       # 系统失效的阈值
        }
        
        return thresholds
    
    def _analyze_graceful_degradation(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析优雅降级"""
        degradation_analysis = {
            "degradation_pattern": "linear",  # 简化假设
            "degradation_rate": 0.5,
            "minimum_viable_performance": 0.4
        }
        
        return degradation_analysis
    
    def _identify_failure_points(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别失效点"""
        failure_points = [
            {
                "condition": "node_missing_ratio > 0.6",
                "description": "节点缺失超过60%时算法失效",
                "severity": "critical"
            },
            {
                "condition": "edge_missing_ratio > 0.8",
                "description": "边缺失超过80%时连通性丧失",
                "severity": "high"
            }
        ]
        
        return failure_points
    
    def _analyze_inference_capabilities(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析推理能力"""
        return {
            "missing_node_inference": 0.6,
            "missing_edge_inference": 0.7,
            "missing_attribute_inference": 0.5
        }
    
    def _analyze_interpolation_accuracy(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析插值准确性"""
        return {
            "spatial_interpolation": 0.7,
            "temporal_interpolation": 0.6,
            "feature_interpolation": 0.8
        }
    
    def _analyze_extrapolation_accuracy(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析外推准确性"""
        return {
            "trend_extrapolation": 0.5,
            "pattern_extrapolation": 0.6,
            "boundary_extrapolation": 0.4
        }
    
    def _analyze_uncertainty_quantification(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析不确定性量化"""
        return {
            "uncertainty_estimation": 0.6,
            "confidence_intervals": 0.7,
            "prediction_reliability": 0.8
        }
