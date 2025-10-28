"""
综合报告生成器实现
"""

from typing import Any, Dict, List
import numpy as np
from .base import BaseMetricReport


class ComprehensiveReport(BaseMetricReport):
    """综合报告生成器 - 生成包含所有评估结果的综合报告"""
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合评估报告"""
        report = {
            "header": self.create_report_header("NGDB综合评估报告", evaluation_results),
            "executive_summary": self._create_executive_summary(evaluation_results),
            "detailed_results": self._create_detailed_results(evaluation_results),
            "comparative_analysis": self._create_comparative_analysis(evaluation_results),
            "recommendations": self._create_recommendations(evaluation_results),
            "appendix": self._create_appendix(evaluation_results)
        }
        
        # 验证报告数据
        validation_result = self.validate_report_data(report)
        if not validation_result["is_valid"]:
            report["validation_errors"] = validation_result["errors"]
            report["validation_warnings"] = validation_result["warnings"]
        
        self._report_data = report
        return report
    
    def _create_executive_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行摘要"""
        summary = {
            "overall_performance": {},
            "key_findings": [],
            "performance_highlights": {},
            "main_concerns": []
        }
        
        # 收集所有评估结果的关键指标
        all_metrics = []
        evaluation_types = []
        
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict) and 'metrics' in eval_data:
                all_metrics.append(eval_data['metrics'])
                evaluation_types.append(eval_type)
        
        # 计算总体性能分数
        if all_metrics:
            overall_scores = []
            
            for metrics in all_metrics:
                # 提取关键性能指标
                key_metrics = ['accuracy', 'match_rate', 'overall_robustness', 'overall_performance_score']
                scores = [metrics.get(metric, 0.0) for metric in key_metrics if metric in metrics]
                if scores:
                    overall_scores.append(np.mean(scores))
            
            if overall_scores:
                summary['overall_performance'] = {
                    "score": np.mean(overall_scores),
                    "grade": self._calculate_performance_grade(np.mean(overall_scores)),
                    "evaluation_count": len(overall_scores)
                }
        
        # 关键发现
        summary['key_findings'] = self._extract_key_findings(evaluation_results)
        
        # 性能亮点
        summary['performance_highlights'] = self._identify_performance_highlights(evaluation_results)
        
        # 主要关注点
        summary['main_concerns'] = self._identify_main_concerns(evaluation_results)
        
        return summary
    
    def _create_detailed_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建详细结果"""
        detailed_results = {}
        
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict):
                detailed_results[eval_type] = {
                    "summary": eval_data.get('summary', {}),
                    "metrics": self.format_metrics(eval_data.get('metrics', {})),
                    "analysis": self._analyze_evaluation_results(eval_data, eval_type)
                }
                
                # 添加特定类型的详细信息
                if eval_type == 'accuracy_evaluation':
                    detailed_results[eval_type]['accuracy_breakdown'] = self._create_accuracy_breakdown(eval_data)
                elif eval_type == 'robustness_evaluation':
                    detailed_results[eval_type]['robustness_breakdown'] = self._create_robustness_breakdown(eval_data)
                elif eval_type == 'performance_evaluation':
                    detailed_results[eval_type]['performance_breakdown'] = self._create_performance_breakdown(eval_data)
        
        return detailed_results
    
    def _create_comparative_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建比较分析"""
        comparative_analysis = {
            "cross_evaluation_comparison": {},
            "perturbation_impact_analysis": {},
            "performance_trade_offs": {}
        }
        
        # 跨评估比较
        evaluation_metrics = {}
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict) and 'metrics' in eval_data:
                evaluation_metrics[eval_type] = eval_data['metrics']
        
        if len(evaluation_metrics) > 1:
            comparative_analysis['cross_evaluation_comparison'] = self._compare_evaluations(evaluation_metrics)
        
        # 扰动影响分析
        if 'robustness_evaluation' in evaluation_results:
            robustness_data = evaluation_results['robustness_evaluation']
            if 'perturbation_analysis' in robustness_data:
                comparative_analysis['perturbation_impact_analysis'] = self._analyze_perturbation_impacts(
                    robustness_data['perturbation_analysis']
                )
        
        # 性能权衡分析
        comparative_analysis['performance_trade_offs'] = self._analyze_performance_trade_offs(evaluation_results)
        
        return comparative_analysis
    
    def _create_recommendations(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建建议"""
        recommendations = {
            "immediate_actions": [],
            "optimization_suggestions": [],
            "long_term_improvements": [],
            "risk_mitigation": []
        }
        
        # 基于评估结果生成建议
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict):
                eval_recommendations = self._generate_evaluation_recommendations(eval_data, eval_type)
                
                recommendations['immediate_actions'].extend(eval_recommendations.get('immediate', []))
                recommendations['optimization_suggestions'].extend(eval_recommendations.get('optimization', []))
                recommendations['long_term_improvements'].extend(eval_recommendations.get('long_term', []))
                recommendations['risk_mitigation'].extend(eval_recommendations.get('risk', []))
        
        # 去重并排序
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))
        
        return recommendations
    
    def _create_appendix(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建附录"""
        appendix = {
            "technical_details": {},
            "raw_metrics": {},
            "configuration_info": {},
            "data_statistics": {}
        }
        
        # 技术细节
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict):
                appendix['technical_details'][eval_type] = {
                    "evaluation_type": eval_data.get('evaluation_type', eval_type),
                    "num_comparisons": eval_data.get('num_comparisons', 0),
                    "execution_time": eval_data.get('execution_time', 0.0)
                }
        
        # 原始指标
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict) and 'metrics' in eval_data:
                appendix['raw_metrics'][eval_type] = eval_data['metrics']
        
        # 配置信息
        appendix['configuration_info'] = self.config
        
        return appendix
    
    def _calculate_performance_grade(self, score: float) -> str:
        """计算性能等级"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _extract_key_findings(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 准确性发现
        if 'accuracy_evaluation' in evaluation_results:
            acc_data = evaluation_results['accuracy_evaluation']
            acc_metrics = acc_data.get('metrics', {})
            
            overall_accuracy = acc_metrics.get('accuracy', 0.0)
            if overall_accuracy >= 0.9:
                findings.append(f"算法准确性优秀 ({overall_accuracy:.2%})")
            elif overall_accuracy < 0.7:
                findings.append(f"算法准确性需要改进 ({overall_accuracy:.2%})")
        
        # 鲁棒性发现
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            rob_metrics = rob_data.get('robustness_metrics', {})
            
            overall_robustness = rob_metrics.get('overall_robustness', 0.0)
            if overall_robustness >= 0.8:
                findings.append(f"算法鲁棒性良好 ({overall_robustness:.2%})")
            elif overall_robustness < 0.6:
                findings.append(f"算法鲁棒性较差 ({overall_robustness:.2%})")
        
        # 性能发现
        if 'performance_evaluation' in evaluation_results:
            perf_data = evaluation_results['performance_evaluation']
            perf_metrics = perf_data.get('execution_metrics', {})
            
            avg_query_time = perf_metrics.get('average_query_time', 0.0)
            if avg_query_time > 0.1:
                findings.append(f"查询执行时间较长 ({avg_query_time:.3f}秒)")
        
        return findings
    
    def _identify_performance_highlights(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """识别性能亮点"""
        highlights = {
            "best_performing_areas": [],
            "notable_achievements": [],
            "competitive_advantages": []
        }
        
        # 分析各个评估维度的表现
        performance_areas = {}
        
        for eval_type, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict) and 'metrics' in eval_data:
                metrics = eval_data['metrics']
                
                # 提取关键指标
                key_metrics = ['accuracy', 'match_rate', 'overall_robustness', 'execution_efficiency']
                area_scores = []
                
                for metric in key_metrics:
                    if metric in metrics:
                        area_scores.append(metrics[metric])
                
                if area_scores:
                    performance_areas[eval_type] = np.mean(area_scores)
        
        # 识别最佳表现领域
        if performance_areas:
            best_area = max(performance_areas, key=performance_areas.get)
            best_score = performance_areas[best_area]
            
            if best_score >= 0.8:
                highlights['best_performing_areas'].append({
                    "area": best_area,
                    "score": best_score,
                    "description": f"{best_area}表现优异"
                })
        
        return highlights
    
    def _identify_main_concerns(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别主要关注点"""
        concerns = []
        
        # 准确性关注点
        if 'accuracy_evaluation' in evaluation_results:
            acc_data = evaluation_results['accuracy_evaluation']
            acc_metrics = acc_data.get('metrics', {})
            
            if acc_metrics.get('error_rate', 0.0) > 0.3:
                concerns.append({
                    "category": "accuracy",
                    "severity": "high",
                    "description": "错误率过高",
                    "metric_value": acc_metrics.get('error_rate', 0.0)
                })
        
        # 鲁棒性关注点
        if 'robustness_evaluation' in evaluation_results:
            rob_data = evaluation_results['robustness_evaluation']
            rob_metrics = rob_data.get('robustness_metrics', {})
            
            if rob_metrics.get('overall_robustness', 0.0) < 0.5:
                concerns.append({
                    "category": "robustness",
                    "severity": "high",
                    "description": "算法鲁棒性不足",
                    "metric_value": rob_metrics.get('overall_robustness', 0.0)
                })
        
        # 性能关注点
        if 'performance_evaluation' in evaluation_results:
            perf_data = evaluation_results['performance_evaluation']
            scalability_metrics = perf_data.get('scalability_metrics', {})
            
            if scalability_metrics.get('overall_scalability_score', 0.0) < 0.5:
                concerns.append({
                    "category": "scalability",
                    "severity": "medium",
                    "description": "可扩展性有限",
                    "metric_value": scalability_metrics.get('overall_scalability_score', 0.0)
                })
        
        return concerns
    
    def _analyze_evaluation_results(self, eval_data: Dict[str, Any], eval_type: str) -> Dict[str, Any]:
        """分析评估结果"""
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "trends": [],
            "insights": []
        }
        
        metrics = eval_data.get('metrics', {})
        
        # 基于评估类型进行特定分析
        if eval_type == 'accuracy_evaluation':
            analysis.update(self._analyze_accuracy_results(metrics))
        elif eval_type == 'robustness_evaluation':
            analysis.update(self._analyze_robustness_results(metrics))
        elif eval_type == 'performance_evaluation':
            analysis.update(self._analyze_performance_results(metrics))
        
        return analysis
    
    def _analyze_accuracy_results(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析准确性结果"""
        analysis = {"strengths": [], "weaknesses": [], "insights": []}
        
        accuracy = metrics.get('accuracy', 0.0)
        match_rate = metrics.get('match_rate', 0.0)
        
        if accuracy >= 0.9:
            analysis['strengths'].append("高准确性表现")
        elif accuracy < 0.7:
            analysis['weaknesses'].append("准确性需要提升")
        
        if match_rate >= 0.8:
            analysis['strengths'].append("良好的结果匹配率")
        elif match_rate < 0.6:
            analysis['weaknesses'].append("结果匹配率较低")
        
        analysis['insights'].append(f"准确性与匹配率相关性: {abs(accuracy - match_rate):.3f}")
        
        return analysis
    
    def _analyze_robustness_results(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析鲁棒性结果"""
        analysis = {"strengths": [], "weaknesses": [], "insights": []}
        
        overall_robustness = metrics.get('overall_robustness', 0.0)
        stability_score = metrics.get('stability_score', 0.0)
        
        if overall_robustness >= 0.8:
            analysis['strengths'].append("优秀的整体鲁棒性")
        elif overall_robustness < 0.6:
            analysis['weaknesses'].append("鲁棒性不足")
        
        if stability_score >= 0.8:
            analysis['strengths'].append("良好的稳定性")
        elif stability_score < 0.6:
            analysis['weaknesses'].append("稳定性有待改进")
        
        return analysis
    
    def _analyze_performance_results(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能结果"""
        analysis = {"strengths": [], "weaknesses": [], "insights": []}
        
        execution_efficiency = metrics.get('execution_efficiency', 0.0)
        time_efficiency = metrics.get('time_efficiency', 0.0)
        
        if execution_efficiency >= 0.8:
            analysis['strengths'].append("高执行效率")
        elif execution_efficiency < 0.6:
            analysis['weaknesses'].append("执行效率需要优化")
        
        if time_efficiency >= 0.8:
            analysis['strengths'].append("良好的时间效率")
        elif time_efficiency < 0.6:
            analysis['weaknesses'].append("时间效率有待提升")
        
        return analysis
    
    def _create_accuracy_breakdown(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建准确性细分"""
        return {
            "by_result_type": eval_data.get('summary', {}).get('result_type_performance', {}),
            "comparison_details": eval_data.get('comparisons', []),
            "error_analysis": self._analyze_accuracy_errors(eval_data)
        }
    
    def _create_robustness_breakdown(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建鲁棒性细分"""
        return {
            "by_perturbation_type": eval_data.get('perturbation_analysis', {}),
            "stability_analysis": eval_data.get('robustness_metrics', {}),
            "degradation_patterns": self._analyze_degradation_patterns(eval_data)
        }
    
    def _create_performance_breakdown(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建性能细分"""
        return {
            "execution_breakdown": eval_data.get('execution_metrics', {}),
            "efficiency_breakdown": eval_data.get('efficiency_metrics', {}),
            "scalability_breakdown": eval_data.get('scalability_metrics', {}),
            "bottleneck_analysis": self._analyze_performance_bottlenecks(eval_data)
        }
    
    def _compare_evaluations(self, evaluation_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """比较不同评估"""
        comparison = {
            "relative_performance": {},
            "correlation_analysis": {},
            "consistency_check": {}
        }
        
        # 相对性能比较
        eval_scores = {}
        for eval_type, metrics in evaluation_metrics.items():
            # 计算每个评估的综合分数
            key_metrics = ['accuracy', 'match_rate', 'overall_robustness', 'execution_efficiency']
            scores = [metrics.get(metric, 0.0) for metric in key_metrics if metric in metrics]
            if scores:
                eval_scores[eval_type] = np.mean(scores)
        
        comparison['relative_performance'] = eval_scores
        
        return comparison
    
    def _analyze_perturbation_impacts(self, perturbation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """分析扰动影响"""
        impact_analysis = {
            "most_impactful_perturbation": None,
            "least_impactful_perturbation": None,
            "impact_ranking": []
        }
        
        perturbation_impacts = {}
        
        for perturbation_type, analysis in perturbation_analysis.items():
            stability_metrics = analysis.get('stability_metrics', {})
            impact_score = 1.0 - stability_metrics.get('output_consistency', 0.0)
            perturbation_impacts[perturbation_type] = impact_score
        
        if perturbation_impacts:
            sorted_impacts = sorted(perturbation_impacts.items(), key=lambda x: x[1], reverse=True)
            
            impact_analysis['most_impactful_perturbation'] = sorted_impacts[0]
            impact_analysis['least_impactful_perturbation'] = sorted_impacts[-1]
            impact_analysis['impact_ranking'] = sorted_impacts
        
        return impact_analysis
    
    def _analyze_performance_trade_offs(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能权衡"""
        trade_offs = {
            "accuracy_vs_performance": {},
            "robustness_vs_efficiency": {},
            "scalability_vs_accuracy": {}
        }
        
        # 提取关键指标
        accuracy = 0.0
        performance = 0.0
        robustness = 0.0
        efficiency = 0.0
        
        if 'accuracy_evaluation' in evaluation_results:
            accuracy = evaluation_results['accuracy_evaluation'].get('metrics', {}).get('accuracy', 0.0)
        
        if 'performance_evaluation' in evaluation_results:
            perf_metrics = evaluation_results['performance_evaluation'].get('metrics', {})
            performance = perf_metrics.get('execution_efficiency', 0.0)
            efficiency = perf_metrics.get('time_efficiency', 0.0)
        
        if 'robustness_evaluation' in evaluation_results:
            robustness = evaluation_results['robustness_evaluation'].get('robustness_metrics', {}).get('overall_robustness', 0.0)
        
        # 分析权衡关系
        if accuracy > 0 and performance > 0:
            trade_offs['accuracy_vs_performance'] = {
                "accuracy": accuracy,
                "performance": performance,
                "trade_off_ratio": accuracy / performance if performance > 0 else 0
            }
        
        return trade_offs
    
    def _generate_evaluation_recommendations(self, eval_data: Dict[str, Any], eval_type: str) -> Dict[str, List[str]]:
        """生成评估建议"""
        recommendations = {
            "immediate": [],
            "optimization": [],
            "long_term": [],
            "risk": []
        }
        
        metrics = eval_data.get('metrics', {})
        
        if eval_type == 'accuracy_evaluation':
            if metrics.get('accuracy', 0.0) < 0.7:
                recommendations['immediate'].append("提高算法准确性")
            if metrics.get('error_rate', 0.0) > 0.2:
                recommendations['optimization'].append("优化错误处理机制")
        
        elif eval_type == 'robustness_evaluation':
            if metrics.get('overall_robustness', 0.0) < 0.6:
                recommendations['immediate'].append("增强算法鲁棒性")
                recommendations['risk'].append("监控数据质量变化")
        
        elif eval_type == 'performance_evaluation':
            exec_metrics = eval_data.get('execution_metrics', {})
            if exec_metrics.get('average_query_time', 0.0) > 0.1:
                recommendations['optimization'].append("优化查询执行性能")
        
        return recommendations
    
    def _analyze_accuracy_errors(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析准确性错误"""
        return {
            "error_patterns": [],
            "common_failure_modes": [],
            "error_distribution": {}
        }
    
    def _analyze_degradation_patterns(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析退化模式"""
        return {
            "degradation_trends": [],
            "critical_thresholds": {},
            "recovery_patterns": []
        }
    
    def _analyze_performance_bottlenecks(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能瓶颈"""
        return {
            "identified_bottlenecks": [],
            "resource_constraints": {},
            "optimization_opportunities": []
        }
