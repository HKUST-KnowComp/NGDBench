"""
评估框架基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time


class BaseEvaluationHarness(ABC):
    """评估框架基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估框架
        
        Args:
            config: 评估配置参数
        """
        self.config = config
        self.evaluation_metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.tolerance = config.get('tolerance', 1e-6)
        self._evaluation_history = []
        
    @abstractmethod
    def evaluate(self, ground_truth: Dict[str, Any], 
                algorithm_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行评估
        
        Args:
            ground_truth: 基准真实结果
            algorithm_output: 算法输出结果
            
        Returns:
            评估结果
        """
        pass
    
    def compare_results(self, ground_truth: Any, algorithm_output: Any, 
                       comparison_type: str = 'exact') -> Dict[str, Any]:
        """
        比较两个结果
        
        Args:
            ground_truth: 基准结果
            algorithm_output: 算法结果
            comparison_type: 比较类型 ('exact', 'approximate', 'structural')
            
        Returns:
            比较结果
        """
        if comparison_type == 'exact':
            return self._exact_comparison(ground_truth, algorithm_output)
        elif comparison_type == 'approximate':
            return self._approximate_comparison(ground_truth, algorithm_output)
        elif comparison_type == 'structural':
            return self._structural_comparison(ground_truth, algorithm_output)
        else:
            raise ValueError(f"不支持的比较类型: {comparison_type}")
    
    def _exact_comparison(self, ground_truth: Any, algorithm_output: Any) -> Dict[str, Any]:
        """精确比较"""
        if isinstance(ground_truth, dict) and isinstance(algorithm_output, dict):
            return self._compare_dicts(ground_truth, algorithm_output)
        elif isinstance(ground_truth, list) and isinstance(algorithm_output, list):
            return self._compare_lists(ground_truth, algorithm_output)
        elif isinstance(ground_truth, (int, float)) and isinstance(algorithm_output, (int, float)):
            return self._compare_numbers(ground_truth, algorithm_output)
        else:
            return {
                "match": ground_truth == algorithm_output,
                "ground_truth_type": type(ground_truth).__name__,
                "algorithm_output_type": type(algorithm_output).__name__
            }
    
    def _approximate_comparison(self, ground_truth: Any, algorithm_output: Any) -> Dict[str, Any]:
        """近似比较"""
        if isinstance(ground_truth, (int, float)) and isinstance(algorithm_output, (int, float)):
            diff = abs(ground_truth - algorithm_output)
            relative_error = diff / max(abs(ground_truth), 1e-10)
            return {
                "match": diff <= self.tolerance,
                "absolute_error": diff,
                "relative_error": relative_error,
                "tolerance": self.tolerance
            }
        elif isinstance(ground_truth, dict) and isinstance(algorithm_output, dict):
            return self._compare_dicts_approximate(ground_truth, algorithm_output)
        else:
            return self._exact_comparison(ground_truth, algorithm_output)
    
    def _structural_comparison(self, ground_truth: Any, algorithm_output: Any) -> Dict[str, Any]:
        """结构比较"""
        if isinstance(ground_truth, dict) and isinstance(algorithm_output, dict):
            gt_keys = set(ground_truth.keys())
            ao_keys = set(algorithm_output.keys())
            
            return {
                "key_match": gt_keys == ao_keys,
                "missing_keys": gt_keys - ao_keys,
                "extra_keys": ao_keys - gt_keys,
                "common_keys": gt_keys & ao_keys,
                "structure_similarity": len(gt_keys & ao_keys) / len(gt_keys | ao_keys) if gt_keys | ao_keys else 1.0
            }
        elif isinstance(ground_truth, list) and isinstance(algorithm_output, list):
            return {
                "length_match": len(ground_truth) == len(algorithm_output),
                "ground_truth_length": len(ground_truth),
                "algorithm_output_length": len(algorithm_output),
                "length_ratio": len(algorithm_output) / max(len(ground_truth), 1)
            }
        else:
            return {"structure_match": type(ground_truth) == type(algorithm_output)}
    
    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """比较字典"""
        keys1, keys2 = set(dict1.keys()), set(dict2.keys())
        common_keys = keys1 & keys2
        
        matches = sum(1 for k in common_keys if dict1[k] == dict2[k])
        
        return {
            "match": dict1 == dict2,
            "key_match": keys1 == keys2,
            "value_matches": matches,
            "total_common_keys": len(common_keys),
            "accuracy": matches / len(common_keys) if common_keys else 0.0,
            "missing_keys": keys1 - keys2,
            "extra_keys": keys2 - keys1
        }
    
    def _compare_dicts_approximate(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """近似比较字典"""
        keys1, keys2 = set(dict1.keys()), set(dict2.keys())
        common_keys = keys1 & keys2
        
        matches = 0
        total_error = 0.0
        
        for k in common_keys:
            if isinstance(dict1[k], (int, float)) and isinstance(dict2[k], (int, float)):
                error = abs(dict1[k] - dict2[k])
                if error <= self.tolerance:
                    matches += 1
                total_error += error
            elif dict1[k] == dict2[k]:
                matches += 1
        
        return {
            "approximate_match": matches == len(common_keys),
            "value_matches": matches,
            "total_common_keys": len(common_keys),
            "accuracy": matches / len(common_keys) if common_keys else 0.0,
            "average_error": total_error / len(common_keys) if common_keys else 0.0,
            "tolerance": self.tolerance
        }
    
    def _compare_lists(self, list1: List, list2: List) -> Dict[str, Any]:
        """比较列表"""
        if len(list1) != len(list2):
            return {
                "match": False,
                "length_match": False,
                "length1": len(list1),
                "length2": len(list2)
            }
        
        matches = sum(1 for a, b in zip(list1, list2) if a == b)
        
        return {
            "match": list1 == list2,
            "element_matches": matches,
            "total_elements": len(list1),
            "accuracy": matches / len(list1) if list1 else 1.0
        }
    
    def _compare_numbers(self, num1: float, num2: float) -> Dict[str, Any]:
        """比较数字"""
        diff = abs(num1 - num2)
        relative_error = diff / max(abs(num1), 1e-10)
        
        return {
            "match": diff <= self.tolerance,
            "absolute_error": diff,
            "relative_error": relative_error,
            "ground_truth": num1,
            "algorithm_output": num2
        }
    
    def calculate_metrics(self, comparisons: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            comparisons: 比较结果列表
            
        Returns:
            评估指标
        """
        if not comparisons:
            return {}
        
        metrics = {}
        
        # 准确率
        if 'accuracy' in self.evaluation_metrics:
            accuracies = [comp.get('accuracy', 0.0) for comp in comparisons if 'accuracy' in comp]
            metrics['accuracy'] = np.mean(accuracies) if accuracies else 0.0
        
        # 匹配率
        matches = [comp.get('match', False) for comp in comparisons]
        metrics['match_rate'] = np.mean(matches) if matches else 0.0
        
        # 错误率
        metrics['error_rate'] = 1.0 - metrics.get('match_rate', 0.0)
        
        # 平均绝对误差
        abs_errors = [comp.get('absolute_error', 0.0) for comp in comparisons if 'absolute_error' in comp]
        if abs_errors:
            metrics['mean_absolute_error'] = np.mean(abs_errors)
            metrics['std_absolute_error'] = np.std(abs_errors)
        
        # 平均相对误差
        rel_errors = [comp.get('relative_error', 0.0) for comp in comparisons if 'relative_error' in comp]
        if rel_errors:
            metrics['mean_relative_error'] = np.mean(rel_errors)
            metrics['std_relative_error'] = np.std(rel_errors)
        
        return metrics
    
    def record_evaluation(self, evaluation_result: Dict[str, Any]):
        """记录评估结果"""
        evaluation_record = {
            "timestamp": time.time(),
            "metrics": evaluation_result.get('metrics', {}),
            "summary": evaluation_result.get('summary', {}),
            "num_comparisons": evaluation_result.get('num_comparisons', 0)
        }
        self._evaluation_history.append(evaluation_record)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self._evaluation_history.copy()
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self._evaluation_history:
            return {"message": "No evaluations performed"}
        
        latest = self._evaluation_history[-1]
        
        return {
            "total_evaluations": len(self._evaluation_history),
            "latest_evaluation": latest,
            "average_metrics": self._calculate_average_metrics()
        }
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """计算平均指标"""
        if not self._evaluation_history:
            return {}
        
        all_metrics = {}
        for record in self._evaluation_history:
            metrics = record.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in all_metrics.items()}
