"""
指标报告基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import time
from datetime import datetime


class BaseMetricReport(ABC):
    """指标报告基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化指标报告
        
        Args:
            config: 报告配置参数
        """
        self.config = config
        self.report_format = config.get('format', 'json')  # 'json', 'html', 'markdown', 'csv'
        self.include_details = config.get('include_details', True)
        self.include_visualizations = config.get('include_visualizations', False)
        self._report_data = {}
        
    @abstractmethod
    def generate_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成报告
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            生成的报告
        """
        pass
    
    def create_report_header(self, title: str, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建报告头部信息
        
        Args:
            title: 报告标题
            evaluation_results: 评估结果
            
        Returns:
            报告头部信息
        """
        return {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "framework_version": "0.1.0",
            "evaluation_summary": {
                "total_evaluations": len(evaluation_results.get('evaluations', [])),
                "evaluation_types": list(evaluation_results.keys()),
                "dataset_info": evaluation_results.get('dataset_info', {}),
                "algorithm_info": evaluation_results.get('algorithm_info', {})
            }
        }
    
    def format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化指标数据
        
        Args:
            metrics: 原始指标数据
            
        Returns:
            格式化后的指标数据
        """
        formatted_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, float):
                # 保留4位小数
                formatted_metrics[key] = round(value, 4)
            elif isinstance(value, dict):
                formatted_metrics[key] = self.format_metrics(value)
            elif isinstance(value, list):
                formatted_metrics[key] = [
                    self.format_metrics(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                formatted_metrics[key] = value
        
        return formatted_metrics
    
    def create_summary_statistics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        创建汇总统计信息
        
        Args:
            metrics_list: 指标列表
            
        Returns:
            汇总统计信息
        """
        if not metrics_list:
            return {}
        
        summary = {
            "count": len(metrics_list),
            "aggregated_metrics": {}
        }
        
        # 收集所有数值指标
        all_metrics = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # 计算统计量
        import numpy as np
        for key, values in all_metrics.items():
            summary['aggregated_metrics'][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return summary
    
    def create_performance_comparison(self, baseline_metrics: Dict[str, Any], 
                                    current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建性能比较
        
        Args:
            baseline_metrics: 基准指标
            current_metrics: 当前指标
            
        Returns:
            性能比较结果
        """
        comparison = {
            "baseline": baseline_metrics,
            "current": current_metrics,
            "improvements": {},
            "degradations": {},
            "overall_change": 0.0
        }
        
        changes = []
        
        for key in baseline_metrics:
            if key in current_metrics:
                baseline_val = baseline_metrics[key]
                current_val = current_metrics[key]
                
                if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                    if baseline_val != 0:
                        change = (current_val - baseline_val) / baseline_val
                        changes.append(change)
                        
                        if change > 0:
                            comparison['improvements'][key] = {
                                "baseline": baseline_val,
                                "current": current_val,
                                "improvement": change
                            }
                        elif change < 0:
                            comparison['degradations'][key] = {
                                "baseline": baseline_val,
                                "current": current_val,
                                "degradation": abs(change)
                            }
        
        if changes:
            comparison['overall_change'] = sum(changes) / len(changes)
        
        return comparison
    
    def export_report(self, report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        导出报告到文件
        
        Args:
            report_data: 报告数据
            filename: 文件名（可选）
            
        Returns:
            导出的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ngdb_report_{timestamp}"
        
        if self.report_format == 'json':
            return self._export_json(report_data, filename)
        elif self.report_format == 'html':
            return self._export_html(report_data, filename)
        elif self.report_format == 'markdown':
            return self._export_markdown(report_data, filename)
        elif self.report_format == 'csv':
            return self._export_csv(report_data, filename)
        else:
            raise ValueError(f"不支持的报告格式: {self.report_format}")
    
    def _export_json(self, report_data: Dict[str, Any], filename: str) -> str:
        """导出JSON格式报告"""
        filepath = f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        return filepath
    
    def _export_html(self, report_data: Dict[str, Any], filename: str) -> str:
        """导出HTML格式报告"""
        filepath = f"{filename}.html"
        html_content = self._generate_html_content(report_data)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return filepath
    
    def _export_markdown(self, report_data: Dict[str, Any], filename: str) -> str:
        """导出Markdown格式报告"""
        filepath = f"{filename}.md"
        markdown_content = self._generate_markdown_content(report_data)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return filepath
    
    def _export_csv(self, report_data: Dict[str, Any], filename: str) -> str:
        """导出CSV格式报告"""
        filepath = f"{filename}.csv"
        csv_content = self._generate_csv_content(report_data)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        return filepath
    
    def _generate_html_content(self, report_data: Dict[str, Any]) -> str:
        """生成HTML内容"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_data.get('header', {}).get('title', 'NGDB Report')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .metric-name {{ font-weight: bold; }}
        .metric-value {{ color: #007acc; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_data.get('header', {}).get('title', 'NGDB Report')}</h1>
        <p>Generated at: {report_data.get('header', {}).get('generated_at', '')}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        {self._format_dict_as_html(report_data.get('summary', {}))}
    </div>
    
    <div class="section">
        <h2>Detailed Results</h2>
        {self._format_dict_as_html(report_data.get('detailed_results', {}))}
    </div>
</body>
</html>
        """
        return html
    
    def _generate_markdown_content(self, report_data: Dict[str, Any]) -> str:
        """生成Markdown内容"""
        markdown = f"""# {report_data.get('header', {}).get('title', 'NGDB Report')}

Generated at: {report_data.get('header', {}).get('generated_at', '')}

## Summary

{self._format_dict_as_markdown(report_data.get('summary', {}))}

## Detailed Results

{self._format_dict_as_markdown(report_data.get('detailed_results', {}))}
"""
        return markdown
    
    def _generate_csv_content(self, report_data: Dict[str, Any]) -> str:
        """生成CSV内容"""
        # 简化的CSV导出，主要导出数值指标
        csv_lines = ["Metric,Value"]
        
        def extract_metrics(data, prefix=""):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    metric_name = f"{prefix}{key}" if prefix else key
                    csv_lines.append(f"{metric_name},{value}")
                elif isinstance(value, dict):
                    new_prefix = f"{prefix}{key}." if prefix else f"{key}."
                    extract_metrics(value, new_prefix)
        
        extract_metrics(report_data.get('summary', {}))
        extract_metrics(report_data.get('detailed_results', {}))
        
        return "\n".join(csv_lines)
    
    def _format_dict_as_html(self, data: Dict[str, Any]) -> str:
        """将字典格式化为HTML"""
        if not data:
            return "<p>No data available</p>"
        
        html = "<table>"
        for key, value in data.items():
            if isinstance(value, dict):
                html += f"<tr><td colspan='2'><strong>{key}</strong></td></tr>"
                html += f"<tr><td colspan='2'>{self._format_dict_as_html(value)}</td></tr>"
            else:
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _format_dict_as_markdown(self, data: Dict[str, Any], level: int = 0) -> str:
        """将字典格式化为Markdown"""
        if not data:
            return "No data available\n"
        
        markdown = ""
        indent = "  " * level
        
        for key, value in data.items():
            if isinstance(value, dict):
                markdown += f"{indent}- **{key}**:\n"
                markdown += self._format_dict_as_markdown(value, level + 1)
            else:
                markdown += f"{indent}- **{key}**: {value}\n"
        
        return markdown
    
    def validate_report_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证报告数据的完整性
        
        Args:
            report_data: 报告数据
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 检查必需字段
        required_fields = ['header', 'summary']
        for field in required_fields:
            if field not in report_data:
                validation_result["errors"].append(f"缺少必需字段: {field}")
                validation_result["is_valid"] = False
        
        # 检查数据类型
        if 'summary' in report_data and not isinstance(report_data['summary'], dict):
            validation_result["errors"].append("summary字段必须是字典类型")
            validation_result["is_valid"] = False
        
        # 检查数值指标的合理性
        if 'summary' in report_data:
            self._validate_metrics(report_data['summary'], validation_result)
        
        return validation_result
    
    def _validate_metrics(self, metrics: Dict[str, Any], validation_result: Dict[str, Any]):
        """验证指标数据"""
        for key, value in metrics.items():
            if isinstance(value, float):
                if value < 0 and key in ['accuracy', 'precision', 'recall', 'f1_score']:
                    validation_result["warnings"].append(f"指标 {key} 的值为负数: {value}")
                elif value > 1 and key in ['accuracy', 'precision', 'recall', 'f1_score']:
                    validation_result["warnings"].append(f"指标 {key} 的值大于1: {value}")
            elif isinstance(value, dict):
                self._validate_metrics(value, validation_result)
