"""
NGDB框架主程序 - 图算法基准测试框架的核心协调器
"""

from typing import Any, Dict, List, Optional
import time
import logging
from pathlib import Path

# 导入各个模块
from .data_source import BaseDataSource, FileDataSource, GeneratorDataSource
from .perturbation_generator import (
    BasePerturbationGenerator, RandomPerturbationGenerator,
    SemanticPerturbationGenerator, TopologyPerturbationGenerator
)
from .methodology import BaseMethodology, GraphAlgorithmMethodology, GraphRAGMethodology, GNNMethodology
from .query_module import QueryGenerator
from .evaluation_harness import AccuracyEvaluator, RobustnessEvaluator, PerformanceEvaluator
from .metric_report import ComprehensiveReport, IncompletenessReport, NoiseReport, UpdateReport


class NGDBBench:
    """NGDB基准测试框架主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化NGDB框架
        
        Args:
            config: 框架配置参数
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # 初始化各个组件
        self.data_source = None
        self.perturbation_generator = None
        self.methodology = None
        self.query_generator = None
        self.evaluators = {}
        self.report_generators = {}
        
        # 执行状态
        self.execution_state = {
            "current_step": "initialized",
            "start_time": None,
            "end_time": None,
            "results": {}
        }
        
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ngdb_framework.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('NGDBFramework')
    
    def _initialize_components(self):
        """初始化框架组件"""
        try:
            # 初始化数据源
            self._initialize_data_source()
            
            # 初始化扰动生成器
            self._initialize_perturbation_generator()
            
            # 初始化算法方法
            self._initialize_methodology()
            
            # 初始化查询生成器
            self._initialize_query_generator()
            
            # 初始化评估器
            self._initialize_evaluators()
            
            # 初始化报告生成器
            self._initialize_report_generators()
            
            self.logger.info("NGDB框架组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def _initialize_data_source(self):
        """初始化数据源"""
        data_config = self.config.get('data_source', {})
        db_config = self.config.get('database_config', {})
        data_path = data_config.get('data_path', None)
        source_type = data_config.get('type', 'file')
        
        if source_type == 'file':
            self.data_source = FileDataSource(
                file_format=data_config.get('file_format', 'auto'),
                data_set_name=data_config.get('data_set_name', 'ldbc_snb_bi'),
                db_config = db_config,
                data_path = data_path
            )
        elif source_type == 'generator':
            self.data_source = GeneratorDataSource(
                generator_type=data_config.get('generator_type', 'erdos_renyi'),
                **data_config.get('generator_params', {})
            )
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
    
    def _initialize_perturbation_generator(self):
        """初始化扰动生成器"""
        perturbation_config = self.config.get('perturbation', {})
        perturbation_method = perturbation_config.get('method', 'random')

        if perturbation_method == 'random':
            self.perturbation_generator = RandomPerturbationGenerator(self.config)
        elif perturbation_method == 'semantic':
            self.perturbation_generator = SemanticPerturbationGenerator(self.config)
        elif perturbation_method == 'topology':
            self.perturbation_generator = TopologyPerturbationGenerator(self.config)
        else:
            raise ValueError(f"不支持的扰动类型: {perturbation_method}")
    
    def _initialize_methodology(self):
        """初始化算法方法"""
        methodology_config = self.config.get('methodology', {})
        methodology_type = methodology_config.get('type', 'graph_algorithm')
        
        if methodology_type == 'graph_algorithm':
            self.methodology = GraphAlgorithmMethodology(methodology_config)
        elif methodology_type == 'graph_rag':
            self.methodology = GraphRAGMethodology(methodology_config)
        elif methodology_type == 'gnn':
            self.methodology = GNNMethodology(methodology_config)
        else:
            raise ValueError(f"不支持的算法类型: {methodology_type}")
    
    def _initialize_query_generator(self):
        """初始化查询生成器"""
        query_config = self.config.get('queries', {})
        self.query_generator = QueryGenerator(query_config)
    
    def _initialize_evaluators(self):
        """初始化评估器"""
        evaluation_config = self.config.get('evaluation', {})
        
        # 准确性评估器
        if evaluation_config.get('accuracy', {}).get('enabled', True):
            self.evaluators['accuracy'] = AccuracyEvaluator(
                evaluation_config.get('accuracy', {})
            )
        
        # 鲁棒性评估器
        if evaluation_config.get('robustness', {}).get('enabled', True):
            self.evaluators['robustness'] = RobustnessEvaluator(
                evaluation_config.get('robustness', {})
            )
        
        # 性能评估器
        if evaluation_config.get('performance', {}).get('enabled', True):
            self.evaluators['performance'] = PerformanceEvaluator(
                evaluation_config.get('performance', {})
            )
    
    def _initialize_report_generators(self):
        """初始化报告生成器"""
        report_config = self.config.get('reports', {})
        
        # 综合报告
        if report_config.get('comprehensive', {}).get('enabled', True):
            self.report_generators['comprehensive'] = ComprehensiveReport(
                report_config.get('comprehensive', {})
            )
        
        # 不完整性报告
        if report_config.get('incompleteness', {}).get('enabled', False):
            self.report_generators['incompleteness'] = IncompletenessReport(
                report_config.get('incompleteness', {})
            )
        
        # 噪声报告
        if report_config.get('noise', {}).get('enabled', False):
            self.report_generators['noise'] = NoiseReport(
                report_config.get('noise', {})
            )
        
        # 更新报告
        if report_config.get('update', {}).get('enabled', False):
            self.report_generators['update'] = UpdateReport(
                report_config.get('update', {})
            )
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        运行完整的基准测试
        
        Returns:
            基准测试结果
        """
        self.logger.info("开始运行NGDB基准测试")
        self.execution_state['start_time'] = time.time()
        self.execution_state['current_step'] = "running"
        
        try:
            # 步骤1: 数据准备和扰动
            self.logger.info("步骤1: 数据准备和扰动")
            original_graph, perturbed_graph, groundtruth = self._prepare_data()
            
            # 步骤2: 生成查询
            self.logger.info("步骤2: 生成查询")
            queries = self.get_queries(perturbed_graph)
            
            # 步骤3: 执行算法
            self.logger.info("步骤3: 执行算法")
            algorithm_results = self._execute_algorithm(perturbed_graph, queries)
            groundtruth_results = self._execute_algorithm(groundtruth, queries)
            
            # 步骤4: 评估结果
            self.logger.info("步骤4: 评估结果")
            evaluation_results = self._evaluate_results(groundtruth_results, algorithm_results)
            
            # 步骤5: 生成报告
            self.logger.info("步骤5: 生成报告")
            reports = self._generate_reports(evaluation_results)
            
            # 整合结果
            benchmark_results = {
                "execution_metadata": {
                    "start_time": self.execution_state['start_time'],
                    "end_time": time.time(),
                    "total_time": time.time() - self.execution_state['start_time'],
                    "framework_version": "0.1.0"
                },
                "data_info": {
                    "original_graph_stats": self._get_graph_stats(original_graph),
                    "perturbed_graph_stats": self._get_graph_stats(perturbed_graph),
                    "perturbation_info": self.execution_state['results'].get('perturbation_info', {})
                },
                "algorithm_info": {
                    "methodology_type": self.config.get('methodology', {}).get('type', 'unknown'),
                    "algorithm_name": self.config.get('methodology', {}).get('algorithm_name', 'unknown'),
                    "algorithm_params": self.config.get('methodology', {}).get('algorithm_params', {})
                },
                "queries_info": {
                    "total_queries": len(queries),
                    "query_types": self._analyze_query_types(queries)
                },
                "evaluation_results": evaluation_results,
                "reports": reports
            }
            
            self.execution_state['current_step'] = "completed"
            self.execution_state['end_time'] = time.time()
            self.execution_state['results'] = benchmark_results
            
            self.logger.info(f"基准测试完成，总耗时: {benchmark_results['execution_metadata']['total_time']:.2f}秒")
            
            return benchmark_results
            
        except Exception as e:
            self.execution_state['current_step'] = "failed"
            self.execution_state['end_time'] = time.time()
            self.logger.error(f"基准测试执行失败: {e}")
            raise
    
    def _prepare_data(self) -> tuple:
        """准备数据和扰动"""
        # 加载原始数据
        gnd_data_config = self.data_source.graph # data_config可以包含数据集路径和容器信息
        dataset_name = self.config.get('data_source', {}).get('data_set_name', 'ldbc_snb_bi')
        # 应用扰动
        perturbation_config = self.config.get('perturbation', {})
        if perturbation_config.get('data_path') is None:
            perturbed_graph, perturbation_info = self.perturbation_generator.apply_perturbation()
        else:
            perturbed_graph, perturbation_info = self.perturbation_generator.apply_perturbation()
        
        self.execution_state['results']['perturbation_info'] = perturbation_info
        self.logger.info(f"应用扰动后: {perturbed_graph.number_of_nodes()}个节点, {perturbed_graph.number_of_edges()}条边")
        
        return gnd_data_config, perturbed_graph, groundtruth
    
    def get_queries(self, graph) -> List[Dict[str, Any]]:
        """生成查询"""
        queries = self.query_generator.generate_mixed_queries(graph, 
                                                            total_queries=self.config.get('queries', {}).get('total_queries', 50))
        self.logger.info(f"生成了{len(queries)}个查询")
        return queries
    
    def _execute_algorithm(self, graph, queries) -> Dict[str, Any]:
        """执行算法"""
        start_time = time.time()
        results = self.methodology.execute(graph, queries)
        execution_time = time.time() - start_time
        
        # 添加执行元数据
        results['execution_metadata'] = {
            'total_time': execution_time,
            'algorithm_time': execution_time,  # 简化
            'num_queries': len(queries)
        }
        
        return results
    
    def _evaluate_results(self, groundtruth_results: Dict[str, Any], 
                         algorithm_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估结果"""
        evaluation_results = {}
        
        for eval_name, evaluator in self.evaluators.items():
            try:
                self.logger.info(f"执行{eval_name}评估")
                eval_result = evaluator.evaluate(groundtruth_results, algorithm_results)
                evaluation_results[f"{eval_name}_evaluation"] = eval_result
            except Exception as e:
                self.logger.error(f"{eval_name}评估失败: {e}")
                evaluation_results[f"{eval_name}_evaluation"] = {"error": str(e)}
        
        return evaluation_results
    
    def _generate_reports(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        reports = {}
        
        for report_name, report_generator in self.report_generators.items():
            try:
                self.logger.info(f"生成{report_name}报告")
                report = report_generator.generate_report(evaluation_results)
                reports[f"{report_name}_report"] = report
                
                # 导出报告文件
                if self.config.get('reports', {}).get('export_files', True):
                    filename = f"ngdb_{report_name}_report"
                    filepath = report_generator.export_report(report, filename)
                    self.logger.info(f"{report_name}报告已导出到: {filepath}")
                    
            except Exception as e:
                self.logger.error(f"{report_name}报告生成失败: {e}")
                reports[f"{report_name}_report"] = {"error": str(e)}
        
        return reports
    
    def _get_graph_stats(self, graph) -> Dict[str, Any]:
        """获取图统计信息"""
        import networkx as nx
        
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed()
        }
        
        try:
            if not graph.is_directed():
                stats["is_connected"] = nx.is_connected(graph)
                if nx.is_connected(graph) and graph.number_of_nodes() > 1:
                    stats["diameter"] = nx.diameter(graph)
                    stats["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
        except:
            pass
        
        return stats
    
    def _analyze_query_types(self, queries: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析查询类型"""
        query_types = {}
        for query in queries:
            qtype = query.get('type', 'unknown')
            query_types[qtype] = query_types.get(qtype, 0) + 1
        return query_types
    
    def get_execution_status(self) -> Dict[str, Any]:
        """获取执行状态"""
        return self.execution_state.copy()
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        import json
        
        if 'results' in self.execution_state:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.execution_state['results'], f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"结果已保存到: {filepath}")
        else:
            self.logger.warning("没有可保存的结果")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """从文件加载结果"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.logger.info(f"结果已从{filepath}加载")
        return results


def create_framework_from_config(config_path: str) -> NGDBBench:
    """
    从配置文件创建框架实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        NGDB框架实例
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return NGDBBench(config)


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NGDB图算法基准测试框架')
    parser.add_argument('--config', '-c', required=True, help='配置文件路径')
    parser.add_argument('--output', '-o', help='结果输出文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建框架实例
        framework = create_framework_from_config(args.config)
        
        # 运行基准测试
        results = framework.run_benchmark()
        
        # 保存结果
        if args.output:
            framework.save_results(args.output)
        
        print("基准测试完成!")
        print(f"总体性能评分: {results.get('reports', {}).get('comprehensive_report', {}).get('executive_summary', {}).get('overall_performance', {}).get('score', 'N/A')}")
        
    except Exception as e:
        print(f"执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
