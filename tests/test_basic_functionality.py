#!/usr/bin/env python3
"""
NGDB框架基本功能测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from ngdb_framework import NGDBFramework


class TestBasicFunctionality(unittest.TestCase):
    """基本功能测试类"""
    
    def setUp(self):
        """测试设置"""
        self.basic_config = {
            "data_source": {
                "type": "generator",
                "generator_type": "karate_club"
            },
            "perturbation": {
                "type": "random",
                "remove_nodes": True,
                "node_removal_ratio": 0.1,
                "remove_edges": True,
                "edge_removal_ratio": 0.1,
                "seed": 42
            },
            "methodology": {
                "type": "graph_algorithm",
                "algorithm_type": "pagerank",
                "algorithm_params": {
                    "alpha": 0.85,
                    "max_iter": 50
                }
            },
            "queries": {
                "total_queries": 5,
                "query_type_ratio": {"read": 1.0, "update": 0.0}
            },
            "evaluation": {
                "accuracy": {"enabled": True},
                "robustness": {"enabled": True},
                "performance": {"enabled": True}
            },
            "reports": {
                "export_files": False,
                "comprehensive": {"enabled": True}
            }
        }
    
    def test_framework_initialization(self):
        """测试框架初始化"""
        framework = NGDBFramework(self.basic_config)
        self.assertIsNotNone(framework.data_source)
        self.assertIsNotNone(framework.perturbation_generator)
        self.assertIsNotNone(framework.methodology)
        self.assertIsNotNone(framework.query_generator)
        self.assertTrue(len(framework.evaluators) > 0)
        self.assertTrue(len(framework.report_generators) > 0)
    
    def test_data_source(self):
        """测试数据源"""
        framework = NGDBFramework(self.basic_config)
        graph = framework.data_source.graph
        self.assertIsNotNone(graph)
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)
    
    def test_perturbation_generator(self):
        """测试扰动生成器"""
        framework = NGDBFramework(self.basic_config)
        original_graph = framework.data_source.graph
        
        perturbed_graph, perturbation_info = framework.perturbation_generator.apply_perturbation(original_graph)
        
        self.assertIsNotNone(perturbed_graph)
        self.assertIsNotNone(perturbation_info)
        self.assertIn('operations', perturbation_info)
        
        # 扰动后的图应该节点或边数量有所减少
        self.assertLessEqual(perturbed_graph.number_of_nodes(), original_graph.number_of_nodes())
        self.assertLessEqual(perturbed_graph.number_of_edges(), original_graph.number_of_edges())
    
    def test_methodology(self):
        """测试算法执行"""
        framework = NGDBFramework(self.basic_config)
        graph = framework.data_source.graph
        
        # 生成简单查询
        queries = [
            {"type": "pagerank", "parameters": {"alpha": 0.85}}
        ]
        
        results = framework.methodology.execute(graph, queries)
        
        self.assertIsNotNone(results)
        self.assertIn('algorithm_results', results)
    
    def test_query_generation(self):
        """测试查询生成"""
        framework = NGDBFramework(self.basic_config)
        graph = framework.data_source.graph
        
        queries = framework.query_generator.generate_mixed_queries(graph, 5)
        
        self.assertIsNotNone(queries)
        self.assertEqual(len(queries), 5)
        
        for query in queries:
            self.assertIn('type', query)
            self.assertIn('parameters', query)
    
    def test_evaluation(self):
        """测试评估功能"""
        framework = NGDBFramework(self.basic_config)
        
        # 创建模拟的算法结果
        mock_groundtruth = {
            "algorithm_results": {
                "scores": {"node1": 0.5, "node2": 0.3, "node3": 0.2}
            },
            "query_results": {
                "query1": {"type": "pagerank", "result": {"node1": 0.5}}
            }
        }
        
        mock_algorithm_output = {
            "algorithm_results": {
                "scores": {"node1": 0.48, "node2": 0.32, "node3": 0.2}
            },
            "query_results": {
                "query1": {"type": "pagerank", "result": {"node1": 0.48}}
            }
        }
        
        # 测试准确性评估
        if 'accuracy' in framework.evaluators:
            acc_result = framework.evaluators['accuracy'].evaluate(mock_groundtruth, mock_algorithm_output)
            self.assertIsNotNone(acc_result)
            self.assertIn('metrics', acc_result)
    
    def test_report_generation(self):
        """测试报告生成"""
        framework = NGDBFramework(self.basic_config)
        
        # 创建模拟的评估结果
        mock_evaluation_results = {
            "accuracy_evaluation": {
                "metrics": {"accuracy": 0.85, "match_rate": 0.8},
                "summary": {"overall_accuracy": 0.85}
            },
            "robustness_evaluation": {
                "robustness_metrics": {"overall_robustness": 0.75},
                "summary": {"is_robust": True}
            }
        }
        
        # 测试综合报告生成
        if 'comprehensive' in framework.report_generators:
            report = framework.report_generators['comprehensive'].generate_report(mock_evaluation_results)
            self.assertIsNotNone(report)
            self.assertIn('header', report)
            self.assertIn('executive_summary', report)
    
    def test_full_benchmark_run(self):
        """测试完整基准测试运行"""
        # 使用更简化的配置以加快测试速度
        simple_config = self.basic_config.copy()
        simple_config['queries']['total_queries'] = 3
        simple_config['methodology']['algorithm_params']['max_iter'] = 10
        
        framework = NGDBFramework(simple_config)
        
        try:
            results = framework.run_benchmark()
            
            # 验证结果结构
            self.assertIsNotNone(results)
            self.assertIn('execution_metadata', results)
            self.assertIn('data_info', results)
            self.assertIn('algorithm_info', results)
            self.assertIn('evaluation_results', results)
            self.assertIn('reports', results)
            
            # 验证执行元数据
            exec_metadata = results['execution_metadata']
            self.assertIn('total_time', exec_metadata)
            self.assertGreater(exec_metadata['total_time'], 0)
            
            # 验证数据信息
            data_info = results['data_info']
            self.assertIn('original_graph_stats', data_info)
            self.assertIn('perturbed_graph_stats', data_info)
            
            # 验证评估结果
            eval_results = results['evaluation_results']
            self.assertTrue(len(eval_results) > 0)
            
            # 验证报告
            reports = results['reports']
            self.assertIn('comprehensive_report', reports)
            
            print("完整基准测试运行成功!")
            
        except Exception as e:
            self.fail(f"完整基准测试运行失败: {e}")


def run_quick_test():
    """运行快速测试"""
    print("=== NGDB框架快速功能测试 ===")
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加基本测试
    suite.addTest(TestBasicFunctionality('test_framework_initialization'))
    suite.addTest(TestBasicFunctionality('test_data_source'))
    suite.addTest(TestBasicFunctionality('test_perturbation_generator'))
    suite.addTest(TestBasicFunctionality('test_methodology'))
    suite.addTest(TestBasicFunctionality('test_query_generation'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_full_test():
    """运行完整测试"""
    print("=== NGDB框架完整功能测试 ===")
    
    # 运行所有测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicFunctionality)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NGDB框架测试')
    parser.add_argument('--quick', action='store_true', help='运行快速测试')
    parser.add_argument('--full', action='store_true', help='运行完整测试')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.full:
        success = run_full_test()
    else:
        # 默认运行快速测试
        success = run_quick_test()
    
    if success:
        print("\n✅ 所有测试通过!")
        exit(0)
    else:
        print("\n❌ 部分测试失败!")
        exit(1)
