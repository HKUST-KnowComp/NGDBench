#!/usr/bin/env python3
"""
NGDB benchmark运行示例脚本
"""

import sys
import os
import yaml
from ngdb_framework import NGDBBench
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """主函数"""
    print("🚀 欢迎使用NGDB benchmark框架!")
    print("=" * 50)
    
    # 从配置文件读取配置
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"📄 已加载配置文件: {config_path}")
    
    
    try:
        print("📊 创建NGDB框架实例...")
        framework = NGDBBench(config)
        
        print("🔄 开始运行基准测试...")
        results = framework.run_benchmark()
        
        print("\n" + "=" * 50)
        print("📈 基准测试结果摘要")
        print("=" * 50)
        
        # 执行信息
        exec_metadata = results.get('execution_metadata', {})
        print(f"⏱️  执行时间: {exec_metadata.get('total_time', 0):.2f}秒")
        
        # 数据信息
        data_info = results.get('data_info', {})
        original_stats = data_info.get('original_graph_stats', {})
        perturbed_stats = data_info.get('perturbed_graph_stats', {})
        
        print(f"📊 原始图: {original_stats.get('num_nodes', 0)}个节点, {original_stats.get('num_edges', 0)}条边")
        print(f"🔀 扰动图: {perturbed_stats.get('num_nodes', 0)}个节点, {perturbed_stats.get('num_edges', 0)}条边")
        
        # 扰动信息
        perturbation_info = data_info.get('perturbation_info', {})
        if 'operations' in perturbation_info:
            print(f"🎯 扰动操作: {len(perturbation_info['operations'])}个")
        
        # 查询信息
        queries_info = results.get('queries_info', {})
        print(f"❓ 查询数量: {queries_info.get('total_queries', 0)}")
        
        # 评估结果
        evaluation_results = results.get('evaluation_results', {})
        
        print("\n📋 评估结果:")
        
        if 'accuracy_evaluation' in evaluation_results:
            acc_metrics = evaluation_results['accuracy_evaluation'].get('metrics', {})
            accuracy = acc_metrics.get('accuracy', 0)
            match_rate = acc_metrics.get('match_rate', 0)
            print(f"  ✅ 准确性: {accuracy:.3f}")
            print(f"  🎯 匹配率: {match_rate:.3f}")
        
        if 'robustness_evaluation' in evaluation_results:
            rob_metrics = evaluation_results['robustness_evaluation'].get('robustness_metrics', {})
            overall_robustness = rob_metrics.get('overall_robustness', 0)
            print(f"  🛡️  鲁棒性: {overall_robustness:.3f}")
        
        if 'performance_evaluation' in evaluation_results:
            perf_metrics = evaluation_results['performance_evaluation'].get('execution_metrics', {})
            avg_query_time = perf_metrics.get('average_query_time', 0)
            print(f"  ⚡ 平均查询时间: {avg_query_time:.4f}秒")
        
        # 综合报告
        reports = results.get('reports', {})
        if 'comprehensive_report' in reports:
            comp_report = reports['comprehensive_report']
            exec_summary = comp_report.get('executive_summary', {})
            overall_perf = exec_summary.get('overall_performance', {})
            
            score = overall_perf.get('score', 0)
            grade = overall_perf.get('grade', 'N/A')
            print(f"\n🏆 总体性能评分: {score:.3f} (等级: {grade})")
            
            # 关键发现
            key_findings = exec_summary.get('key_findings', [])
            if key_findings:
                print("\n🔍 关键发现:")
                for finding in key_findings[:3]:  # 显示前3个
                    print(f"  • {finding}")
            
            # 主要关注点
            main_concerns = exec_summary.get('main_concerns', [])
            if main_concerns:
                print("\n⚠️  主要关注点:")
                for concern in main_concerns[:2]:  # 显示前2个
                    print(f"  • {concern.get('description', 'N/A')}")
        
        print("\n" + "=" * 50)
        print("✅ 基准测试完成!")
        
        # 保存结果
        framework.save_results("ngdb_example_results.json")
        print("💾 结果已保存到: ngdb_example_results.json")
        
        # 性能评级
        if 'comprehensive_report' in reports:
            score = reports['comprehensive_report'].get('executive_summary', {}).get('overall_performance', {}).get('score', 0)
            if score >= 0.8:
                print("🌟 算法性能优秀!")
            elif score >= 0.6:
                print("👍 算法性能良好!")
            elif score >= 0.4:
                print("⚡ 算法性能一般，有改进空间")
            else:
                print("🔧 算法性能需要显著改进")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
