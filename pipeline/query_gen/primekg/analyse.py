#!/usr/bin/env python3
"""
分析 noise_queries1.json 文件中 status 为 "no_match" 的项目
提取 reason 字段，去重统计每类 reason 的数量
"""

import json
from collections import Counter
from pathlib import Path


def analyse_no_match_reasons(input_file: str, output_file: str):
    """
    分析 no_match 状态的 reason 字段
    
    Args:
        input_file: 输入的 JSON 文件路径
        output_file: 输出结果的文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从 unmatched 列表中提取所有 status 为 "no_match" 的 reason
    reasons = []
    unmatched = data.get('unmatched', [])
    
    total_queries = len(data.get('queries', []))
    total_unmatched = len(unmatched)
    
    print(f"共有 {total_queries} 条成功查询记录")
    print(f"共有 {total_unmatched} 条 unmatched 记录")
    
    for item in unmatched:
        if item.get('status') == 'no_match':
            reason = item.get('reason', 'unknown')
            reasons.append(reason)
    
    print(f"其中 status 为 no_match 的记录有 {len(reasons)} 条")
    
    # 统计每类 reason 的数量
    reason_counts = Counter(reasons)
    
    # 按数量降序排序
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 输出结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("No Match Reason 统计分析报告\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"输入文件: {input_file}\n")
        f.write(f"成功查询数: {total_queries}\n")
        f.write(f"Unmatched 总数: {total_unmatched}\n")
        f.write(f"No Match 数量: {len(reasons)}\n")
        f.write(f"Reason 种类数(去重后): {len(reason_counts)}\n\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"{'序号':<6} {'数量':<10} {'占比':<10} {'Reason'}\n")
        f.write("-" * 100 + "\n")
        
        for idx, (reason, count) in enumerate(sorted_reasons, 1):
            percentage = (count / len(reasons) * 100) if len(reasons) > 0 else 0
            f.write(f"{idx:<6} {count:<10} {percentage:>6.2f}%    {reason}\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"\n总计: {len(sorted_reasons)} 种不同的 reason, 共 {len(reasons)} 条 no_match 记录\n")
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    print(f"\nReason 种类数(去重后): {len(reason_counts)}")
    print("\n各类 reason 统计:")
    for idx, (reason, count) in enumerate(sorted_reasons, 1):
        percentage = (count / len(reasons) * 100) if len(reasons) > 0 else 0
        print(f"  {idx}. [{count:>5}] ({percentage:>5.2f}%) {reason}")


if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 输入输出文件路径
    input_file = script_dir / 'noise_queries1.json'
    output_file = script_dir / 'no_match_reason_analysis.txt'
    
    analyse_no_match_reasons(str(input_file), str(output_file))
