#!/usr/bin/env python3
"""清洗脚本：筛选并清洗 pre_validation 和 post_validation answer 不同的项"""

import sys
import os

# 添加当前目录到路径，以便导入 handler 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler.cleaner import filter_and_clean_validation_differences

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python clean_validation_differences.py <输入文件路径> [输出文件路径]")
        print("示例: python clean_validation_differences.py management_query_ldbc_fin.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = input_file.replace(".json", "_filtered.json")
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    print(f"开始清洗文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()
    
    try:
        filter_and_clean_validation_differences(
            input_file=input_file,
            output_file=output_file
        )
        print(f"\n清洗完成！输出文件: {output_file}")
    except Exception as e:
        print(f"清洗失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
