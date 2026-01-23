#!/usr/bin/env python3
"""合并两个JSON文件"""

import json
import sys

def merge_json_files(file0_path, file1_path, output_path=None):
    """
    将 file0 的内容拼接到 file1 中
    
    Args:
        file0_path: 要拼接的源文件路径
        file1_path: 目标文件路径（会被修改）
        output_path: 输出文件路径（如果为None，则覆盖file1_path）
    """
    if output_path is None:
        output_path = file1_path
    
    print(f'正在读取 {file0_path}...')
    try:
        with open(file0_path, 'r', encoding='utf-8') as f:
            data0 = json.load(f)
        print(f'{file0_path} 包含 {len(data0)} 个项')
    except json.JSONDecodeError as e:
        print(f'错误: {file0_path} 不是有效的JSON文件: {e}')
        print('尝试检查文件是否完整...')
        # 检查文件大小
        import os
        size = os.path.getsize(file0_path)
        print(f'文件大小: {size} 字节')
        return False
    
    print(f'正在读取 {file1_path}...')
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        print(f'{file1_path} 包含 {len(data1)} 个项')
    except json.JSONDecodeError as e:
        print(f'错误: {file1_path} 不是有效的JSON文件: {e}')
        return False
    
    # 拼接数据
    print('正在拼接数据...')
    merged_data = data1 + data0
    print(f'合并后共 {len(merged_data)} 个项')
    
    # 保存到输出文件
    print(f'正在保存到 {output_path}...')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print('拼接完成！')
        return True
    except Exception as e:
        print(f'保存文件时出错: {e}')
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python merge_json_files.py <源文件> <目标文件> [输出文件]")
        print("示例: python merge_json_files.py cleaned0.json cleaned1.json")
        sys.exit(1)
    
    file0 = sys.argv[1]
    file1 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    
    success = merge_json_files(file0, file1, output)
    sys.exit(0 if success else 1)
