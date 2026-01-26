#!/usr/bin/env python3
"""
查看管理查询生成测试的进度
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def check_progress(output_file='management_query_ldbc_fin.json', target_count=1000):
    """检查测试进度"""
    
    if not os.path.exists(output_file):
        print(f'❌ 输出文件不存在: {output_file}')
        print('   测试可能还未开始')
        return
    
    # 获取文件信息
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(os.path.getmtime(output_file))
    
    # 读取文件内容
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 检查文件是否完整（以 ] 结尾）
    is_complete = content.endswith(']')
    
    # 统计已生成的查询数量
    if is_complete:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = len(data)
            status = '✅ 测试已完成！'
        except json.JSONDecodeError as e:
            count = content.count('"pre_validation"')
            status = '⚠️  文件格式可能有问题'
    else:
        # 文件未完成，统计已写入的记录
        # 计算包含 "pre_validation" 的数量（每个查询记录都有这个字段）
        count = content.count('"pre_validation"')
        status = '⏳ 测试进行中...'
    
    # 显示进度
    print('=' * 60)
    print(f'📋 测试进度报告')
    print('=' * 60)
    print(f'📁 输出文件: {output_file}')
    print(f'📊 已生成查询数量: {count}')
    print(f'🎯 目标数量: {target_count}')
    
    if count > 0:
        progress = min(100, (count / target_count) * 100)
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'📈 进度: {progress:.1f}% [{bar}] ({count}/{target_count})')
    
    print(f'📁 文件大小: {file_size_mb:.2f} MB')
    print(f'🕐 最后更新: {mtime.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'📌 状态: {status}')
    print('=' * 60)
    
    # 如果测试还在进行，检查是否有进程在运行
    if not is_complete:
        import subprocess
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'management_test.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print('🔄 检测到测试进程正在运行')
            else:
                print('⚠️  未检测到测试进程，测试可能已停止')
        except:
            pass

if __name__ == '__main__':
    # 可以从命令行参数获取文件名和目标数量
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'management_query_ldbc_fin.json'
    target_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    check_progress(output_file, target_count)
