#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从参数文件填充查询模板，生成查询文件
"""

import os
import csv
import re
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "base_query_template"
PARAMS_DIR = BASE_DIR / "params"
QUERIES_DIR = BASE_DIR / "queries"

# 确保 queries 目录存在
QUERIES_DIR.mkdir(exist_ok=True)

def extract_placeholders(template_content):
    """提取模板中的所有占位符，返回 (类型, 位置) 列表"""
    placeholders = []
    # 匹配 %d 和 %f
    for match in re.finditer(r'%([df])', template_content):
        placeholders.append((match.group(1), match.start()))
    return placeholders

def get_param_columns(csv_file):
    """读取 CSV 文件的列名"""
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        header = next(reader)
        return header

def read_params(csv_file, num_queries=5):
    """从 CSV 文件读取参数，返回前 num_queries 行数据"""
    params_list = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        # 使用 reader 而不是 DictReader，因为可能有重复的列名
        reader = csv.reader(f, delimiter='|')
        header = next(reader)
        
        # 处理重复的列名
        column_map = {}
        for idx, col_name in enumerate(header):
            if col_name not in column_map:
                column_map[col_name] = []
            column_map[col_name].append(idx)
        
        for i, row in enumerate(reader):
            if i >= num_queries:
                break
            # 过滤空行
            if not any(row):
                continue
            
            # 转换为字典，处理重复列名
            param_row = {}
            for col_name, indices in column_map.items():
                if len(indices) == 1:
                    param_row[col_name] = row[indices[0]] if indices[0] < len(row) else ''
                else:
                    # 如果有多个同名列，使用 id1, id2 等命名
                    for j, idx in enumerate(indices):
                        if j == 0:
                            param_row[col_name] = row[idx] if idx < len(row) else ''
                        else:
                            param_row[f'{col_name}{j+1}'] = row[idx] if idx < len(row) else ''
            
            params_list.append(param_row)
    return params_list

def fill_template(template_content, param_row, columns):
    """使用参数填充模板"""
    # 提取所有占位符（按顺序）
    placeholders = extract_placeholders(template_content)
    
    if not placeholders:
        return template_content
    
    # 获取参数值
    # 优先使用 id，如果没有则使用 id1，再没有则使用第一个 id2
    id_val = param_row.get('id') or param_row.get('id1') or param_row.get('id2') or ''
    # 处理 id2 和 id22（当有两个 id2 列时）
    # 如果 id_val 来自 id2，那么 id2_val 应该来自 id22（第二个 id2 列）
    if id_val == param_row.get('id2'):
        id2_val = param_row.get('id22') or ''
    else:
        id2_val = param_row.get('id22') or param_row.get('id2') or ''
    start_time = param_row.get('startTime', '')
    end_time = param_row.get('endTime', '')
    threshold = param_row.get('threshold', '0')
    limit = param_row.get('truncationLimit', param_row.get('limit', '500'))
    
    # 构建参数值列表
    param_values = []
    id_count = 0
    threshold_count = 0
    time_pair_count = 0
    limit_used = False
    
    # 统计整数占位符的总数
    total_int_placeholders = sum(1 for t, _ in placeholders if t == 'd')
    
    for placeholder_type, _ in placeholders:
        if placeholder_type == 'd':  # 整数占位符
            # 第一个 %d 使用 id
            if id_count == 0 and id_val:
                param_values.append(id_val)
                id_count += 1
            # 第二个 %d 使用 id2（如果有）
            elif id_count == 1 and id2_val:
                param_values.append(id2_val)
                id_count += 1
            # 其他情况：优先使用时间参数，最后才使用 limit
            else:
                # 计算已使用的 id 数量
                used_ids = id_count
                # 计算剩余的 %d 数量（不包括当前这个）
                remaining_d = sum(1 for t, _ in placeholders[placeholders.index((placeholder_type, _))+1:] if t == 'd')
                
                # 如果这是最后一个 %d 且还没有使用 limit，且有限制参数，使用 limit
                # 但只有在确实需要 limit 的情况下（比如 read8）
                if remaining_d == 0 and not limit_used and limit and 'limit' in str(template_content).lower():
                    param_values.append(limit)
                    limit_used = True
                # 否则使用时间参数
                elif start_time and end_time:
                    # 检查模板中是否有 smaller_than 和 greater_than（read3 的特殊情况）
                    # smaller_than 应该使用 endTime，greater_than 应该使用 startTime
                    current_idx = placeholders.index((placeholder_type, _))
                    placeholder_pos = placeholders[current_idx][1]
                    
                    # 检查当前占位符是否在 smaller_than 或 greater_than 附近
                    text_before = template_content[:placeholder_pos]
                    text_after = template_content[placeholder_pos:]
                    
                    # 检查 smaller_than 和 greater_than 的位置
                    has_smaller = 'smaller_than' in template_content
                    has_greater = 'greater_than' in template_content
                    
                    if has_smaller and has_greater:
                        # 找到 smaller_than 和 greater_than 后面的 %d 位置
                        smaller_d_pos = -1
                        greater_d_pos = -1
                        for i, (t, pos) in enumerate(placeholders):
                            text_before_placeholder = template_content[:pos]
                            if 'smaller_than' in text_before_placeholder and smaller_d_pos == -1:
                                # 找到 smaller_than 后面的第一个 %d
                                smaller_d_pos = i
                            if 'greater_than' in text_before_placeholder and greater_d_pos == -1:
                                # 找到 greater_than 后面的第一个 %d
                                greater_d_pos = i
                        
                        # 如果当前占位符是 smaller_than 对应的，使用 endTime
                        if current_idx == smaller_d_pos:
                            param_values.append(end_time)
                        # 如果当前占位符是 greater_than 对应的，使用 startTime
                        elif current_idx == greater_d_pos:
                            param_values.append(start_time)
                        else:
                            # 正常情况：交替使用 startTime 和 endTime
                            if time_pair_count % 2 == 0:
                                param_values.append(start_time)
                            else:
                                param_values.append(end_time)
                            time_pair_count += 1
                    else:
                        # 正常情况：交替使用 startTime 和 endTime
                        if time_pair_count % 2 == 0:
                            param_values.append(start_time)
                        else:
                            param_values.append(end_time)
                        time_pair_count += 1
                else:
                    param_values.append('0')
        elif placeholder_type == 'f':  # 浮点数占位符
            # 使用 threshold
            param_values.append(threshold if threshold else '0.0')
            threshold_count += 1
    
    # 将参数值转换为正确的类型
    typed_values = []
    for i, (placeholder_type, _) in enumerate(placeholders):
        value = param_values[i]
        if placeholder_type == 'd':  # 整数
            try:
                typed_values.append(int(value))
            except (ValueError, TypeError):
                typed_values.append(0)
        elif placeholder_type == 'f':  # 浮点数
            try:
                typed_values.append(float(value))
            except (ValueError, TypeError):
                typed_values.append(0.0)
        else:
            typed_values.append(value)
    
    # 填充模板
    try:
        filled_query = template_content % tuple(typed_values)
    except (TypeError, ValueError) as e:
        print(f"警告: 填充模板时出错: {e}")
        print(f"  占位符数量: {len(placeholders)}, 参数数量: {len(typed_values)}")
        print(f"  参数值: {typed_values}")
        # 返回原始模板
        filled_query = template_content
    
    return filled_query

def generate_queries_for_template(template_file, param_file, query_prefix):
    """为单个模板生成查询"""
    # 读取模板
    with open(template_file, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # 读取参数
    columns = get_param_columns(param_file)
    params_list = read_params(param_file, num_queries=5)
    
    if not params_list:
        print(f"警告: {param_file} 中没有足够的参数数据")
        return
    
    # 生成查询
    for i, param_row in enumerate(params_list, 1):
        query_content = fill_template(template_content, param_row, columns)
        
        # 生成查询文件名
        query_filename = f"{query_prefix}_{i}.cypher"
        query_path = QUERIES_DIR / query_filename
        
        # 保存查询
        with open(query_path, 'w', encoding='utf-8') as f:
            f.write(query_content)
        
        print(f"生成: {query_filename}")

def main():
    """主函数"""
    # 获取所有模板文件（包括 .cypher 和 .ctpher）
    template_files = sorted(list(TEMPLATE_DIR.glob("*.cypher")) + list(TEMPLATE_DIR.glob("*.ctpher")))
    
    # 处理 simple_read 文件（可能没有对应的参数文件）
    simple_read_templates = [f for f in template_files if f.name.startswith('simple_read')]
    complex_templates = [f for f in template_files if f.name.startswith('read') and not f.name.startswith('simple_read')]
    
    # 处理 complex 查询
    for template_file in complex_templates:
        # 从文件名提取编号：read1.cypher -> 1
        match = re.search(r'read(\d+)', template_file.name)
        if match:
            query_num = match.group(1)
            param_file = PARAMS_DIR / f"complex_{query_num}_param.csv"
            
            if param_file.exists():
                print(f"\n处理模板: {template_file.name}")
                query_prefix = f"read{query_num}"
                generate_queries_for_template(template_file, param_file, query_prefix)
            else:
                print(f"警告: 未找到参数文件 {param_file}")
    
    # 处理 simple_read 查询（如果有对应的参数文件）
    for template_file in simple_read_templates:
        match = re.search(r'simple_read(\d+)', template_file.name)
        if match:
            query_num = match.group(1)
            # simple_read 可能使用 complex 的参数文件，或者有单独的参数文件
            # 这里先尝试使用 complex 的参数文件
            param_file = PARAMS_DIR / f"complex_{query_num}_param.csv"
            
            if param_file.exists():
                print(f"\n处理模板: {template_file.name}")
                query_prefix = f"simple_read{query_num}"
                generate_queries_for_template(template_file, param_file, query_prefix)
            else:
                print(f"警告: 未找到参数文件 {param_file}")

if __name__ == "__main__":
    main()
