"""
从JSON文件中提取包含聚合操作的查询项
"""
import json
import re
import os


def contains_aggregation(query: str) -> bool:
    """
    检查查询是否包含聚合操作
    
    Args:
        query: Cypher查询字符串
        
    Returns:
        如果包含聚合操作返回True，否则返回False
    """
    # 定义聚合操作的正则表达式模式（不区分大小写）
    aggregation_patterns = [
        r'\bcount\s*\(',           # count(
        r'\bsum\s*\(',              # sum(
        r'\bavg\s*\(',              # avg(
        r'\bmax\s*\(',              # max(
        r'\bmin\s*\(',              # min(
        r'\bcollect\s*\(',          # collect(
        r'\bCOUNT\s*\(',            # COUNT(
        r'\bSUM\s*\(',              # SUM(
        r'\bAVG\s*\(',              # AVG(
        r'\bMAX\s*\(',              # MAX(
        r'\bMIN\s*\(',              # MIN(
        r'\bCOLLECT\s*\(',          # COLLECT(
    ]
    
    query_upper = query.upper()
    for pattern in aggregation_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    
    return False


def extract_aggregation_queries(input_file: str, output_file: str):
    """
    从输入文件中提取包含聚合操作的查询项，保存到输出文件
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
    """
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共 {len(data)} 个查询项")
    
    # 提取包含聚合操作的项
    aggregation_items = []
    for i, item in enumerate(data):
        query = item.get('query', '')
        if contains_aggregation(query):
            aggregation_items.append(item)
    
    print(f"找到 {len(aggregation_items)} 个包含聚合操作的查询项")
    
    # 保存到新文件
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregation_items, f, indent=2, ensure_ascii=False)
    
    print(f"完成！已保存 {len(aggregation_items)} 个包含聚合操作的查询项到 {output_file}")
    
    # 显示一些统计信息
    aggregation_types = {
        'count': 0,
        'sum': 0,
        'avg': 0,
        'max': 0,
        'min': 0,
        'collect': 0
    }
    
    for item in aggregation_items:
        query = item.get('query', '').upper()
        if 'COUNT(' in query:
            aggregation_types['count'] += 1
        if 'SUM(' in query:
            aggregation_types['sum'] += 1
        if 'AVG(' in query:
            aggregation_types['avg'] += 1
        if 'MAX(' in query:
            aggregation_types['max'] += 1
        if 'MIN(' in query:
            aggregation_types['min'] += 1
        if 'COLLECT(' in query:
            aggregation_types['collect'] += 1
    
    print("\n聚合操作统计:")
    for agg_type, count in aggregation_types.items():
        if count > 0:
            print(f"  {agg_type.upper()}: {count}")


if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "query/ldbc_snb_finbench/noise_query_execution_step1_ldbcfin_translated.json"
    output_file = "query/ldbc_snb_finbench/noise_query_execution_step1_ldbcfin_translated_aggregation.json"
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_file)
    
    extract_aggregation_queries(input_path, output_path)
