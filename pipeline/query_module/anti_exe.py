"""
在Neo4j数据库中执行anti_query查询并记录结果
"""
import os
import sys
import json

# 添加当前目录到路径，确保可以导入 db_base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_base import DatabaseExecutor


def filter_anti_queries(queries: list) -> list:
    """
    从查询列表中筛选出包含 anti_query 的条目
    
    Args:
        queries: 查询列表
        
    Returns:
        包含 anti_query 的查询列表，每个查询的 query 字段被替换为 anti_query
    """
    anti_queries = []
    for query_data in queries:
        if 'anti_query' in query_data and query_data['anti_query']:
            # 创建新的查询对象，将 anti_query 作为 query 字段
            anti_query_data = query_data.copy()
            anti_query_data['query'] = query_data['anti_query']
            # 保留原始 query 作为 original_query 字段以便对比
            anti_query_data['original_query'] = query_data.get('query', '')
            anti_query_data['is_anti_query'] = True
            anti_queries.append(anti_query_data)
    
    print(f"从 {len(queries)} 个查询中筛选出 {len(anti_queries)} 个 anti_query")
    return anti_queries


def main():
    """主函数：执行anti_query并保存结果"""
    # 数据库连接配置
    # Docker容器映射端口 7692 -> 7687，所以使用 localhost:7692
    uri = "bolt://localhost:7692"
    user = "neo4j"
    password = "fei123456"
    
    # 输入和输出文件路径
    input_json_file = "../query_gen/query/ldbc_snb_finbench/query_results_ldbcfin_with_orderby.json"
    output_json_file = "anti_query_execution_results.json"
    
    # 创建数据库执行器
    executor = DatabaseExecutor(uri, user, password)
    
    try:
        # 连接数据库
        executor.connect()
        
        # 读取所有查询
        all_queries = executor.read_queries_from_json(input_json_file)
        
        # 筛选出 anti_query
        anti_queries = filter_anti_queries(all_queries)
        
        if not anti_queries:
            print("没有找到任何 anti_query，退出")
            return
        
        # 执行查询并记录结果，启用增量保存（一边执行一边记录）
        results = executor.execute_queries_batch(
            anti_queries, 
            compare_with_original=True,  # 与原始 answer 比较
            incremental_save=True,  # 启用增量保存
            output_file_path=output_json_file
        )
        
        print("\n执行完成！")
        print(f"共执行了 {len(results)} 个 anti_query")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        raise
    finally:
        # 关闭连接
        executor.close()


if __name__ == "__main__":
    main()
