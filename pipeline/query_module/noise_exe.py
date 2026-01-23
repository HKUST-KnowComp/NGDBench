"""
在Neo4j数据库中执行噪声查询并比较结果
"""
import os
import sys

# 添加当前目录到路径，确保可以导入 db_base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_base import DatabaseExecutor


def main():
    """主函数：执行查询并保存结果"""
    # 数据库连接配置
    # Docker容器映射端口 7693 -> 7687，所以使用 localhost:7693
    uri = "bolt://localhost:7693"
    user = "neo4j"
    password = "fei123456"
    
    # 输入和输出文件路径
    input_json_file = "../query_gen/query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned.json"
    output_json_file = "noise_execution_step1_ldbcfin_results.json"
    
    # 创建数据库执行器
    executor = DatabaseExecutor(uri, user, password)
    
    try:
        # 连接数据库
        executor.connect()
        
        # 读取查询
        queries = executor.read_queries_from_json(input_json_file)
        
        # 执行查询并比较结果，启用增量保存（一边执行一边记录）
        results = executor.execute_queries_batch(
            queries, 
            compare_with_original=True,
            incremental_save=True,  # 启用增量保存
            output_file_path=output_json_file
        )
        
        # 如果未启用增量保存，则在这里保存结果
        # executor.save_results_to_json(results, output_json_file)
        
        print("\n执行完成！")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        raise
    finally:
        # 关闭连接
        executor.close()


if __name__ == "__main__":
    main()
