"""
在Neo4j数据库中执行噪声查询并比较结果
"""
import os
import sys
import json
from typing import List, Dict, Any

# 添加当前目录到路径，确保可以导入 base.db_base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base.db_base import DatabaseExecutor

# 添加 handler 目录到路径，以便导入清洗函数
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from handler.cleaner import _clean_judge_answers


def execute_judge_queries_on_noise_graph(
    input_file: str,
    uri: str = "bolt://localhost:7693",
    user: str = "neo4j",
    password: str = "fei123456",
    max_unique_answers: int = 20
) -> str:
    """
    在噪声图上执行 judge 类型文件的 template_query 和 anti_template_query，
    并将结果添加到新文件的 NoiseCandidateSet 字段中（不修改原文件）
    
    Args:
        input_file: 输入的 JSON 文件路径（包含 template_query 和 anti_template_query）
        uri: 数据库连接 URI，默认为噪声图端口 7693
        user: 数据库用户名
        password: 数据库密码
        max_unique_answers: 每个查询结果中最多返回的答案数量，默认 20
    
    Returns:
        输出文件路径
    """
    # 创建输出文件路径（原文件名 + _with_noise_candidates.json）
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.basename(input_file)
    input_name, input_ext = os.path.splitext(input_basename)
    output_file = os.path.join(input_dir, f"{input_name}_with_noise_candidates{input_ext}")
    
    # 创建数据库执行器
    executor = DatabaseExecutor(uri, user, password)
    
    try:
        # 连接数据库
        executor.connect()
        
        # 读取 JSON 文件
        print(f"读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("文件为空，无需处理")
            return output_file
        
        print(f"共 {len(data)} 个查询需要处理")
        print(f"输出文件: {output_file}")
        
        # 处理每个条目，一边执行一边保存
        for idx, item in enumerate(data, 1):
            template_query = item.get("template_query", "")
            anti_template_query = item.get("anti_template_query", "")
            
            if not template_query or not anti_template_query:
                print(f"跳过第 {idx} 个条目：缺少 template_query 或 anti_template_query")
                # 即使跳过也保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                continue
            
            print(f"\n处理第 {idx}/{len(data)} 个查询...")
            
            try:
                # 执行 template_query（作为 valid_answer）
                print(f"  执行 template_query...")
                template_results = executor.execute_query(template_query)
                cleaned_template_results = _clean_judge_answers(template_results)
                
                # 执行 anti_template_query（作为 invalid_answer）
                print(f"  执行 anti_template_query...")
                anti_template_results = executor.execute_query(anti_template_query)
                cleaned_anti_template_results = _clean_judge_answers(anti_template_results)
                
                # 限制返回的结果数量，避免返回太多结果
                if len(cleaned_template_results) > max_unique_answers:
                    cleaned_template_results = cleaned_template_results[:max_unique_answers]
                    print(f"  ⚠ valid_answer 结果过多，已限制为前 {max_unique_answers} 条")
                
                if len(cleaned_anti_template_results) > max_unique_answers:
                    cleaned_anti_template_results = cleaned_anti_template_results[:max_unique_answers]
                    print(f"  ⚠ invalid_answer 结果过多，已限制为前 {max_unique_answers} 条")
                
                # 组织成 NoiseCandidateSet 格式
                item["NoiseCandidateSet"] = {
                    "valid_answer": cleaned_template_results,
                    "invalid_answer": cleaned_anti_template_results
                }
                
                print(f"  ✓ 完成：valid_answer {len(cleaned_template_results)} 条，invalid_answer {len(cleaned_anti_template_results)} 条")
                
            except Exception as e:
                print(f"  ✗ 执行失败: {e}")
                # 即使失败也添加字段，但值为空
                item["NoiseCandidateSet"] = {
                    "valid_answer": [],
                    "invalid_answer": []
                }
            
            # 每处理完一个查询，立即保存到文件（增量保存）
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  ✓ 已保存到文件: {output_file}")
            except Exception as save_error:
                print(f"  ⚠ 保存文件失败: {save_error}")
        
        print(f"\n处理完成！结果已保存到: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        # 即使出错，也尝试保存已处理的结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已保存部分结果到: {output_file}")
        except:
            pass
        raise
    finally:
        # 关闭连接
        executor.close()


def main():
    """主函数：执行查询并保存结果"""
    # 数据库连接配置
    # Docker容器映射端口 7693 -> 7687，所以使用 localhost:7693
    uri = "bolt://localhost:7693"
    user = "neo4j"
    password = "fei123456"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果提供了参数，使用新功能处理 judge 类型文件
        input_file = sys.argv[1]
        if not os.path.isabs(input_file):
            # 如果是相对路径，尝试多种方式解析
            # 1. 先尝试相对于当前工作目录
            if os.path.exists(input_file):
                input_file = os.path.abspath(input_file)
            else:
                # 2. 尝试相对于脚本所在目录
                script_dir = os.path.dirname(os.path.abspath(__file__))
                candidate_path = os.path.join(script_dir, input_file)
                if os.path.exists(candidate_path):
                    input_file = candidate_path
                else:
                    # 3. 尝试相对于项目根目录
                    project_root = os.path.dirname(script_dir)
                    candidate_path = os.path.join(project_root, input_file)
                    if os.path.exists(candidate_path):
                        input_file = candidate_path
                    else:
                        # 如果都不存在，使用原始路径（让后续代码报错）
                        input_file = os.path.abspath(input_file)
        
        # 检查是否有 max_unique_answers 参数
        max_unique_answers = 20  # 默认值
        if len(sys.argv) > 2:
            try:
                max_unique_answers = int(sys.argv[2])
            except ValueError:
                print(f"警告：无法解析 max_unique_answers 参数 '{sys.argv[2]}'，使用默认值 {max_unique_answers}")
        
        print(f"处理 judge 类型文件: {input_file}")
        print(f"max_unique_answers: {max_unique_answers}")
        output_file = execute_judge_queries_on_noise_graph(input_file, uri, user, password, max_unique_answers=max_unique_answers)
        print(f"结果已保存到: {output_file}")
    else:
        # 默认行为：执行原有的噪声查询
        # 输入和输出文件路径（使用基于脚本目录的绝对路径）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(project_root, "query_gen", "query", "ldbc_snb_finbench", "noise_query_results_step1_ldbcfin_cleaned.json"),
            os.path.join(project_root, "query_gen", "noise_query_results_step1_ldbcfin_cleaned.json"),
        ]
        
        input_json_file = None
        for path in possible_paths:
            if os.path.exists(path):
                input_json_file = path
                break
        
        if input_json_file is None:
            # 如果都不存在，使用第一个路径（让后续代码报错）
            input_json_file = possible_paths[0]
            print(f"警告：文件不存在，将尝试: {input_json_file}")
        else:
            print(f"找到输入文件: {input_json_file}")
        
        output_json_file = os.path.join(script_dir, "noise_execution_step1_ldbcfin_results.json")
        
        # 创建数据库执行器
        executor = DatabaseExecutor(uri, user, password, node_id_key="_node_id")
        
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
