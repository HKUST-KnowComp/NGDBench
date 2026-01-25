"""
从查询文件中提取节点，并查找这些节点的mention_in关系对应的目标节点id
"""
import os
import sys
import json
import re
from typing import List, Dict, Any, Set

# 添加当前目录到路径，确保可以导入 mcp_base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base.mcp_base import MCPDatabaseExecutor


def extract_node_ids_from_query(query: str) -> Set[str]:
    """
    从Cypher查询中提取节点id
    
    Args:
        query: Cypher查询语句
        
    Returns:
        节点id集合
    """
    node_ids = set()
    
    # 查找形如 {id: 'xxx'} 或 {id: "xxx"} 的模式
    patterns = [
        r"\{[^}]*id\s*:\s*['\"]([^'\"]+)['\"]",  # {id: 'xxx'} 或 {id: "xxx"}
        r"id\s*=\s*['\"]([^'\"]+)['\"]",  # id = 'xxx' 或 id = "xxx"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, query)
        for node_id in matches:
            node_ids.add(node_id)
    
    return node_ids


def extract_node_ids_from_answer(answer: List[Dict[str, Any]]) -> Set[str]:
    """
    从answer中提取所有节点的id
    
    Args:
        answer: 查询结果列表
        
    Returns:
        节点id集合
    """
    node_ids = set()
    
    def extract_from_value(value: Any):
        """递归提取节点id"""
        if isinstance(value, dict):
            # 如果字典中有id字段，提取它
            if 'id' in value:
                node_ids.add(str(value['id']))
            # 递归处理字典中的所有值
            for v in value.values():
                extract_from_value(v)
        elif isinstance(value, list):
            # 递归处理列表中的每个元素
            for item in value:
                extract_from_value(item)
    
    for item in answer:
        extract_from_value(item)
    
    return node_ids


def get_mention_in_target_ids(executor: MCPDatabaseExecutor, node_ids: Set[str]) -> List[str]:
    """
    查询节点的mention_in关系，返回目标节点id列表
    
    Args:
        executor: 数据库执行器
        node_ids: 源节点id集合
        
    Returns:
        目标节点id列表（去重）
    """
    if not node_ids:
        return []
    
    target_ids = set()
    node_id_list = list(node_ids)
    
    # 分批查询，避免查询过长
    batch_size = 100
    for i in range(0, len(node_id_list), batch_size):
        batch_ids = node_id_list[i:i+batch_size]
        
        # 查询这些节点作为源节点的mention_in关系，获取目标节点id
        mention_query = """
        MATCH (n)-[:mention_in]->(m)
        WHERE n.id IN $node_ids
        RETURN DISTINCT m.id AS target_id
        """
        
        try:
            results = executor.execute_query(mention_query, {'node_ids': batch_ids})
            for r in results:
                target_id = r.get('target_id')
                if target_id:
                    target_ids.add(str(target_id))
        except Exception as e:
            print(f"批量查询mention_in关系失败，尝试单个查询: {e}")
            # 如果批量查询失败，尝试单个查询
            for node_id in batch_ids:
                try:
                    mention_query_single = """
                    MATCH (n)-[:mention_in]->(m)
                    WHERE n.id = $node_id
                    RETURN DISTINCT m.id AS target_id
                    """
                    results = executor.execute_query(mention_query_single, {'node_id': node_id})
                    for r in results:
                        target_id = r.get('target_id')
                        if target_id:
                            target_ids.add(str(target_id))
                except Exception as e2:
                    print(f"查询节点 {node_id} 的mention_in关系失败: {e2}")
                    continue
    
    return sorted(list(target_ids))


def process_queries(executor: MCPDatabaseExecutor, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理查询列表，提取mention_in关系的目标节点id
    
    Args:
        executor: 数据库执行器
        queries: 查询列表
        
    Returns:
        处理结果列表
    """
    results = []
    
    for idx, query_data in enumerate(queries, 1):
        query_text = query_data.get('query', '')
        answer = query_data.get('answer', [])
        template_id = query_data.get('template_id', f'query_{idx}')
        
        print(f"处理查询 {idx}/{len(queries)}: {template_id}")
        
        try:
            # 1. 从query中提取节点id
            query_node_ids = extract_node_ids_from_query(query_text)
            print(f"  从query中提取到 {len(query_node_ids)} 个节点id: {query_node_ids}")
            
            # 2. 从answer中提取节点id
            answer_node_ids = extract_node_ids_from_answer(answer)
            print(f"  从answer中提取到 {len(answer_node_ids)} 个节点id: {answer_node_ids}")
            
            # 3. 合并所有节点id
            all_node_ids = query_node_ids | answer_node_ids
            print(f"  总共 {len(all_node_ids)} 个唯一节点id")
            
            # 4. 查询这些节点的mention_in关系，获取目标节点id
            mention_in_target_ids = get_mention_in_target_ids(executor, all_node_ids)
            print(f"  找到 {len(mention_in_target_ids)} 个mention_in目标节点id")
            
            # 5. 构建结果
            result_item = {
                'template_id': template_id,
                'template_type': query_data.get('template_type', ''),
                'query': query_text,
                'parameters': query_data.get('parameters', {}),
                'query_node_ids': sorted(list(query_node_ids)),
                'answer_node_ids': sorted(list(answer_node_ids)),
                'mention_in_nodes': mention_in_target_ids,
                'error': None,
            }
            
            results.append(result_item)
            
        except Exception as e:
            print(f"处理查询时出错: {e}")
            import traceback
            traceback.print_exc()
            
            result_item = {
                'template_id': template_id,
                'template_type': query_data.get('template_type', ''),
                'query': query_text,
                'parameters': query_data.get('parameters', {}),
                'query_node_ids': [],
                'answer_node_ids': [],
                'mention_in_nodes': [],
                'error': str(e),
            }
            results.append(result_item)
    
    return results


def main():
    """主函数：提取节点并查找mention_in关系"""
    # 数据库连接配置
    # 根据实际情况修改端口和认证信息
    uri = "bolt://localhost:7690"  # 根据实际MCP数据库端口修改
    user = "neo4j"
    password = "fei123456"
    
    # 输入和输出文件路径
    input_json_file = "../query_gen/query_results_mcp2.json"
    output_json_file = "mcp_execution_results.json"
    
    # 创建MCP数据库执行器
    executor = MCPDatabaseExecutor(uri, user, password)
    
    try:
        # 连接数据库
        executor.connect()
        
        # 读取查询文件
        input_path = os.path.join(os.path.dirname(__file__), input_json_file)
        with open(input_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        print(f"共读取 {len(queries)} 个查询")
        
        # 处理查询并提取mention_in节点信息
        results = process_queries(executor, queries)
        
        # 保存结果
        output_path = os.path.join(os.path.dirname(__file__), output_json_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        total_queries = len(results)
        queries_with_mention_in = sum(1 for r in results if r.get('mention_in_nodes') and len(r.get('mention_in_nodes', [])) > 0)
        total_mention_in_nodes = sum(len(r.get('mention_in_nodes', [])) for r in results)
        
        print("\n处理完成！")
        print(f"共处理了 {total_queries} 个查询")
        print(f"其中 {queries_with_mention_in} 个查询包含mention_in关系")
        print(f"共记录了 {total_mention_in_nodes} 个mention_in目标节点")
        print(f"结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 关闭连接
        executor.close()


if __name__ == "__main__":
    main()
