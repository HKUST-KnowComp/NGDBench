"""
MCP数据库执行器，继承自DatabaseExecutor，支持提取mention_in关系的节点id
"""
import json
import os
import sys
import re
from typing import List, Dict, Any, Optional

# 确保可以导入基类
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 从同目录导入db_base
from .db_base import DatabaseExecutor


class MCPDatabaseExecutor(DatabaseExecutor):
    """MCP数据库执行器，支持提取mention_in关系的节点id"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化MCP数据库连接
        
        Args:
            uri: 数据库连接URI
            user: 用户名
            password: 密码
        """
        super().__init__(uri, user, password)
    
    def extract_mention_in_node_ids(self, query: str, execution_result: Optional[List[Dict[str, Any]]], 
                                   query_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        提取查询结果中相关节点的mention_in关系对应的节点id
        
        Args:
            query: 原始查询语句
            execution_result: 查询执行结果
            query_data: 查询数据（包含parameters等信息）
            
        Returns:
            mention_in关系列表，每个关系包含source_id和target_id
        """
        mention_in_nodes = []
        
        try:
            # 收集所有相关节点id
            all_node_ids = set()
            
            # 1. 从parameters中提取节点id
            if query_data and 'parameters' in query_data:
                params = query_data['parameters']
                if 'VALUE' in params:
                    all_node_ids.add(str(params['VALUE']))
            
            # 2. 从查询模式中提取节点id（通过解析Cypher查询）
            # 查找形如 {id: 'xxx'} 的模式
            id_patterns = re.findall(r"\{[^}]*id\s*:\s*['\"]([^'\"]+)['\"]", query)
            for node_id in id_patterns:
                all_node_ids.add(node_id)
            
            # 3. 从执行结果中提取节点id
            if execution_result:
                for record in execution_result:
                    for key, value in record.items():
                        if value is None:
                            continue
                        
                        # 如果字段名包含.id，直接提取
                        if '.id' in key.lower():
                            if isinstance(value, (str, int)):
                                all_node_ids.add(str(value))
                            continue
                        
                        # 如果值是字符串或数字，可能是节点id
                        if isinstance(value, (str, int)):
                            # 排除明显的非节点id值（如count结果）
                            if key.lower() not in ['cnt', 'count', 'sum', 'avg', 'max', 'min', 'incoming_relationships']:
                                # 排除数字类型的聚合结果
                                if not (isinstance(value, int) and key.lower() in ['cnt', 'count']):
                                    all_node_ids.add(str(value))
                        # 如果值是列表，递归查找
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, (str, int)):
                                    all_node_ids.add(str(item))
                                elif isinstance(item, dict):
                                    # 处理列表中的字典（可能是节点对象）
                                    if 'id' in item:
                                        all_node_ids.add(str(item['id']))
                        # 如果值是字典，可能是节点对象
                        elif isinstance(value, dict):
                            # Neo4j节点对象通常有id属性
                            if 'id' in value:
                                all_node_ids.add(str(value['id']))
                            # 也可能直接包含节点id字段
                            for k, v in value.items():
                                if k.endswith('.id') or k == 'id':
                                    if isinstance(v, (str, int)):
                                        all_node_ids.add(str(v))
            
            # 4. 查询这些节点的mention_in关系
            if all_node_ids:
                # 批量查询mention_in关系，提高效率
                node_id_list = list(all_node_ids)
                # 分批查询，避免查询过长
                batch_size = 100
                for i in range(0, len(node_id_list), batch_size):
                    batch_ids = node_id_list[i:i+batch_size]
                    # 构建查询，查找这些节点作为源节点或目标节点的mention_in关系
                    mention_query = """
                    MATCH (n)-[:mention_in]->(m)
                    WHERE n.id IN $node_ids OR m.id IN $node_ids
                    RETURN DISTINCT n.id AS source_id, m.id AS target_id
                    """
                    try:
                        results = self.execute_query(mention_query, {'node_ids': batch_ids})
                        for r in results:
                            source_id = r.get('source_id')
                            target_id = r.get('target_id')
                            if source_id and target_id:
                                mention_in_nodes.append({
                                    'source_id': source_id,
                                    'target_id': target_id
                                })
                    except Exception as e:
                        # 如果批量查询失败，尝试单个查询
                        print(f"批量查询mention_in关系失败，尝试单个查询: {e}")
                        for node_id in batch_ids:
                            try:
                                mention_query_single = """
                                MATCH (n)-[:mention_in]->(m)
                                WHERE n.id = $node_id OR m.id = $node_id
                                RETURN DISTINCT n.id AS source_id, m.id AS target_id
                                """
                                results = self.execute_query(mention_query_single, {'node_id': node_id})
                                for r in results:
                                    source_id = r.get('source_id')
                                    target_id = r.get('target_id')
                                    if source_id and target_id:
                                        mention_in_nodes.append({
                                            'source_id': source_id,
                                            'target_id': target_id
                                        })
                            except Exception as e2:
                                print(f"查询节点 {node_id} 的mention_in关系失败: {e2}")
                                continue
            
            # 去重
            seen = set()
            unique_mention_in_nodes = []
            for node in mention_in_nodes:
                key = (str(node['source_id']), str(node['target_id']))
                if key not in seen:
                    seen.add(key)
                    unique_mention_in_nodes.append(node)
            
            return unique_mention_in_nodes
            
        except Exception as e:
            print(f"提取mention_in节点id时出错: {e}")
            import traceback
            traceback.print_exc()
            return mention_in_nodes
    
    def execute_query_with_mention_in(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                                      query_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行查询并提取mention_in关系的节点id
        
        Args:
            query: Cypher查询语句
            parameters: 查询参数（可选）
            query_data: 查询数据（包含parameters等信息）
            
        Returns:
            包含执行结果和mention_in节点信息的字典
        """
        # 执行原始查询
        execution_result = self.execute_query(query, parameters)
        
        # 提取mention_in关系的节点id
        mention_in_nodes = self.extract_mention_in_node_ids(query, execution_result, query_data)
        
        return {
            'execution_result': execution_result,
            'mention_in_nodes': mention_in_nodes
        }
    
    def execute_queries_batch(self, queries: List[Dict[str, Any]], 
                             compare_with_original: bool = True,
                             incremental_save: bool = False,
                             output_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量执行查询，并记录mention_in关系的节点id
        
        Args:
            queries: 查询列表
            compare_with_original: 是否与原始答案比较
            incremental_save: 是否增量保存
            output_file_path: 输出文件路径
            
        Returns:
            执行结果列表，包含mention_in节点信息
        """
        if incremental_save and not output_file_path:
            raise ValueError("启用增量保存时必须提供 output_file_path 参数")
        
        results = []
        file_handle = None
        jsonl_file = None
        
        if incremental_save:
            jsonl_file = output_file_path.replace('.json', '.jsonl') if output_file_path.endswith('.json') else output_file_path + '.jsonl'
            file_handle = open(jsonl_file, 'w', encoding='utf-8')
            print(f"启用增量保存，结果将实时写入: {jsonl_file}")
        
        try:
            for idx, query_data in enumerate(queries, 1):
                query_text = query_data.get('query', '')
                original_answer = query_data.get('answer', [])
                template_id = query_data.get('template_id', f'query_{idx}')
                is_noise_query = query_data.get('is_noise_query', False)
                
                print(f"执行查询 {idx}/{len(queries)}: {template_id}")
                
                try:
                    # 执行查询并提取mention_in节点
                    query_result = self.execute_query_with_mention_in(
                        query_text, 
                        query_data.get('parameters'),
                        query_data
                    )
                    
                    execution_result_raw = query_result['execution_result']
                    mention_in_nodes = query_result['mention_in_nodes']
                    
                    # 噪声查询与普通查询采用不同的结果结构
                    if is_noise_query:
                        from handler.cleaner import clean_normal_answer
                        cleaned_execution_result = clean_normal_answer(execution_result_raw)
                        
                        is_match = False
                        if compare_with_original:
                            is_match = cleaned_execution_result == original_answer
                        
                        result_item = {
                            'query': query_text,
                            'clean_answer': original_answer,
                            'noise_answer': cleaned_execution_result,
                            'same_as_cleangraph': is_match,
                            'mention_in_nodes': mention_in_nodes,
                            'error': None,
                        }
                    else:
                        execution_result = execution_result_raw
                        
                        is_match = False
                        if compare_with_original:
                            is_match = self._compare_results(execution_result, original_answer)
                        
                        result_item = {
                            'template_id': template_id,
                            'template_type': query_data.get('template_type', ''),
                            'query': query_text,
                            'parameters': query_data.get('parameters', {}),
                            'original_answer': original_answer,
                            'execution_result': execution_result,
                            'same_as_cleangraph': is_match,
                            'is_noise_query': is_noise_query,
                            'mention_in_nodes': mention_in_nodes,
                            'error': None,
                        }
                    
                    results.append(result_item)
                    
                    if incremental_save and file_handle:
                        json_line = json.dumps(result_item, ensure_ascii=False)
                        file_handle.write(json_line + '\n')
                        file_handle.flush()
                        print(f"  结果已保存到文件")
                    
                except Exception as e:
                    # 即使查询执行失败，也尝试提取mention_in关系
                    # 因为可以从parameters和查询语句中提取节点ID
                    mention_in_nodes = []
                    try:
                        mention_in_nodes = self.extract_mention_in_node_ids(
                            query_text, 
                            None,  # execution_result为None
                            query_data
                        )
                    except Exception as extract_error:
                        print(f"提取mention_in关系时出错: {extract_error}")
                    
                    if is_noise_query:
                        result_item = {
                            'query': query_text,
                            'clean_answer': original_answer,
                            'noise_answer': None,
                            'same_as_cleangraph': False,
                            'mention_in_nodes': mention_in_nodes,
                            'error': str(e),
                        }
                    else:
                        result_item = {
                            'template_id': template_id,
                            'template_type': query_data.get('template_type', ''),
                            'query': query_text,
                            'parameters': query_data.get('parameters', {}),
                            'original_answer': original_answer,
                            'execution_result': None,
                            'error': str(e),
                            'same_as_cleangraph': False,
                            'is_noise_query': is_noise_query,
                            'mention_in_nodes': mention_in_nodes,
                        }
                    results.append(result_item)
                    print(f"查询执行失败: {e}")
                    
                    if incremental_save and file_handle:
                        json_line = json.dumps(result_item, ensure_ascii=False)
                        file_handle.write(json_line + '\n')
                        file_handle.flush()
                        print(f"  错误结果已保存到文件")
        
        finally:
            if file_handle:
                file_handle.close()
                if incremental_save and jsonl_file:
                    json_file = jsonl_file.replace('.jsonl', '.json')
                    self._convert_jsonl_to_json(jsonl_file, json_file)
                    print(f"增量保存完成，JSON格式文件: {json_file}")
        
        return results
