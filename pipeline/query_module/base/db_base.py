"""
数据库连接和查询执行基类
"""
import json
import os
import sys
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

# 确保可以导入清洗函数 cleaner.clean_normal_answer
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 从 query_module/base/ 回到 pipeline/ 目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from handler.cleaner import clean_normal_answer


def _convert_neo4j_to_dict(value: Any, node_id_key: str = "_node_id") -> Any:
    """
    将 Neo4j 对象（Node、Relationship、Path 等）转换为可序列化的字典
    
    Args:
        value: 可能是 Neo4j 对象或普通 Python 对象
        node_id_key: Node 对象的 ID 属性名，默认为 "_node_id"
        
    Returns:
        转换后的可序列化对象。对于 Node 对象，只返回指定 ID 属性的值
    """
    # 导入 Neo4j 类型用于类型检查
    try:
        from neo4j.graph import Node, Relationship, Path
    except ImportError:
        # 如果无法导入，使用属性检查
        Node = None
        Relationship = None
        Path = None
    
    # 检查是否是 Neo4j Node 对象
    if Node is not None and isinstance(value, Node):
        # 这是一个 Node 对象，只返回指定 ID 属性的值
        # 首先检查节点是否有指定的 ID 属性
        try:
            if node_id_key in value:
                # 节点有指定的 ID 属性，返回该属性的值
                return value[node_id_key]
        except (KeyError, TypeError):
            pass
        
        # 如果节点没有指定的 ID 属性，返回 element_id 或 id
        if hasattr(value, 'element_id'):
            return value.element_id
        elif hasattr(value, 'id'):
            return value.id
        else:
            return None
    elif Node is None and hasattr(value, 'id') and hasattr(value, 'labels') and hasattr(value, 'keys'):
        # 备用检测方法：通过属性判断
        try:
            # 尝试访问 labels 属性来确认是 Node
            _ = value.labels
            # 首先检查节点是否有指定的 ID 属性
            try:
                if node_id_key in value:
                    # 节点有指定的 ID 属性，返回该属性的值
                    return value[node_id_key]
            except (KeyError, TypeError):
                pass
            
            # 如果节点没有指定的 ID 属性，返回 element_id 或 id
            if hasattr(value, 'element_id'):
                return value.element_id
            elif hasattr(value, 'id'):
                return value.id
            else:
                return None
        except (AttributeError, TypeError):
            pass
    
    # 检查是否是 Neo4j Relationship 对象
    if Relationship is not None and isinstance(value, Relationship):
        # 这是一个 Relationship 对象
        rel_dict = {}
        # 添加所有属性
        for key in value.keys():
            rel_dict[key] = _convert_neo4j_to_dict(value[key], node_id_key)
        # 添加关系类型
        rel_dict['_type'] = value.type
        # 添加 _edge_id 字段
        if hasattr(value, 'element_id'):
            rel_dict['_edge_id'] = value.element_id
        elif hasattr(value, 'id'):
            rel_dict['_edge_id'] = value.id
        return rel_dict
    elif Relationship is None and hasattr(value, 'id') and hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
        # 备用检测方法：通过属性判断
        try:
            # 尝试访问 type 属性来确认是 Relationship
            _ = value.type
            rel_dict = {}
            for key in value.keys():
                rel_dict[key] = _convert_neo4j_to_dict(value[key], node_id_key)
            rel_dict['_type'] = value.type
            if hasattr(value, 'element_id'):
                rel_dict['_edge_id'] = value.element_id
            elif hasattr(value, 'id'):
                rel_dict['_edge_id'] = value.id
            return rel_dict
        except (AttributeError, TypeError):
            pass
    
    # 检查是否是 Neo4j Path 对象
    if Path is not None and isinstance(value, Path):
        # 这是一个 Path 对象
        return {
            'nodes': [_convert_neo4j_to_dict(node, node_id_key) for node in value.nodes],
            'relationships': [_convert_neo4j_to_dict(rel, node_id_key) for rel in value.relationships]
        }
    elif Path is None and hasattr(value, 'nodes') and hasattr(value, 'relationships'):
        # 备用检测方法
        try:
            return {
                'nodes': [_convert_neo4j_to_dict(node, node_id_key) for node in value.nodes],
                'relationships': [_convert_neo4j_to_dict(rel, node_id_key) for rel in value.relationships]
            }
        except (AttributeError, TypeError):
            pass
    
    # 如果是列表，递归处理每个元素
    if isinstance(value, list):
        return [_convert_neo4j_to_dict(item, node_id_key) for item in value]
    
    # 如果是字典，递归处理每个值
    if isinstance(value, dict):
        return {k: _convert_neo4j_to_dict(v, node_id_key) for k, v in value.items()}
    
    # 其他类型（str, int, float, bool, None）直接返回
    return value


class DatabaseExecutor:
    """数据库执行器基类，支持连接数据库、读取查询、执行查询"""
    
    def __init__(self, uri: str, user: str, password: str, node_id_key: str = "_node_id"):
        """
        初始化数据库连接
        
        Args:
            uri: 数据库连接URI，例如 "bolt://localhost:7693"
            user: 用户名
            password: 密码
            node_id_key: Node 对象的 ID 属性名，默认为 "_node_id"
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.node_id_key = node_id_key
        self.driver = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 验证连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"成功连接到数据库: {self.uri}")
        except Exception as e:
            print(f"连接数据库失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("数据库连接已关闭")
    
    def read_queries_from_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        从JSON文件中读取查询
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            查询列表，每个查询是一个字典
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            print(f"从 {json_file_path} 读取了 {len(queries)} 个查询")
            return queries
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            raise
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, node_id_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        执行单个查询
        
        Args:
            query: Cypher查询语句
            parameters: 查询参数（可选）
            node_id_key: Node 对象的 ID 属性名，如果为 None 则使用初始化时设置的值，默认为 "_node_id"
            
        Returns:
            查询结果列表，每个结果是一个字典。Node 对象会被转换为指定 ID 属性的值
        """
        if not self.driver:
            raise RuntimeError("数据库未连接，请先调用 connect() 方法")
        
        # 使用传入的 node_id_key，如果没有则使用实例的默认值
        id_key = node_id_key if node_id_key is not None else self.node_id_key
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                # 将结果转换为字典列表，并转换 Neo4j 对象为可序列化格式
                records = []
                for record in result:
                    # 先转换为字典
                    record_dict = dict(record)
                    # 递归转换所有 Neo4j 对象
                    converted_dict = {}
                    for key, value in record_dict.items():
                        converted_dict[key] = _convert_neo4j_to_dict(value, node_id_key=id_key)
                    records.append(converted_dict)
                return records
        except Exception as e:
            print(f"执行查询失败: {e}")
            print(f"查询语句: {query}")
            raise
    
    def execute_queries_batch(self, queries: List[Dict[str, Any]], 
                             compare_with_original: bool = True,
                             incremental_save: bool = False,
                             output_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量执行查询
        
        Args:
            queries: 查询列表，每个查询包含 query, answer 等字段
            compare_with_original: 是否与原始答案比较
            incremental_save: 是否增量保存（每执行完一个查询就立即写入文件）
            output_file_path: 增量保存时的输出文件路径（如果为None且incremental_save为True，会抛出错误）
            
        Returns:
            执行结果列表，每个结果包含查询信息、执行结果和比较结果
        """ 
        if incremental_save and not output_file_path:
            raise ValueError("启用增量保存时必须提供 output_file_path 参数")
        
        results = []
        file_handle = None
        jsonl_file = None
        
        # 如果启用增量保存，初始化文件
        if incremental_save:
            # 使用 JSON Lines 格式（.jsonl）便于追加写入
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
                    # 执行查询
                    execution_result_raw = self.execute_query(query_text)
                    
                    # 噪声查询与普通查询采用不同的结果结构和比较逻辑
                    if is_noise_query:
                        # 对执行结果做与 original_answer 相同的清洗，再比较
                        # 使用 clean_normal_answer 清洗，它会处理各种格式：
                        # - [{"a": {"_node_id": "Company:123"}}] -> ["Company:123"]
                        # - [{"a": "Company:123"}] -> ["Company:123"]
                        # - 其他格式也会被正确处理
                        cleaned_execution_result = clean_normal_answer(execution_result_raw, node_id_key=self.node_id_key)
                        
                        # 同时清洗 original_answer，确保格式一致
                        # 因为 original_answer 可能来自输入文件，可能没有被正确清洗
                        # 统一使用 clean_normal_answer 清洗，它会自动处理各种格式
                        if isinstance(original_answer, list):
                            # 如果 original_answer 是列表，检查是否需要清洗
                            # 如果第一个元素是字典，说明需要清洗
                            if len(original_answer) > 0 and isinstance(original_answer[0], dict):
                                cleaned_original_answer = clean_normal_answer(original_answer, node_id_key=self.node_id_key)
                            else:
                                # 已经是清洗后的格式，直接使用
                                cleaned_original_answer = original_answer
                        else:
                            # 如果不是列表，可能是单个值，直接使用
                            cleaned_original_answer = original_answer
                        
                        is_match = False
                        if compare_with_original:
                            # 两个答案都应该被清洗成相同格式后比较
                            # 直接比较清洗后的结果
                            is_match = cleaned_execution_result == cleaned_original_answer
                        
                        # 噪声查询的输出格式与
                        # noise_query_execution_results_cleaned_from_step1.json 保持一致
                        # 注意：这里保存清洗后的 clean_answer，确保格式一致
                        result_item = {
                            'query': query_text,
                            'clean_answer': cleaned_original_answer,  # 保存清洗后的 clean_answer
                            'noise_answer': cleaned_execution_result,
                            'same_as_cleangraph': is_match,
                            'error': None,
                        }
                    else:
                        # 普通查询保持原有逻辑和输出格式
                        execution_result = execution_result_raw
                        
                        # 比较结果
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
                            'error': None,
                        }
                    
                    results.append(result_item)
                    
                    # 增量保存：立即写入文件
                    if incremental_save and file_handle:
                        json_line = json.dumps(result_item, ensure_ascii=False)
                        file_handle.write(json_line + '\n')
                        file_handle.flush()  # 立即刷新到磁盘
                        print(f"  结果已保存到文件")
                    
                except Exception as e:
                    # 记录执行失败的查询
                    if is_noise_query:
                        # 噪声查询失败时也保持与成功时类似的结构
                        result_item = {
                            'query': query_text,
                            'clean_answer': original_answer,
                            'noise_answer': None,
                            'same_as_cleangraph': False,
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
                        }
                    results.append(result_item)
                    print(f"查询执行失败: {e}")
                    
                    # 增量保存：即使失败也写入文件
                    if incremental_save and file_handle:
                        json_line = json.dumps(result_item, ensure_ascii=False)
                        file_handle.write(json_line + '\n')
                        file_handle.flush()
                        print(f"  错误结果已保存到文件")
        
        finally:
            # 关闭文件句柄
            if file_handle:
                file_handle.close()
                if incremental_save and jsonl_file:
                    # 如果用户希望最终输出为 JSON 格式，可以转换
                    json_file = jsonl_file.replace('.jsonl', '.json')
                    self._convert_jsonl_to_json(jsonl_file, json_file)
                    print(f"增量保存完成，JSON格式文件: {json_file}")
        
        return results
    
    def _compare_results(self, result1: List[Dict[str, Any]], 
                        result2: List[Dict[str, Any]]) -> bool:
        """
        比较两个查询结果是否相同
        
        Args:
            result1: 第一个结果列表
            result2: 第二个结果列表
            
        Returns:
            是否匹配
        """
        # 如果长度不同，肯定不匹配
        if len(result1) != len(result2):
            return False
        
        # 如果都为空，认为匹配
        if len(result1) == 0 and len(result2) == 0:
            return True
        
        # 对结果进行排序以便比较（基于所有键的值的组合）
        def sort_key(d):
            return tuple(sorted(d.items()))
        
        sorted_result1 = sorted(result1, key=sort_key)
        sorted_result2 = sorted(result2, key=sort_key)
        
        # 逐个比较
        for r1, r2 in zip(sorted_result1, sorted_result2):
            if r1 != r2:
                return False
        
        return True
    
    def save_results_to_json(self, results: List[Dict[str, Any]], output_file_path: str):
        """
        将执行结果保存到JSON文件
        
        Args:
            results: 执行结果列表
            output_file_path: 输出文件路径
        """
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output_file_path}")
            
            # 统计信息
            total = len(results)
            # 统计答案不匹配的数量（same_as_cleangraph 为 False 且没有执行错误）
            mismatched = sum(1 for r in results if r.get('same_as_cleangraph', False) is False and r.get('error') is None)
            failed = sum(1 for r in results if r.get('error') is not None)
            
            print(f"执行统计: 总计 {total} 个查询, 答案不一样 {mismatched} 个, 失败 {failed} 个")
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            raise
    
    def _convert_jsonl_to_json(self, jsonl_file: str, json_file: str):
        """
        将 JSON Lines 格式文件转换为 JSON 数组格式
        
        Args:
            jsonl_file: JSON Lines 文件路径
            json_file: 输出的 JSON 文件路径
        """
        try:
            results = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"已转换 JSON Lines 为 JSON 格式: {json_file}")
        except Exception as e:
            print(f"转换文件格式失败: {e}")
            raise
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
