import pandas as pd
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

# ============ 配置区域 ============
# 扰动记录文件路径
PERTURB_RECORD_PATH = "../../data_gen/perturbation_generator/perturb_record/semantic_SemanticPerturbationGenerator_train_20251226_033149.json"
# CSV 文件路径 (扰动后的数据集)
CSV_PATH = "../../data_gen/perturbed_dataset/PrimeKG_2512260331/kg_clean_train.csv"
# 查询模板文件路径
TEMPLATE_1PARAM_PATH = "../query_template/primekg_train.json"
TEMPLATE_2PARAM_PATH = "../query_template/primekg_2param.json"
# 输出路径
OUTPUT_PATH = "noise_queries_test.json"
# 批处理大小
BATCH_SIZE = 1000
# =================================


def load_perturb_records(path: str) -> Dict[str, Any]:
    """加载扰动记录文件"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_templates(template_1param_path: str, template_2param_path: str) -> List[Dict]:
    """加载查询模板"""
    templates = []
    with open(template_1param_path, 'r', encoding='utf-8') as f:
        templates.extend(json.load(f))
    # with open(template_2param_path, 'r', encoding='utf-8') as f:
    #     templates.extend(json.load(f))
    return templates


def read_specific_rows(csv_path: str, row_indices: List[int]) -> pd.DataFrame:
    """
    高效读取 CSV 文件中的指定行
    
    Args:
        csv_path: CSV 文件路径
        row_indices: 要读取的行索引列表（0-based，不包含表头）
    
    Returns:
        包含指定行的 DataFrame，行索引为原始索引
    """
    if not row_indices:
        return pd.DataFrame()
    
    header = pd.read_csv(csv_path, nrows=0)
    rows_to_read = set(row_indices)
    max_row = max(row_indices)
    
    result_rows = {}
    chunk_size = 10000
    current_row = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_end = current_row + len(chunk)
        
        # 找出在当前 chunk 中需要读取的行
        for target_idx in list(rows_to_read):
            if current_row <= target_idx < chunk_end:
                local_idx = target_idx - current_row
                result_rows[target_idx] = chunk.iloc[local_idx].to_dict()
                rows_to_read.discard(target_idx)
        
        current_row = chunk_end
        
        # 如果已经找到所有需要的行，提前退出
        if not rows_to_read or current_row > max_row:
            break
    
    if result_rows:
        df = pd.DataFrame.from_dict(result_rows, orient='index')
        df.index.name = 'row_index'
        return df
    else:
        return pd.DataFrame(columns=header.columns)


def normalize_type(node_type: str) -> str:
    """
    标准化节点类型名称，用于模板匹配
    
    将数据中的类型名转换为模板中使用的标签格式
    """
    type_mapping = {
        'protein': ['Protein', 'gene_protein', 'protein', 'gene/protein'],
        'drug': ['Drug', 'drug'],
        'disease': ['Disease', 'disease'],
        'anatomy': ['Anatomy', 'anatomy'],
        'BiologicalProcess': ['BiologicalProcess', 'biological_process'],
        'MolecularFunction': ['MolecularFunction', 'molecular_function'],
        'CellularComponent': ['CellularComponent', 'cellular_component'],
        'pathway': ['Pathway', 'pathway'],
        'phenotype': ['Phenotype', 'effect_phenotype', 'phenotype', 'effect/phenotype'],
        'exposure': ['Exposure', 'exposure'],
    }
    
    for key, values in type_mapping.items():
        if node_type.lower() in [v.lower() for v in values]:
            return key
    return node_type.lower()


def get_template_node_types(query_template: str) -> List[Tuple[str, str]]:
    """
    从查询模板中提取节点类型和对应的参数名
    
    返回: [(node_label, param_name), ...]
    """
    # 匹配模式: (x:Label {name: "{param}"}) 或 (x:Label {name: $param})
    patterns = [
        r'\((\w+):(\w+)\s*\{[^}]*name:\s*["\']?\{\s*(\w+)\s*\}["\']?[^}]*\}\)',
        r'\((\w+):(\w+)\s*\{[^}]*name:\s*\$(\w+)[^}]*\}\)',
    ]
    
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, query_template)
        for match in matches:
            # match: (variable, Label, param_name)
            results.append((match[1], match[2]))
    
    return results


def get_template_relations(query_template: str) -> List[Tuple[str, Optional[str]]]:
    """
    从查询模板中提取关系类型和label
    
    返回: [(relation_type, label), ...]
    """
    # 匹配模式: [:RELATION_TYPE] 或 [:relation_type {label: "xxx"}]
    pattern = r'\[:(\w+)(?:\s*\{[^}]*label:\s*["\']([^"\']+)["\'][^}]*\})?\]'
    matches = re.findall(pattern, query_template, re.IGNORECASE)
    return matches


def normalize_relation(relation: str) -> str:
    """标准化关系类型名称"""
    return relation.lower().replace(' ', '_')


def match_template_for_name_typo(
    templates: List[Dict],
    node_type: str,
    relation: str,
    display_relation: str
) -> List[Dict]:
    """
    为 name_typo 扰动类型匹配模板
    
    根据节点类型、关系和显示关系查找匹配的模板
    """
    matched = []
    normalized_type = normalize_type(node_type)
    normalized_relation = normalize_relation(relation)
    normalized_display_relation = normalize_relation(display_relation)
    
    for template in templates:
        query = template.get('query_template', '')
        params = template.get('query_parameters', {})
        
        # 检查模板是否包含对应的节点类型
        node_types = get_template_node_types(query)
        type_match = False
        matched_param = None
        
        for label, param_name in node_types:
            # 检查节点标签是否匹配
            if normalize_type(label) == normalized_type:
                type_match = True
                matched_param = param_name
                break
        
        if not type_match:
            continue
        
        # 检查关系类型是否匹配
        relations = get_template_relations(query)
        relation_match = False
        
        for rel_type, rel_label in relations:
            norm_rel_type = normalize_relation(rel_type)
            
            # 检查关系类型匹配
            if norm_rel_type == normalized_relation:
                # 如果模板有 label 要求，检查 display_relation 匹配
                if rel_label:
                    if normalize_relation(rel_label) == normalized_display_relation:
                        relation_match = True
                        break
                else:
                    relation_match = True
                    break
        
        if relation_match and matched_param:
            matched.append({
                'template': template,
                'param_name': matched_param
            })
    
    return matched


def match_template_for_false_edge(
    templates: List[Dict],
    node_type: str,
    relation: str,
    display_relation: str,
    replaced_field: str
) -> List[Dict]:
    """
    为 false_edge 扰动类型匹配模板
    
    根据节点类型、关系、显示关系和替换字段查找匹配的模板
    """
    matched = []
    normalized_type = normalize_type(node_type)
    normalized_relation = normalize_relation(relation)
    normalized_display_relation = normalize_relation(display_relation)
    
    # 根据 replaced_field 确定是 x 还是 y 的节点
    is_x_field = replaced_field.startswith('x_')
    
    for template in templates:
        query = template.get('query_template', '')
        params = template.get('query_parameters', {})
        
        # 提取模板中的节点类型和参数
        node_types = get_template_node_types(query)
        
        type_match = False
        matched_param = None
        
        for label, param_name in node_types:
            if normalize_type(label) == normalized_type:
                type_match = True
                matched_param = param_name
                break
        
        if not type_match:
            continue
        
        # 检查关系类型是否匹配
        relations = get_template_relations(query)
        relation_match = False
        
        for rel_type, rel_label in relations:
            norm_rel_type = normalize_relation(rel_type)
            
            if norm_rel_type == normalized_relation:
                if rel_label:
                    if normalize_relation(rel_label) == normalized_display_relation:
                        relation_match = True
                        break
                else:
                    relation_match = True
                    break
        
        if relation_match and matched_param:
            matched.append({
                'template': template,
                'param_name': matched_param,
                'use_x': is_x_field
            })
    
    return matched


def match_template_for_relation_type_noise(
    templates: List[Dict],
    original_relation: str,
    x_type: str,
    y_type: str,
    display_relation: str
) -> List[Dict]:
    """
    为 relation_type_noise 扰动类型匹配模板
    
    尝试使用 (original_relation, display_relation, x_type) 或 
    (original_relation, display_relation, y_type) 进行匹配
    """
    matched = []
    normalized_relation = normalize_relation(original_relation)
    normalized_display_relation = normalize_relation(display_relation)
    normalized_x_type = normalize_type(x_type)
    normalized_y_type = normalize_type(y_type)
    
    for template in templates:
        query = template.get('query_template', '')
        params = template.get('query_parameters', {})
        
        # 提取关系
        relations = get_template_relations(query)
        relation_match = False
        
        for rel_type, rel_label in relations:
            norm_rel_type = normalize_relation(rel_type)
            
            if norm_rel_type == normalized_relation:
                if rel_label:
                    if normalize_relation(rel_label) == normalized_display_relation:
                        relation_match = True
                        break
                else:
                    relation_match = True
                    break
        
        if not relation_match:
            continue
        
        # 提取节点类型
        node_types = get_template_node_types(query)
        
        # 尝试匹配 x_type
        for label, param_name in node_types:
            if normalize_type(label) == normalized_x_type:
                matched.append({
                    'template': template,
                    'param_name': param_name,
                    'use_field': 'x_name',
                    'match_type': 'x_type'
                })
                break
        
        # 尝试匹配 y_type
        for label, param_name in node_types:
            if normalize_type(label) == normalized_y_type:
                # 避免重复添加同一模板（如果 x_type == y_type）
                if not any(m['template'] == template and m['use_field'] == 'y_name' for m in matched):
                    matched.append({
                        'template': template,
                        'param_name': param_name,
                        'use_field': 'y_name',
                        'match_type': 'y_type'
                    })
                break
    
    return matched


def match_template_for_node_type_noise(
    templates: List[Dict],
    new_type: str,
    relation: str,
    display_relation: str,
    field: str
) -> List[Dict]:
    """
    为 node_type_noise 扰动类型匹配模板
    
    根据噪声节点类型、关系和显示关系查找匹配的模板
    """
    matched = []
    normalized_new_type = normalize_type(new_type)
    normalized_relation = normalize_relation(relation)
    normalized_display_relation = normalize_relation(display_relation)
    
    # 根据 field 确定使用 x 还是 y
    use_x = field == 'x_type'
    
    for template in templates:
        query = template.get('query_template', '')
        params = template.get('query_parameters', {})
        
        # 提取节点类型
        node_types = get_template_node_types(query)
        type_match = False
        matched_param = None
        
        for label, param_name in node_types:
            if normalize_type(label) == normalized_new_type:
                type_match = True
                matched_param = param_name
                break
        
        if not type_match:
            continue
        
        # 检查关系类型是否匹配
        relations = get_template_relations(query)
        relation_match = False
        
        for rel_type, rel_label in relations:
            norm_rel_type = normalize_relation(rel_type)
            
            if norm_rel_type == normalized_relation:
                if rel_label:
                    if normalize_relation(rel_label) == normalized_display_relation:
                        relation_match = True
                        break
                else:
                    relation_match = True
                    break
        
        if relation_match and matched_param:
            matched.append({
                'template': template,
                'param_name': matched_param,
                'use_x': use_x
            })
    
    return matched


def fill_template(template: Dict, param_name: str, value: str) -> str:
    """
    填充模板参数生成查询
    """
    query = template.get('query_template', '')
    
    # 支持两种参数格式: {param_name} 和 $param_name
    query = query.replace(f'{{{param_name}}}', value)
    query = query.replace(f'${param_name}', f'"{value}"')
    
    return query


def fill_nl_template(template: Dict, param_name: str, value: str) -> str:
    """
    填充自然语言模板参数
    
    Args:
        template: 查询模板字典
        param_name: 参数名称
        value: 参数值
    
    Returns:
        填充后的自然语言描述
    """
    nl = template.get('nl', '')
    
    if not nl:
        return ''
    
    # 替换 {param_name} 格式的占位符
    nl = nl.replace(f'{{{param_name}}}', value)
    
    return nl


def process_name_typo(
    record: Dict,
    row_data: Dict,
    templates: List[Dict]
) -> List[Dict]:
    """
    处理 name_typo 类型的扰动记录
    """
    results = []
    
    # 从新格式的记录中提取字段
    field = record['target']['location']['column_name']  # x_name 或 y_name
    original_name = record['change']['original_value']
    
    # 根据 field 确定类型字段
    type_field = 'x_type' if field == 'x_name' else 'y_type'
    node_type = row_data.get(type_field, '')
    relation = row_data.get('relation', '')
    display_relation = row_data.get('display_relation', '')
    
    # 匹配模板
    matched_templates = match_template_for_name_typo(
        templates, node_type, relation, display_relation
    )
    
    if not matched_templates:
        return [{
            'record': record,
            'status': 'no_match',
            'reason': f'No template found for type={node_type}, relation={relation}, display_relation={display_relation}'
        }]
    
    for match in matched_templates:
        template = match['template']
        param_name = match['param_name']
        
        # 使用原始名称填充模板
        query = fill_template(template, param_name, original_name)
        # 填充自然语言模板
        nl = fill_nl_template(template, param_name, original_name)
        
        results.append({
            'record': record,
            'template_id': template.get('template_id', template.get('query_id', 'unknown')),
            'query_type': template.get('query_type', ''),
            'generated_query': query,
            'generated_nl': nl,
            'node_type': node_type,
            'status': 'success'
        })
    
    return results


def process_false_edge(
    record: Dict,
    row_data: Dict,
    templates: List[Dict]
) -> List[Dict]:
    """
    处理 false_edge 类型的扰动记录
    """
    results = []
    
    # 从新格式的记录中提取字段
    replaced_field = record['target']['location']['column_name']  # x_id 或 y_id
    node_type = record['change']['node_type']
    
    relation = row_data.get('relation', '')
    display_relation = row_data.get('display_relation', '')
    
    # 直接从扰动记录中获取原始名称
    node_name = record['change'].get('original_name', '')
    
    # 匹配模板
    matched_templates = match_template_for_false_edge(
        templates, node_type, relation, display_relation, replaced_field
    )
    
    if not matched_templates:
        return [{
            'record': record,
            'status': 'no_match',
            'reason': f'No template found for type={node_type}, relation={relation}, display_relation={display_relation}, replaced_field={replaced_field}'
        }]
    
    for match in matched_templates:
        template = match['template']
        param_name = match['param_name']
        
        query = fill_template(template, param_name, node_name)
        # 填充自然语言模板
        nl = fill_nl_template(template, param_name, node_name)
        
        results.append({
            'record': record,
            'template_id': template.get('template_id', template.get('query_id', 'unknown')),
            'query_type': template.get('query_type', ''),
            'generated_query': query,
            'generated_nl': nl,
            'node_name': node_name,
            'node_type': node_type,
            'replaced_field': replaced_field,
            'status': 'success'
        })
    
    return results


def process_relation_type_noise(
    record: Dict,
    row_data: Dict,
    templates: List[Dict]
) -> List[Dict]:
    """
    处理 relation_type_noise 类型的扰动记录
    """
    results = []
    
    # 从新格式的记录中提取字段
    original_relation = record['change']['original_value']
    target_field = record['target']['location']['column_name']
    
    x_type = row_data.get('x_type', '')
    y_type = row_data.get('y_type', '')
    x_name = row_data.get('x_name', '')
    y_name = row_data.get('y_name', '')
    display_relation = row_data.get('display_relation', '')
    
    # 如果修改的是 display_relation，使用原始关系
    if target_field == 'display_relation':
        relation_for_match = row_data.get('relation', original_relation)
    else:
        relation_for_match = original_relation
    
    # 匹配模板
    matched_templates = match_template_for_relation_type_noise(
        templates, relation_for_match, x_type, y_type, display_relation
    )
    
    if not matched_templates:
        return [{
            'record': record,
            'status': 'no_match',
            'reason': f'No template found for original_relation={original_relation}, x_type={x_type}, y_type={y_type}, display_relation={display_relation}'
        }]
    
    for match in matched_templates:
        template = match['template']
        param_name = match['param_name']
        use_field = match['use_field']
        
        # 根据匹配类型选择使用 x_name 或 y_name
        name_value = x_name if use_field == 'x_name' else y_name
        
        query = fill_template(template, param_name, name_value)
        # 填充自然语言模板
        nl = fill_nl_template(template, param_name, name_value)
        
        results.append({
            'record': record,
            'template_id': template.get('template_id', template.get('query_id', 'unknown')),
            'query_type': template.get('query_type', ''),
            'generated_query': query,
            'generated_nl': nl,
            'used_field': use_field,
            'name_value': name_value,
            'match_type': match['match_type'],
            'status': 'success'
        })
    
    return results


def process_node_type_noise(
    record: Dict,
    row_data: Dict,
    templates: List[Dict]
) -> List[Dict]:
    """
    处理 node_type_noise 类型的扰动记录
    """
    results = []
    
    # 从新格式的记录中提取字段
    field = record['target']['location']['column_name']  # x_type 或 y_type
    new_type = record['change']['new_value']
    original_type = record['change']['original_value']
    
    relation = row_data.get('relation', '')
    display_relation = row_data.get('display_relation', '')
    
    # 根据 field 获取对应的名称
    if field == 'x_type':
        name_field = 'x_name'
    else:
        name_field = 'y_name'
    
    node_name = row_data.get(name_field, '')
    
    # 匹配模板 - 使用噪声类型
    matched_templates = match_template_for_node_type_noise(
        templates, new_type, relation, display_relation, field
    )
    
    if not matched_templates:
        return [{
            'record': record,
            'status': 'no_match',
            'reason': f'No template found for new_type={new_type}, relation={relation}, display_relation={display_relation}, field={field}'
        }]
    
    for match in matched_templates:
        template = match['template']
        param_name = match['param_name']
        
        query = fill_template(template, param_name, node_name)
        # 填充自然语言模板
        nl = fill_nl_template(template, param_name, node_name)
        
        results.append({
            'record': record,
            'template_id': template.get('template_id', template.get('query_id', 'unknown')),
            'query_type': template.get('query_type', ''),
            'generated_query': query,
            'generated_nl': nl,
            'node_name': node_name,
            'original_type': original_type,
            'new_type': new_type,
            'field': field,
            'status': 'success'
        })
    
    return results


def generate_noise_queries(
    perturb_record_path: str = PERTURB_RECORD_PATH,
    csv_path: str = CSV_PATH,
    template_1param_path: str = TEMPLATE_1PARAM_PATH,
    template_2param_path: str = TEMPLATE_2PARAM_PATH,
    output_path: str = OUTPUT_PATH,
    batch_size: int = BATCH_SIZE
) -> Dict[str, Any]:
    """
    主函数：生成噪声查询
    
    Args:
        perturb_record_path: 扰动记录文件路径
        csv_path: CSV 数据文件路径
        template_1param_path: 单参数模板路径
        template_2param_path: 双参数模板路径
        output_path: 输出文件路径
        batch_size: 批处理大小
    
    Returns:
        生成结果统计
    """
    print("Loading perturbation records...")
    perturb_data = load_perturb_records(perturb_record_path)
    operations = perturb_data.get('operations', [])
    
    print(f"Total perturbation records: {len(operations)}")
    
    print("Loading templates...")
    templates = load_templates(template_1param_path, template_2param_path)
    print(f"Total templates loaded: {len(templates)}")
    
    # 统计各类型扰动数量
    type_counts = defaultdict(int)
    for op in operations:
        type_counts[op.get('meta', {}).get('operation_type', 'unknown')] += 1
    
    print("Perturbation types distribution:")
    for op_type, count in type_counts.items():
        print(f"  {op_type}: {count}")
    
    # 按批次处理
    all_results = []
    success_count = 0
    no_match_count = 0
    error_count = 0
    
    # 按 row_index 分组，提高读取效率
    operations_by_row = defaultdict(list)
    for i, op in enumerate(operations):
        row_index = op.get('target', {}).get('location', {}).get('row_index')
        if row_index is not None:
            operations_by_row[row_index].append((i, op))
    
    print(f"Unique row indices to read: {len(operations_by_row)}")
    
    # 分批读取数据
    all_row_indices = list(operations_by_row.keys())
    total_batches = (len(all_row_indices) + batch_size - 1) // batch_size
    
    print(f"Processing {total_batches} batches...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_row_indices))
        batch_row_indices = all_row_indices[start_idx:end_idx]
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        
        # 批量读取数据行
        try:
            df = read_specific_rows(csv_path, batch_row_indices)
        except Exception as e:
            print(f"Error reading batch {batch_idx + 1}: {e}")
            for row_idx in batch_row_indices:
                for _, op in operations_by_row[row_idx]:
                    all_results.append({
                        'record': op,
                        'status': 'error',
                        'reason': f'Failed to read row: {str(e)}'
                    })
                    error_count += 1
            continue
        
        # 处理每个扰动记录
        for row_idx in batch_row_indices:
            if row_idx not in df.index:
                for _, op in operations_by_row[row_idx]:
                    all_results.append({
                        'record': op,
                        'status': 'error',
                        'reason': f'Row {row_idx} not found in CSV'
                    })
                    error_count += 1
                continue
            
            row_data = df.loc[row_idx].to_dict()
            
            for _, op in operations_by_row[row_idx]:
                operation_type = op.get('meta', {}).get('operation_type', '').strip()
                
                try:
                    if operation_type == 'name_typo':
                        results = process_name_typo(op, row_data, templates)
                    elif operation_type == 'false_edge':
                        results = process_false_edge(op, row_data, templates)
                    elif operation_type == 'relation_type_noise' or operation_type == 'relation _type_noise':
                        results = process_relation_type_noise(op, row_data, templates)
                    elif operation_type == 'node_type_noise':
                        results = process_node_type_noise(op, row_data, templates)
                    else:
                        results = [{
                            'record': op,
                            'status': 'error',
                            'reason': f'Unknown operation type: {operation_type}'
                        }]
                    
                    for r in results:
                        all_results.append(r)
                        if r.get('status') == 'success':
                            success_count += 1
                        elif r.get('status') == 'no_match':
                            no_match_count += 1
                        else:
                            error_count += 1
                            
                except Exception as e:
                    all_results.append({
                        'record': op,
                        'status': 'error',
                        'reason': f'Processing error: {str(e)}'
                    })
                    error_count += 1
    
    # 准备输出
    output_data = {
        'metadata': {
            'source_perturb_record': perturb_record_path,
            'source_csv': csv_path,
            'templates_used': [template_1param_path],
            'total_perturb_records': len(operations),
            'total_generated': len(all_results),
            'success_count': success_count,
            'no_match_count': no_match_count,
            'error_count': error_count
        },
        'queries': [r for r in all_results if r.get('status') == 'success'],
        'unmatched': [r for r in all_results if r.get('status') == 'no_match'],
        'errors': [r for r in all_results if r.get('status') == 'error']
    }
    
    # 保存结果
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print("\n=== Generation Summary ===")
    print(f"Total perturbation records: {len(operations)}")
    print(f"Total generated queries: {len(all_results)}")
    print(f"  - Success: {success_count}")
    print(f"  - No match: {no_match_count}")
    print(f"  - Errors: {error_count}")
    
    # 按扰动类型统计成功率
    type_stats = defaultdict(lambda: {'success': 0, 'no_match': 0, 'error': 0})
    for r in all_results:
        op_type = r.get('record', {}).get('meta', {}).get('operation_type', 'unknown')
        status = r.get('status', 'error')
        type_stats[op_type][status] += 1
    
    print("\nSuccess rate by perturbation type:")
    for op_type, stats in type_stats.items():
        total = sum(stats.values())
        success_rate = stats['success'] / total * 100 if total > 0 else 0
        print(f"  {op_type}: {stats['success']}/{total} ({success_rate:.1f}%)")
    
    return output_data


if __name__ == "__main__":
    import os
    
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    generate_noise_queries()
