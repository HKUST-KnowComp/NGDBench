import pickle
import random
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from itertools import combinations

def load_graph(pickle_path: str) -> nx.MultiDiGraph:
    """从pickle文件加载图"""
    with open(pickle_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

class QueryGenerator:
    """基于图结构的查询生成器"""
    
    def __init__(self, graph: nx.MultiDiGraph, random_seed: int = 42):
        """
        Args:
            graph: nx.MultiDiGraph 图数据
            random_seed: 随机种子
        """
        self.graph = graph
        random.seed(random_seed)
        
        # 构建索引以加速查询
        self._build_indices()
    
    def _build_indices(self):
        """构建节点类型和边类型的索引"""
        # 节点类型 -> 节点列表
        self.nodes_by_type: Dict[str, List] = defaultdict(list)
        # 节点 -> 节点类型
        self.node_type_map: Dict[Any, str] = {}
        # 边类型 -> [(src, dst, key), ...]
        self.edges_by_type: Dict[str, List[Tuple]] = defaultdict(list)
        # (节点类型, 边类型) -> 相邻节点类型集合
        self.neighbor_types: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        
        print("正在构建节点索引...")
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type') or attrs.get('label') or attrs.get('node_type', 'unknown')
            self.nodes_by_type[node_type].append(node)
            self.node_type_map[node] = node_type
        
        print("正在构建边索引...")
        for src, dst, key, attrs in self.graph.edges(data=True, keys=True):
            edge_type = attrs.get('type') or attrs.get('label') or attrs.get('edge_type', 'unknown')
            self.edges_by_type[edge_type].append((src, dst, key))
            
            src_type = self.node_type_map.get(src, 'unknown')
            dst_type = self.node_type_map.get(dst, 'unknown')
            self.neighbor_types[(src_type, edge_type)].add(dst_type)
            self.neighbor_types[(dst_type, edge_type)].add(src_type)  # 无向处理
        
        print(f"索引完成: {len(self.nodes_by_type)} 种节点类型, {len(self.edges_by_type)} 种边类型")
    
    def get_node_name(self, node) -> str:
        """获取节点名称"""
        attrs = self.graph.nodes[node]
        return attrs.get('name') or attrs.get('label') or str(node)
    
    def get_node_type(self, node) -> str:
        """获取节点类型"""
        return self.node_type_map.get(node, 'unknown')
    
    def get_neighbors_by_edge_type(
        self, 
        node, 
        edge_type: str, 
        target_node_type: Optional[str] = None
    ) -> List[Tuple[Any, str]]:
        """
        获取通过指定边类型连接的邻居节点
        
        Args:
            node: 起始节点
            edge_type: 边类型
            target_node_type: 目标节点类型（可选）
        
        Returns:
            [(邻居节点, 边方向), ...] 边方向为 'out' 或 'in'
        """
        neighbors = []
        
        # 出边
        if self.graph.has_node(node):
            for _, dst, _, attrs in self.graph.out_edges(node, data=True, keys=True):
                e_type = attrs.get('type') or attrs.get('label') or attrs.get('edge_type', 'unknown')
                if e_type == edge_type:
                    if target_node_type is None or self.get_node_type(dst) == target_node_type:
                        neighbors.append((dst, 'out'))
            
            # 入边（因为方向性弱，也考虑反向）
            for src, _, _, attrs in self.graph.in_edges(node, data=True, keys=True):
                e_type = attrs.get('type') or attrs.get('label') or attrs.get('edge_type', 'unknown')
                if e_type == edge_type:
                    if target_node_type is None or self.get_node_type(src) == target_node_type:
                        neighbors.append((src, 'in'))
        
        return neighbors
    
    def find_valid_anchor_for_triangle(
        self,
        anchor_type: str,
        entity_type: str,
        attr_type: str,
        relation1: str,
        relation2: str,
        min_entities: int = 2,
        min_attrs_per_entity: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        为三角/环路查询找到有效的锚点和对应的实体、属性
        
        模式: anchor -[relation1]- entity -[relation2]- attr
        要求: 至少有 min_entities 个不同的 entity，每个 entity 至少有 min_attrs_per_entity 个 attr
        
        Returns:
            {
                'anchor': node,
                'anchor_name': str,
                'entities': [(entity_node, [attr_nodes]), ...],
            } 或 None
        """
        candidates = self.nodes_by_type.get(anchor_type, [])
        if not candidates:
            return None
        
        # 随机打乱候选锚点
        shuffled_candidates = candidates.copy()
        random.shuffle(shuffled_candidates)
        
        for anchor in shuffled_candidates:
            # 找到与 anchor 通过 relation1 相连的 entity_type 节点
            entity_neighbors = self.get_neighbors_by_edge_type(anchor, relation1, entity_type)
            
            if len(entity_neighbors) < min_entities:
                continue
            
            # 对每个 entity，找到其通过 relation2 相连的 attr_type 节点
            valid_entities = []
            for entity, _ in entity_neighbors:
                attr_neighbors = self.get_neighbors_by_edge_type(entity, relation2, attr_type)
                if len(attr_neighbors) >= min_attrs_per_entity:
                    attrs = [attr for attr, _ in attr_neighbors]
                    valid_entities.append((entity, attrs))
            
            if len(valid_entities) >= min_entities:
                return {
                    'anchor': anchor,
                    'anchor_name': self.get_node_name(anchor),
                    'entities': valid_entities
                }
        
        return None
    
    def generate_query_A21(
        self,
        template: Dict[str, Any],
        max_attempts: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        生成 A21 类型查询（差异化配对查询 - 三角/环路）
        
        模式: anchor -[R1]- entity -[R2]- attr
        """
        params = template['parameters']
        
        # 获取可用的类型组合
        available_node_types = list(self.nodes_by_type.keys())
        available_edge_types = list(self.edges_by_type.keys())
        
        for attempt in range(max_attempts):
            # 随机选择类型
            if len(available_node_types) < 3 or len(available_edge_types) < 2:
                print("节点或边类型不足")
                return None
            
            anchor_type = random.choice(available_node_types)
            entity_type = random.choice([t for t in available_node_types if t != anchor_type])
            attr_type = random.choice([t for t in available_node_types if t != entity_type])
            relation1 = random.choice(available_edge_types)
            relation2 = random.choice(available_edge_types)
            
            # 尝试找到符合条件的锚点
            result = self.find_valid_anchor_for_triangle(
                anchor_type=anchor_type,
                entity_type=entity_type,
                attr_type=attr_type,
                relation1=relation1,
                relation2=relation2,
                min_entities=2,
                min_attrs_per_entity=1
            )
            
            if result:
                # 生成查询答案
                answer = self._compute_answer_A21(result)
                
                # 填充参数
                filled_params = {
                    'AnchorType': anchor_type,
                    'EntityType': entity_type,
                    'AttrType': attr_type,
                    'Relation1': relation1,
                    'Relation2': relation2,
                    'anchor_name': result['anchor_name']
                }
                
                # 填充模版
                filled_template = template['abstract_template']
                for key, value in filled_params.items():
                    if key == 'anchor_name':
                        continue  # runtime parameter，保持 $anchor_name
                    filled_template = filled_template.replace(f'{{{key}}}', value)
                
                return {
                    'template_id': template['abstract_template_id'],
                    'filled_template': filled_template,
                    'parameters': filled_params,
                    'answer': answer,
                    'subgraph_info': {
                        'anchor': result['anchor'],
                        'anchor_name': result['anchor_name'],
                        'entities_count': len(result['entities'])
                    }
                }
        
        return None
    
    def _compute_answer_A21(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        计算 A21 查询的答案
        
        找到所有 (entity1, entity2) 配对，其中 entity1 <> entity2 且它们的 attr 不同
        """
        entities = result['entities']
        answers = []
        
        # 对所有实体对进行组合
        for (entity1, attrs1), (entity2, attrs2) in combinations(entities, 2):
            # 检查属性是否不同（至少有一个不同）
            attrs1_set = set(attrs1)
            attrs2_set = set(attrs2)
            
            if attrs1_set != attrs2_set:  # 属性集合不完全相同
                answers.append({
                    'Entity1': self.get_node_name(entity1),
                    'Entity2': self.get_node_name(entity2),
                    'attr_count1': len(attrs1_set),
                    'attr_count2': len(attrs2_set)
                })
        
        return answers


class TemplateQueryGenerator:
    """模版化查询生成器"""
    
    def __init__(self, graph: nx.MultiDiGraph, templates: List[Dict[str, Any]], random_seed: int = 42):
        """
        Args:
            graph: 图数据
            templates: 查询模版列表
            random_seed: 随机种子
        """
        self.query_gen = QueryGenerator(graph, random_seed)
        self.templates = {t['abstract_template_id']: t for t in templates}
        
        # 注册不同模版类型的生成函数
        self.generators = {
            'A21': self.query_gen.generate_query_A21,
            # 在这里添加其他模版类型的生成函数
            # 'A22': self.query_gen.generate_query_A22,
        }
    
    def generate_query(
        self,
        template_id: str,
        max_attempts: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        根据模版ID生成查询
        
        Args:
            template_id: 模版ID
            max_attempts: 最大尝试次数
        
        Returns:
            生成的查询和答案
        """
        if template_id not in self.templates:
            raise ValueError(f"未知模版ID: {template_id}")
        
        template = self.templates[template_id]
        
        if template_id in self.generators:
            return self.generators[template_id](template, max_attempts)
        else:
            print(f"模版 {template_id} 暂未实现生成函数")
            return None
    
    def generate_queries_batch(
        self,
        template_id: str,
        num_queries: int,
        max_attempts_per_query: int = 100
    ) -> List[Dict[str, Any]]:
        """
        批量生成查询
        
        Args:
            template_id: 模版ID
            num_queries: 需要生成的查询数量
            max_attempts_per_query: 每个查询的最大尝试次数
        
        Returns:
            生成的查询列表
        """
        queries = []
        for i in range(num_queries):
            query = self.generate_query(template_id, max_attempts_per_query)
            if query:
                queries.append(query)
                print(f"生成查询 {i+1}/{num_queries}")
            else:
                print(f"查询 {i+1} 生成失败")
        
        return queries


def save_queries(queries: List[Dict[str, Any]], output_path: str):
    """保存生成的查询到pickle文件"""
    with open(output_path, 'wb') as f:
        pickle.dump(queries, f)
    print(f"已保存 {len(queries)} 条查询到 {output_path}")


def main():
    # 示例模版
    templates = [
        {
            "abstract_template_id": "A21",
            "category": "C10",
            "category_name": "三角/环路查询",
            "pattern_name": "差异化配对查询",
            "description": "查找同时与某实体相关但彼此不同的实体对",
            "abstract_template": "MATCH (anchor:{AnchorType} {name: $anchor_name})-[:{Relation1}]-(entity1:{EntityType})-[:{Relation2}]-(attr1:{AttrType}) MATCH (anchor)-[:{Relation1}]-(entity2:{EntityType})-[:{Relation2}]-(attr2:{AttrType}) WHERE entity1 <> entity2 AND attr1 <> attr2 RETURN entity1.name AS Entity1, entity2.name AS Entity2, COUNT(DISTINCT attr1) AS attr_count1, COUNT(DISTINCT attr2) AS attr_count2",
            "parameters": {
                "AnchorType": {"description": "锚点节点类型"},
                "EntityType": {"description": "实体类型"},
                "AttrType": {"description": "属性节点类型"},
                "Relation1": {"description": "锚点到实体的关系"},
                "Relation2": {"description": "实体到属性的关系"},
                "anchor_name": {"description": "锚点名称", "type": "runtime_parameter"}
            },
            "original_templates": ["T26"],
            "use_cases": ["互补分析", "差异化发现"]
        }
    ]
    
    # 加载图
    INPUT_GRAPH_PATH = "your_graph.pickle"
    OUTPUT_QUERIES_PATH = "generated_queries.pickle"
    
    print("加载图...")
    graph = load_graph(INPUT_GRAPH_PATH)
    print(f"图统计: 节点数 = {graph.number_of_nodes():,}, 边数 = {graph.number_of_edges():,}")
    
    # 初始化生成器
    generator = TemplateQueryGenerator(graph, templates, random_seed=42)
    
    # 生成查询
    print("\n开始生成查询...")
    queries = generator.generate_queries_batch(
        template_id="A21",
        num_queries=10,
        max_attempts_per_query=100
    )
    
    # 打印示例
    if queries:
        print("\n示例查询:")
        q = queries[0]
        print(f"模版ID: {q['template_id']}")
        print(f"填充后的查询: {q['filled_template']}")
        print(f"参数: {q['parameters']}")
        print(f"答案数量: {len(q['answer'])}")
        if q['answer']:
            print(f"答案示例: {q['answer'][0]}")
    
    # 保存
    save_queries(queries, OUTPUT_QUERIES_PATH)


if __name__ == "__main__":
    main()