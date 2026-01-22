import random
import json

# ==========================================
# 1. 模拟数据采样层 (Mock Data Sampler)
# 在实际应用中，这里应该连接 Neo4j 数据库执行:
# "MATCH (n:Drug) RETURN n.name, n.synonyms LIMIT 100"
# ==========================================

class MockDataSampler:
    def __init__(self):
        # 模拟数据库中已有的实体及其同义词/元数据
        self.drugs = [
            {"name": "Semaglutide", "synonyms": ["Ozempic", "Wegovy", "Rybelsus"], "type": "drug"},
            {"name": "Metformin", "synonyms": ["Glucophage", "Riomet"], "type": "drug"},
            {"name": "Lecanemab", "synonyms": ["Leqembi"], "type": "drug"},
            {"name": "Warfarin", "synonyms": ["Coumadin", "Jantoven"], "type": "drug"}
        ]
        self.diseases = [
            {"name": "Type 2 diabetes mellitus", "synonyms": ["T2D", "Diabetes type 2"], "type": "disease"},
            {"name": "Alzheimer's disease", "synonyms": ["AD", "Alzheimers"], "type": "disease"},
            {"name": "Hypertension", "synonyms": ["High blood pressure"], "type": "disease"}
        ]
        self.genes = [
            {"name": "GLP1R", "synonyms": ["GLP-1 receptor"], "type": "gene/protein"},
            {"name": "APP", "synonyms": ["Amyloid beta precursor protein"], "type": "gene/protein"},
            {"name": "ACE2", "synonyms": ["Angiotensin-converting enzyme 2"], "type": "gene/protein"}
        ]
        self.phenotypes = [
            {"name": "Nausea", "type": "effect/phenotype"},
            {"name": "Headache", "type": "effect/phenotype"},
            {"name": "Weight loss", "type": "effect/phenotype"}
        ]

    def get_random_entity(self, entity_type):
        if entity_type == "drug": return random.choice(self.drugs)
        if entity_type == "disease": return random.choice(self.diseases)
        if entity_type == "gene": return random.choice(self.genes)
        if entity_type == "phenotype": return random.choice(self.phenotypes)
        return None

# ==========================================
# 2. 查询生成核心逻辑
# ==========================================

class QueryGenerator:
    def __init__(self, sampler):
        self.sampler = sampler

    def generate_add_indication(self):
        """生成：添加药物-疾病适应症 (Indication)"""
        drug = self.sampler.get_random_entity("drug")
        disease = self.sampler.get_random_entity("disease")
        
        # 1. 核心 ADD 查询
        cypher_add = (
            f"MATCH (d:Drug {{name: '{drug['name']}'}}), (dis:Disease {{name: '{disease['name']}'}}) "
            f"MERGE (d)-[r:INDICATION]->(dis) "
            f"SET r.version = 'v_update_2023', r.source = 'AutoTest'"
        )
        nlp_add = f"Add a new indication showing that {drug['name']} treats {disease['name']}."

        # 2. 模糊验证查询
        # 策略：如果存在同义词，使用同义词进行验证；否则使用部分匹配
        if drug['synonyms']:
            fuzzy_name = random.choice(drug['synonyms'])
            verification_logic = f"using the brand name '{fuzzy_name}'"
            where_clause = f"WHERE (d.name = '{fuzzy_name}' OR '{fuzzy_name}' IN d.synonyms)"
        else:
            fuzzy_name = drug['name'][0:4] # 取前4个字母
            verification_logic = f"using the partial name '{fuzzy_name}...'"
            where_clause = f"WHERE d.name CONTAINS '{fuzzy_name}'"

        cypher_verify = (
            f"MATCH (d:Drug)-[r:INDICATION]->(dis:Disease) "
            f"{where_clause} AND dis.name CONTAINS '{disease['name'].split()[0]}' "
            f"RETURN d.name, dis.name, r.version"
        )
        
        nlp_verify = (
            f"Verify if the drug known as '{fuzzy_name}' has an indication for {disease['name']} "
            f"({verification_logic}). Expect to see version 'v_update_2023'."
        )

        return {
            "type": "ADD",
            "target_edge": "indication",
            "cypher_action": cypher_add,
            "nlp_action": nlp_add,
            "cypher_verify": cypher_verify,
            "nlp_verify": nlp_verify
        }

    def generate_archive_interaction(self):
        """生成：归档药物相互作用 (Drug-Drug Interaction)"""
        drug1 = self.sampler.get_random_entity("drug")
        drug2 = self.sampler.get_random_entity("drug")
        while drug1 == drug2: drug2 = self.sampler.get_random_entity("drug")

        # 1. 核心 ARCHIVE 查询
        cypher_archive = (
            f"MATCH (a:Drug {{name: '{drug1['name']}'}})-[r:SYNERGISTIC_INTERACTION]-(b:Drug {{name: '{drug2['name']}'}}) "
            f"SET r.status = 'archived', r.archived_date = date()"
        )
        nlp_archive = f"Archive the synergistic interaction between {drug1['name']} and {drug2['name']}."

        # 2. 模糊验证查询
        # 策略：忽略大小写 (Case Insensitive)
        fuzzy_name1 = drug1['name'].lower()
        
        cypher_verify = (
            f"MATCH (a:Drug)-[r:SYNERGISTIC_INTERACTION]-(b:Drug) "
            f"WHERE toLower(a.name) = '{fuzzy_name1}' AND b.name = '{drug2['name']}' "
            f"RETURN r.status"
        )
        
        nlp_verify = (
            f"Check the interaction status for '{fuzzy_name1}' (lowercase) and {drug2['name']}. "
            f"It should return 'archived'."
        )

        return {
            "type": "ARCHIVE",
            "target_edge": "drug_drug",
            "cypher_action": cypher_archive,
            "nlp_action": nlp_archive,
            "cypher_verify": cypher_verify,
            "nlp_verify": nlp_verify
        }

    def generate_add_side_effect(self):
        """生成：添加药物副作用 (Side Effect)"""
        drug = self.sampler.get_random_entity("drug")
        phenotype = self.sampler.get_random_entity("phenotype")

        # 1. 核心 ADD 查询
        cypher_add = (
            f"MATCH (d:Drug {{name: '{drug['name']}'}}), (p:Phenotype {{name: '{phenotype['name']}'}}) "
            f"MERGE (d)-[r:SIDE_EFFECT]->(p) "
            f"SET r.verified = true"
        )
        nlp_add = f"Record that {drug['name']} causes {phenotype['name']} as a side effect."

        # 2. 模糊验证查询
        # 策略：反向查找 (Reverse Lookup) - 通过副作用找药物
        cypher_verify = (
            f"MATCH (d:Drug)-[r:SIDE_EFFECT]->(p:Phenotype) "
            f"WHERE p.name = '{phenotype['name']}' "
            f"AND d.name = '{drug['name']}' " # 这里为了演示加了精确匹配，实际模糊验证可以去掉这行看列表
            f"RETURN d.name, r.verified"
        )
        
        nlp_verify = (
            f"Find which drug causes '{phenotype['name']}' and check if the relationship is verified. "
            f"Confirm {drug['name']} is in the list."
        )

        return {
            "type": "ADD",
            "target_edge": "drug_effect",
            "cypher_action": cypher_add,
            "nlp_action": nlp_add,
            "cypher_verify": cypher_verify,
            "nlp_verify": nlp_verify
        }

# ==========================================
# 3. 主执行函数
# ==========================================

def generate_test_suite(num_queries=5):
    sampler = MockDataSampler()
    generator = QueryGenerator(sampler)
    
    dataset_info = {
        "total_nodes": 135010,
        "dataset_context": "Biomedical Knowledge Graph (Drugs, Diseases, Genes)"
    }
    
    results = []
    
    # 随机生成指定数量的查询
    for _ in range(num_queries):
        scenario_type = random.choice(["add_indication", "archive_interaction", "add_side_effect"])
        
        if scenario_type == "add_indication":
            results.append(generator.generate_add_indication())
        elif scenario_type == "archive_interaction":
            results.append(generator.generate_archive_interaction())
        elif scenario_type == "add_side_effect":
            results.append(generator.generate_add_side_effect())
            
    return {
        "meta": dataset_info,
        "generated_queries": results
    }

# 运行生成
if __name__ == "__main__":
    test_suite = generate_test_suite(num_queries=3)
    print(json.dumps(test_suite, indent=2, ensure_ascii=False))