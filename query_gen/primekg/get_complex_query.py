import csv
import json
import random
from collections import defaultdict

# ==============================================================================
#                           1. Load KG.csv
# ==============================================================================

def load_kg_csv(path):
    """
    Load your KG CSV with columns:
    relation, display_relation,
    x_index, x_id, x_type, x_name, x_source,
    y_index, y_id, y_type, y_name, y_source
    """

    nodes = defaultdict(list)     # node_type → list of nodes
    relations = set()             # relation types
    edges = []                    # list of (x_node, relation, y_node)

    with open(path, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_node = {
                "id": row["x_id"],
                "type": row["x_type"],
                "name": row["x_name"],
                "src": row["x_source"]
            }
            y_node = {
                "id": row["y_id"],
                "type": row["y_type"],
                "name": row["y_name"],
                "src": row["y_source"]
            }

            nodes[x_node["type"]].append(x_node)
            nodes[y_node["type"]].append(y_node)

            relations.add(row["relation"])
            edges.append((x_node, row["relation"], y_node))

    return nodes, relations, edges


# ==============================================================================
#                          2. Sampling Utilities
# ==============================================================================

def sample_node(nodes, t):
    return random.choice(nodes[t]) if t in nodes else None

def sample_relation(relations):
    return random.choice(list(relations))


# ==============================================================================
#                     3. Query Templates (Core Logic)
# ==============================================================================

def query_disease_phenotype_protein_drug(nodes):
    """Disease → Phenotype → Protein → Drug"""
    d = sample_node(nodes, "disease")

    nl = (
        f"Find drugs that may affect the disease '{d['name']}' by identifying "
        f"related phenotypes, proteins involved, and drugs that target these proteins."
    )

    cypher = f"""
MATCH (d:disease {{name: '{d['name']}'}})-[:disease_phenotype_positive]->(ph:effect_phenotype)
MATCH (g:gene_protein)-[:phenotype_protein]->(ph)
MATCH (dr:drug)-[:drug_protein]->(g)
RETURN DISTINCT dr LIMIT 50
    """

    return nl, cypher.strip()


def query_anatomy_protein_disease(nodes):
    """Anatomy → Protein → Disease"""
    a = sample_node(nodes, "anatomy")

    nl = (
        f"Find diseases associated with proteins expressed in the anatomical structure "
        f"'{a['name']}'."
    )

    cypher = f"""
MATCH (p:gene_protein)-[:anatomy_protein_present]->(a:anatomy {{name: '{a['name']}'}})
MATCH (p)-[:disease_protein]->(d:disease)
RETURN DISTINCT d LIMIT 50
    """

    return nl, cypher.strip()


def query_drug_target_pathway(nodes):
    """Drug → Protein Target → Pathway"""
    dr = sample_node(nodes, "drug")

    nl = (
        f"Find pathways affected by drug '{dr['name']}' by identifying the proteins targeted by the drug "
        f"and the pathways in which these proteins participate."
    )

    cypher = f"""
MATCH (d:drug {{name: '{dr['name']}'}})-[:drug_protein]->(p:gene_protein)
MATCH (p)-[:pathway_protein]->(pw:pathway)
RETURN DISTINCT pw LIMIT 50
    """

    return nl, cypher.strip()


def query_random_multi_hop(nodes, relations):
    """Random 3-hop path between two random nodes."""
    start = random.choice(random.choice(list(nodes.values())))
    end = random.choice(random.choice(list(nodes.values())))
    
    # 将节点类型转换为有效的 Cypher 标签格式
    start_label = start['type'].replace('/', '_')
    end_label = end['type'].replace('/', '_')

    r1, r2, r3 = sample_relation(relations), sample_relation(relations), sample_relation(relations)

    nl = (
        f"Find all possible 3-hop paths between '{start['name']}' ({start['type']}) and '{end['name']}' ({end['type']}) "
        f"using relations such as '{r1}', '{r2}', and '{r3}'."
    )

    cypher = f"""
MATCH (a:{start_label} {{name: '{start['name']}'}})
MATCH (b:{end_label} {{name: '{end['name']}'}})
MATCH p = (a)-[:{r1}]-()-[:{r2}]-()-[:{r3}]-(b)
RETURN p LIMIT 20
    """

    return nl, cypher.strip()


def query_rank_drugs_by_target_count(nodes):
    """Rank drugs by how many proteins they target."""
    nl = (
        "Rank drugs by the number of proteins they target, and return the top drugs "
        "with the highest number of targets."
    )

    cypher = f"""
MATCH (d:drug)-[:drug_protein]->(p:gene_protein)
WITH d, COUNT(p) as cnt
ORDER BY cnt DESC
RETURN d, cnt LIMIT 50
    """

    return nl, cypher.strip()


# ==============================================================================
#                     4. Master Query Generation Orchestrator
# ==============================================================================

TEMPLATES = [
    query_disease_phenotype_protein_drug,
    query_anatomy_protein_disease,
    query_drug_target_pathway,
    query_random_multi_hop,
    query_rank_drugs_by_target_count,
]


def generate_queries(n, nodes, relations):
    output = []
    for _ in range(n):
        fn = random.choice(TEMPLATES)
        nl, cypher = fn(nodes, relations) if fn.__code__.co_argcount == 2 else fn(nodes)
        output.append({"nl": nl, "cypher": cypher})
    return output


# ==============================================================================
#                           5. Main entry
# ==============================================================================

if __name__ == "__main__":
    kg_path = "kg.csv"
    output_path = "generated_queries.json"
    
    print("Loading KG...")
    nodes, relations, edges = load_kg_csv(kg_path)

    print("Generating queries...")
    num_queries = 100  # 可以调整生成的查询数量
    queries = generate_queries(num_queries, nodes, relations)

    print(f"Saving {len(queries)} queries to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(queries)} queries to {output_path}")
