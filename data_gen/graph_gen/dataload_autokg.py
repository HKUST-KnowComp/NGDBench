import json
import networkx as nx
import csv
import ast
import hashlib
import os
import pickle
import html
import re
from pathlib import Path

# Regex to match *illegal* XML characters (XML 1.0 spec)
_ILLEGAL_XML_RE = re.compile(
    "[" +
    "\x00-\x08" +
    "\x0B" +
    "\x0C" +
    "\x0E-\x1F" +
    "\uD800-\uDFFF" +   # Surrogates
    "\uFFFE\uFFFF" +    # Noncharacters
    "]"
)
BUFFER_DIR = Path(__file__).resolve().parent / "graph_buffer"


def _resolve_dataset_root(target_dataset):
    if not target_dataset:
        raise ValueError("target_dataset must be provided.")
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data_gen" / "gnd_dataset" / target_dataset


def _ensure_parent_dir(path_like):
    output_path = Path(path_like)
    parent = output_path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _build_dataset_paths(target_dataset, filename_pattern, include_concept=True):
    dataset_root = _resolve_dataset_root(target_dataset)
    triple_node_file = dataset_root / "triples_csv" / f"triple_nodes_{filename_pattern}_from_json_without_emb.csv"
    text_node_file = dataset_root / "triples_csv" / f"text_nodes_{filename_pattern}_from_json.csv"
    triple_edge_dir = "concept_csv" if include_concept else "triples_csv"
    triple_edge_suffix = "with_concept" if include_concept else "without_emb"
    triple_edge_file = dataset_root / triple_edge_dir / f"triple_edges_{filename_pattern}_from_json_{triple_edge_suffix}.csv"
    text_edge_file = dataset_root / "triples_csv" / f"text_edges_{filename_pattern}_from_json.csv"
    concept_node_file = dataset_root / "concept_csv" / f"concept_nodes_{filename_pattern}_from_json_with_concept.csv" if include_concept else None
    concept_edge_file = dataset_root / "concept_csv" / f"concept_edges_{filename_pattern}_from_json_with_concept.csv" if include_concept else None

    return {
        "dataset_root": dataset_root,
        "triple_node_file": triple_node_file,
        "text_node_file": text_node_file,
        "triple_edge_file": triple_edge_file,
        "text_edge_file": text_edge_file,
        "concept_node_file": concept_node_file,
        "concept_edge_file": concept_edge_file,
    }


def convert_to_graphml(target_dataset, filename_pattern, include_concept=True):
    paths = _build_dataset_paths(target_dataset, filename_pattern, include_concept)
    output_file = BUFFER_DIR / f"{filename_pattern}_graph.graphml"

    csvs_to_graphml(
        triple_node_file=str(paths["triple_node_file"]),
        text_node_file=str(paths["text_node_file"]),
        triple_edge_file=str(paths["triple_edge_file"]),
        text_edge_file=str(paths["text_edge_file"]),
        concept_node_file=str(paths["concept_node_file"]) if paths["concept_node_file"] else None,
        concept_edge_file=str(paths["concept_edge_file"]) if paths["concept_edge_file"] else None,
        output_file=str(output_file),
        include_concept=include_concept
    )


def convert_to_gpickle(target_dataset, filename_pattern, include_concept=True):
    paths = _build_dataset_paths(target_dataset, filename_pattern, include_concept)
    output_file = BUFFER_DIR / f"{filename_pattern}.gpickle"

    csvs_to_gpickle(
        triple_node_file=str(paths["triple_node_file"]),
        text_node_file=str(paths["text_node_file"]),
        triple_edge_file=str(paths["triple_edge_file"]),
        text_edge_file=str(paths["text_edge_file"]),
        concept_node_file=str(paths["concept_node_file"]) if paths["concept_node_file"] else None,
        concept_edge_file=str(paths["concept_edge_file"]) if paths["concept_edge_file"] else None,
        output_file=str(output_file),
        include_concept=include_concept
    )

def sanitize_xml_string(s: str) -> str:
    """Remove illegal XML characters from a string."""
    return _ILLEGAL_XML_RE.sub("", s)

def get_node_id(entity_name, entity_to_id={}):
    """Returns existing or creates new nX ID for an entity using a hash-based approach."""
    if entity_name not in entity_to_id:
        # Use a hash function to generate a unique ID
        entity_name = entity_name+'_entity'
        hash_object = hashlib.sha256(entity_name.encode('utf-8'))
        hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash
        # Use the first 8 characters of the hash as the ID (you can adjust the length as needed)
        entity_to_id[entity_name] = hash_hex
    return entity_to_id[entity_name]

def csvs_to_temp_graphml(triple_node_file, triple_edge_file, output_file=None):
    g = nx.DiGraph()
    entity_to_id = {}

    # Add triple nodes
    with open(triple_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            mapped_id = get_node_id(node_id, entity_to_id)
            if mapped_id not in g.nodes:
                g.add_node(mapped_id, id=node_id, type=row["type"]) 
            

    # Add triple edges
    with open(triple_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            # Check if edge already exists to prevent duplicates
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])

    # save graph to buffer if no output path provided
    if output_file is None:
        output_file = BUFFER_DIR / "graph_without_concept.pkl"
    output_path = _ensure_parent_dir(output_file)
    with open(output_path, 'wb') as output_handle:
        pickle.dump(g, output_handle)
    
def validate_graphml(output_file):
    """Validate that a GraphML file can be read back correctly."""
    try:
        # Try to read the file back
        test_graph = nx.read_graphml(output_file)
        node_count = test_graph.number_of_nodes()
        edge_count = test_graph.number_of_edges()
        print(f"GraphML validation successful: {node_count} nodes, {edge_count} edges")
        return True
    except Exception as e:
        print(f"ERROR: GraphML validation failed: {str(e)}")
        # Optionally print the line number where the error occurred
        if hasattr(e, 'position'):
            line_no = e.position[0]
            print(f"Error at line {line_no}")
            
            # Read the problematic line
            with open(output_file, 'r') as f:
                lines = f.readlines()
                if line_no - 1 < len(lines):
                    print(f"Problematic line: {lines[line_no-1].strip()}")
        return False

def _build_graph_from_csvs(triple_node_file, text_node_file, triple_edge_file, text_edge_file,
                           concept_node_file=None, concept_edge_file=None, include_concept=True):
    """
    Build a NetworkX DiGraph from the provided CSV files.
    """
    g = nx.DiGraph()
    entity_to_id = {}

    # Add triple nodes
    with open(triple_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            mapped_id = get_node_id(node_id, entity_to_id)
            if mapped_id not in g.nodes:
                g.add_node(mapped_id, id=sanitize_xml_string(node_id), type=row["type"])

    # Add text nodes
    with open(text_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["text_id:ID"]
            if node_id not in g.nodes:
                sanitized_id = sanitize_xml_string(node_id)
                g.add_node(sanitized_id, file_id=sanitized_id, id=row["original_text"], type="passage")

    # Add concept nodes
    if concept_node_file is not None:
        with open(concept_node_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row["concept_id:ID"]
                if node_id not in g.nodes:
                    g.add_node(sanitize_xml_string(node_id), file_id="concept_file", id=row["name"], type="concept")

    # Add triple edges
    with open(triple_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])
                for node_id in [start_id, end_id]:
                    if g.nodes[node_id]['type'] in ['triple', 'concept'] and 'file_id' not in g.nodes[node_id]:
                        g.nodes[node_id]['file_id'] = row.get("file_id", "triple_file")

            if include_concept:
                concepts = ast.literal_eval(row["concepts"])
                for concept in concepts:
                    if "concepts" not in g.edges[start_id, end_id]:
                        g.edges[start_id, end_id]['concepts'] = str(concept)
                    else:
                        current_concepts = g.edges[start_id, end_id]['concepts'].split(",")
                        if str(concept) not in current_concepts:
                            g.edges[start_id, end_id]['concepts'] += "," + str(concept)

    # Add text edges
    with open(text_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = row[":END_ID"]
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id, relation="mention in", type=row[":TYPE"])
                if 'file_id' in g.nodes[start_id]:
                    g.nodes[start_id]['file_id'] += "," + str(end_id)
                else:
                    g.nodes[start_id]['file_id'] = str(end_id)

    # Add concept edges between triple nodes and concept nodes
    if concept_edge_file is not None:
        with open(concept_edge_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_id = get_node_id(row[":START_ID"], entity_to_id)
                end_id = row[":END_ID"]  # end id is concept node id
                if not g.has_edge(start_id, end_id):
                    g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])

    return g


def csvs_to_graphml(triple_node_file, text_node_file, triple_edge_file, text_edge_file,
                    concept_node_file=None, concept_edge_file=None,
                    output_file="kg.graphml",
                    include_concept=True):
    '''
    Convert multiple CSV files into a single GraphML file.
    '''
    g = _build_graph_from_csvs(
        triple_node_file,
        text_node_file,
        triple_edge_file,
        text_edge_file,
        concept_node_file=concept_node_file,
        concept_edge_file=concept_edge_file,
        include_concept=include_concept
    )
    output_path = _ensure_parent_dir(output_file)
    nx.write_graphml(g, output_path, infer_numeric_types=True)
    if validate_graphml(str(output_path)):
        print(f"Successfully created GraphML file: {output_path}")
    else:
        print(f"Failed to create valid GraphML file: {output_path}")


def csvs_to_gpickle(triple_node_file, text_node_file, triple_edge_file, text_edge_file,
                    concept_node_file=None, concept_edge_file=None,
                    output_file="kg.gpickle",
                    include_concept=True):
    '''
    Convert multiple CSV files into a NetworkX gpickle binary.
    '''
    g = _build_graph_from_csvs(
        triple_node_file,
        text_node_file,
        triple_edge_file,
        text_edge_file,
        concept_node_file=concept_node_file,
        concept_edge_file=concept_edge_file,
        include_concept=include_concept
    )
    output_path = _ensure_parent_dir(output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
    print(f"Successfully created gpickle file: {output_path}")

