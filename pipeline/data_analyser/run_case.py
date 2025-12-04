from dataload_autokg import *
from graph_handler import GraphInspector
from dataload_toolkit import *
import random

if __name__ == "__main__":
    csv_path = "/home/ylivm/ngdb/ngdb_benchmark/data_gen/gnd_dataset/PrimeKG/kg.csv"
    save_path = "/home/ylivm/ngdb/ngdb_benchmark/pipeline/data_analyser/buffer/Primekg_perturb.gpickle"
    # graph = build_graph_from_kg_csv(csv_path, save_path)
    convert_to_gpickle("agent_memory1", "agent_memory1", include_concept=False)
    # graph = load_graph(save_path)
    # graph_inspector = GraphInspector(graph)
    # graph_inspector.summary()
    build_graph_from_kg_csv(csv_path, save_path)