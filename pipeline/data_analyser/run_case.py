from dataload_autokg import *
from graph_handler import GraphInspector
from dataload_toolkit import *
import random

if __name__ == "__main__":
    csv_path = "/home/ylivm/fei_work/NGDB_Benchmark/data_gen/perturbed_dataset/PrimeKG_2512260650/kg_test.csv"
    save_path = "/home/ylivm/fei_work/NGDB_Benchmark/pipeline/data_analyser/buffer/Primekg_noise.gpickle"
    # graph = build_graph_from_kg_csv(csv_path, save_path)
    # convert_to_gpickle("agent_memory1", "agent_memory1", include_concept=False)
    # graph = load_graph(save_path)
    # graph_inspector = GraphInspector(graph)
    # graph_inspector.full_analysis(output_file=str(output_path))
    build_graph_from_kg_csv(csv_path, save_path)
