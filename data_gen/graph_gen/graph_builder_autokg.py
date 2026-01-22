from dataload_autokg import *
# from graph_gen.graph_handler import GraphInspector
from dataload_toolkit import *
from dataload_clean import *
import random

if __name__ == "__main__":
    # csv_path = "/home/ylivm/fei_work/NGDB_Benchmark/data_gen/perturbed_dataset/PrimeKG_2512260650/kg_test.csv"
    # save_path = "/home/ylivm/fei_work/NGDB_Benchmark/pipeline/data_analyser/buffer/Primekg_noise.gpickle"
    # graph = build_graph_from_kg_csv(csv_path, save_path)
    # convert_to_gpickle("agent_memory1", "agent_memory1", include_concept=False)
    # graph = load_graph(save_path)
    # graph_inspector = GraphInspector(graph)
    # graph_inspector.full_analysis(output_file=str(output_path))
    # build_graph_from_kg_csv(csv_path, save_path)
    # convert_to_gpickle("mcp_tragectory", "mcp_tragectory", include_concept=True)
    # remove_isolated_nodes_from_file("graph_buffer/mcp_tragectory.gpickle", "graph_buffer/mcp_tragectory_cleaned.gpickle")
    normalize_mcp_relations(input_file="graph_buffer/mcp_tragectory_cleaned_normalized.gpickle", output_file="graph_buffer/mcp_tragectory_cleaned_normalized.gpickle", remove_original_edges=True)