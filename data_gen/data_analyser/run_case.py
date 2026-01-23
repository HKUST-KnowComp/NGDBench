from graph_handler import GraphInspector
import pickle
import networkx as nx
from contextlib import redirect_stdout
from pathlib import Path
def load_graph(path: str) -> nx.MultiDiGraph:
    # load the graph from the file
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    # Load graph according to file type
    if suffix == ".gpickle":
        # 使用标准 pickle 读入，由于前面写入时就是直接 pickle.dump
        with open(input_path, "rb") as f:
            g = pickle.load(f)
    elif suffix == ".graphml":
        g = nx.read_graphml(input_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only .gpickle and .graphml are supported.")
    return g
if __name__ == "__main__":
    # csv_path = "/home/ylivm/fei_work/NGDB_Benchmark/data_gen/perturbed_dataset/PrimeKG_2512260650/kg_test.csv"
    save_path = "../graph_gen/graph_buffer/mcp_tragectory_cleaned_normalized_copy_no_concepts.gpickle"
    output_path = "buffer/mcp_tragectory_no_concepts_analysis1.txt"
    # graph = build_graph_from_kg_csv(csv_path, save_path)
    # convert_to_gpickle("agent_memory1", "agent_memory1", include_concept=False)
    graph = load_graph(save_path)
    graph_inspector = GraphInspector(graph)
    graph_inspector.full_analysis(output_file=str(output_path))
    # build_graph_from_kg_csv(csv_path, save_path)
    # graph_inspector.full_analysis(output_file=str(output_path))
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     with redirect_stdout(f):
    #         graph_inspector.sample_nodes()
    print(f"输出已保存到: {output_path}")