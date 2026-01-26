from dataload_clean import *
import pickle

def mcp_clean(input_file, output_file):
    with open(input_file, "rb") as f:
        g = pickle.load(f)
    g, stats = remove_concept_nodes_and_annotate_neighbors(g)
    with open(output_file, "wb") as f:
        pickle.dump(g, f)

if __name__ == "__main__":
    mcp_clean("graph_buffer/mcp_tragectory_cleaned_normalized.gpickle", "graph_buffer/mcp_tragectory_cleaned_normalized_concepts_aligned.gpickle")