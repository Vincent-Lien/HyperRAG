import os
import json
import argparse
from pathlib import Path
import networkx as nx

# Set up openai api key
if os.environ.get("OPENAI_API_KEY") is None:
    with open("../config.json", "r") as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config.get("hyperretriever_api_key", "your_openai_api_key_here")

# Define domains
DOMAINS = ['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax']

# Parse command line arguments
parser = argparse.ArgumentParser(description="Construct Knowledge Graph for WikiTopics.")
parser.add_argument("domain", type=str, choices=DOMAINS + ['all'],
                    help="Domain to construct Knowledge Graph for. Use 'all' to process all domains.")
args = parser.parse_args()

def construct_and_save_graph(domain):
    """Constructs and saves the knowledge graph for a given domain."""
    print(f"\nProcessing domain: '{domain}'...")
    dataset_dir = Path("../dataset/WikiTopicsQE_decoded") / domain

    hypergraph_path = Path("../expr/wikitopics/") / domain / "graph_chunk_entity_relation.graphml"

    train_document_path = dataset_dir / "train_graph.txt"
    test_document_path = dataset_dir / "test_inference.txt"

    try:
        with open(train_document_path, mode="r") as f:
            train_contexts = [train_context.strip() for train_context in f.readlines()]
        with open(test_document_path, mode="r") as f:
            test_contexts = [test_context.strip() for test_context in f.readlines()]
        unique_triples = [triple.split('\t') for triple in list(set(train_contexts + test_contexts))]

        H = nx.read_graphml(hypergraph_path)
        
        # Create a dictionary to store entity information from the hypergraph
        entity_info = {
            node: data
            for node, data in H.nodes(data=True)
            if not node.startswith('<hyperedge>')
        }

    except FileNotFoundError:
        print(f"Data files for domain '{domain}' not found. Skipping.")
        return

    # --- Knowledge Graph Construction ---
    print(f"Constructing knowledge graph for domain: '{domain}'...")

    # Initialize a directed graph for the knowledge graph
    G = nx.DiGraph()

    # Add nodes and edges from the unique triples
    # Each triple (head, relation, tail) is converted to a directed edge
    # from head to tail with the relation as an attribute.
    for triple in unique_triples:
        if len(triple) == 3:
            head, relation, tail = [item.strip() for item in triple]
            
            head_node = f"\"{head.upper()}\""
            tail_node = f"\"{tail.upper()}\""

            # Add edge to the graph
            G.add_edge(head_node, tail_node, relation=relation)

            # Update head node attributes if info exists in the hypergraph
            if head_node in entity_info:
                nx.set_node_attributes(G, {head_node: entity_info[head_node]})
            
            # Update tail node attributes if info exists in the hypergraph
            if tail_node in entity_info:
                nx.set_node_attributes(G, {tail_node: entity_info[tail_node]})

    print("Knowledge graph construction complete.")
    print(f" - Total nodes: {G.number_of_nodes()}")
    print(f" - Total edges: {G.number_of_edges()}")

    # --- Save Graph to File ---

    # Create a directory for the output if it doesn't exist
    output_dir = Path("../kg/wikitopics") / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the output path and save the graph as a GraphML file
    output_path = output_dir / f"graph_chunk_entity_relation.graphml"
    nx.write_graphml(G, output_path)

    print(f"Graph successfully saved to: {output_path}\n")

if args.domain == "all":
    for domain in DOMAINS:
        construct_and_save_graph(domain)
else:
    construct_and_save_graph(args.domain)
