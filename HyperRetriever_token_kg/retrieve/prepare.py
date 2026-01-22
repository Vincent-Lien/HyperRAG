import os
import json
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import random
import torch
import argparse

from entity_extract import extract_entities
from model.emb import get_embedding

random.seed(42)  # For reproducibility

DOMAINS = ['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax']

### Start - Load graph ###
def load_graph(working_dir: Path):
    hypergraph_path = working_dir / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(hypergraph_path)
    return graph
### End of - Load graph ###

### Start - Create and Save Graph Embeddings ###
def create_and_save_graph_embeddings(working_dir: Path):
    """
    Loads the graph, generates embeddings for all nodes and relations (from edges),
    and saves them to files for future use.
    """
    # Define paths for node and relation embeddings
    embedding_nodes_path = working_dir / "graph_nodes.json"
    embedding_nodes_tensor_path = working_dir / "graph_nodes_embeddings.pt"
    embedding_relations_path = working_dir / "graph_relations.json"
    embedding_relations_tensor_path = working_dir / "graph_relations_embeddings.pt"

    # Check if all embedding files already exist
    if (embedding_nodes_path.exists() and
        embedding_nodes_tensor_path.exists() and
        embedding_relations_path.exists() and
        embedding_relations_tensor_path.exists()):
        print("Graph node and relation embeddings already exist. Skipping generation.")
        return

    print("Generating graph embeddings for nodes and relations...")
    graph = load_graph(working_dir)
    
    # --- 1. Process and save node embeddings ---
    if not (embedding_nodes_path.exists() and embedding_nodes_tensor_path.exists()):
        print("\nProcessing nodes...")
        nodes = list(graph.nodes())
        
        # Generate embeddings for nodes
        node_embeddings = get_embedding(nodes)
        
        # Save the nodes list (for mapping) and the embeddings tensor
        with open(embedding_nodes_path, 'w', encoding='utf-8') as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)
        torch.save(node_embeddings, embedding_nodes_tensor_path)
        
        print(f"Saved {len(nodes)} node embeddings to {embedding_nodes_tensor_path}")
        print(f"Saved node list to {embedding_nodes_path}")
    else:
        print("Node embeddings already exist. Skipping.")

    # --- 2. Process and save relation embeddings ---
    if not (embedding_relations_path.exists() and embedding_relations_tensor_path.exists()):
        print("\nProcessing relations...")
        # Extract unique relations from edge attributes
        relations = sorted(list(set(nx.get_edge_attributes(graph, 'relation').values())))
        
        if relations:
            # Generate embeddings for relations
            relation_embeddings = get_embedding(relations)
            
            # Save the relations list (for mapping) and the embeddings tensor
            with open(embedding_relations_path, 'w', encoding='utf-8') as f:
                json.dump(relations, f, ensure_ascii=False, indent=2)
            torch.save(relation_embeddings, embedding_relations_tensor_path)
            
            print(f"Saved {len(relations)} relation embeddings to {embedding_relations_tensor_path}")
            print(f"Saved relation list to {embedding_relations_path}")
        else:
            print("No relations found in the graph. Skipping relation embedding generation.")
    else:
        print("Relation embeddings already exist. Skipping.")
### End - Create and Save Graph Embeddings ###

### Start - Load queries and answers ###
def load_queries_and_answers(dataset_dir: Path):
    queries_path = dataset_dir / "train_queries.json"
    with open(queries_path, 'r') as f:
        queries = json.load(f)["('e', ('r', 'r', 'r'))"]

    answers_path = dataset_dir / "train_answers_hard.json"
    with open(answers_path, 'r') as f:
        answers = json.load(f)["('e', ('r', 'r', 'r'))"]

    return queries, answers
### End of - Load queries and answers ###


### Start - Prepare data ###
def prepare_data(domain: str):
    working_dir = Path("../kg/wikitopics") / domain
    dataset_dir = Path("../dataset/WikiTopicsQE_NLG") / domain
    output_path = working_dir / f"train/retrieval_samples.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = load_graph(working_dir) # This function loads the DiGraph correctly
    queries, answers = load_queries_and_answers(dataset_dir)
    
    all_output = []

    query_topic_entity_dict = extract_entities(domain, queries)
    for query, topic_entities_for_query in tqdm(query_topic_entity_dict.items(), desc="Processing queries"):
        filtered_topic_entities = [topic_entity for topic_entity in topic_entities_for_query 
                                   if topic_entity in graph.nodes()]
        
        query_answers = answers.get(query, None)
        if query_answers is None:
            continue
        
        filtered_answer_entities = [f"\"{answer.upper()}\"" for answer in answers[query] 
                                    if f"\"{answer.upper()}\"" in graph.nodes()]
        
        if not filtered_topic_entities or not filtered_answer_entities:
            continue

        all_positive_triplets_for_query = set()
        all_path_nodes = set()

        for topic_entity in filtered_topic_entities:
            for answer_entity in filtered_answer_entities:
                try:
                    # Find all shortest paths in the directed graph
                    paths = nx.all_shortest_paths(graph, source=topic_entity, target=answer_entity)
                    for path in paths:
                        all_path_nodes.update(path)
                        # Extract triplets from the path
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            # In DiGraph, there can be multiple edges between two nodes, but kg_construct.py creates one
                            if graph.has_edge(u, v):
                                relation = graph[u][v].get('relation')
                                if relation:
                                    all_positive_triplets_for_query.add((u, relation, v))
                except nx.NetworkXNoPath:
                    continue
        
        if all_positive_triplets_for_query:
            # Create a subgraph from all nodes found in the shortest paths
            subgraph = graph.subgraph(all_path_nodes)
            
            # Generate all possible triplets from the subgraph
            all_subgraph_triplets = set()
            for u, v, edge_data in subgraph.edges(data=True):
                relation = edge_data.get('relation')
                if relation:
                    all_subgraph_triplets.add((u, relation, v))

            # Negative samples are all triplets in the subgraph minus the positive ones
            negative_samples_set = all_subgraph_triplets - all_positive_triplets_for_query
            
            # Convert sets of tuples to lists of lists for JSON serialization
            positive_samples = [list(triplet) for triplet in all_positive_triplets_for_query]
            negative_samples = [list(triplet) for triplet in negative_samples_set]

            sample_data = {
                "query": query,
                "topic_entities": filtered_topic_entities,
                "positive_triplets": positive_samples,
                "negative_triplets": negative_samples
            }
            all_output.append(sample_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
### End of - Prepare data ###

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        config_path = Path("../config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            os.environ["OPENAI_API_KEY"] = config["hyperretriever_api_key"]

    parser = argparse.ArgumentParser(description="Prepare data for Hyperretriever.")
    parser.add_argument("domain", type=str, choices=DOMAINS + ['all'], 
                        help="Domain to prepare data for. Use 'all' to process all domains.")
    args = parser.parse_args()
    
    domains_to_process = DOMAINS if args.domain == "all" else [args.domain]

    for domain in domains_to_process:
        print(f"\n----- Processing domain: {domain} -----")
        working_dir = Path("../kg/wikitopics") / domain
        create_and_save_graph_embeddings(working_dir)
        prepare_data(domain)