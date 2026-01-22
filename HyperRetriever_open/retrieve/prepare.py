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

### Start - Load graph ###
def load_hypergraph(working_dir: Path):
    hypergraph_path = working_dir / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(hypergraph_path)
    return graph
### End of - Load graph ###

### Start - Create and Save Graph Embeddings ###
def create_and_save_graph_embeddings(working_dir: Path):
    """
    Loads the hypergraph, generates embeddings for all nodes (entities and hyperedges),
    and saves them to files for future use.
    """
    embedding_nodes_path = working_dir / "graph_embedding_nodes.json"
    embedding_tensor_path = working_dir / "graph_embeddings.pt"

    if embedding_nodes_path.exists() and embedding_tensor_path.exists():
        print("Graph embeddings already exist. Skipping generation.")
        return

    print("Generating graph embeddings...")
    graph = load_hypergraph(working_dir)
    
    nodes = list(graph.nodes())
    
    # Generate embeddings using the model from emb.py
    # The get_embedding function handles batching and progress bar
    embeddings = get_embedding(nodes)
    
    # Save the nodes list (for mapping) and the embeddings tensor
    with open(embedding_nodes_path, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
        
    torch.save(embeddings, embedding_tensor_path)
    
    print(f"\nSaved {len(nodes)} node embeddings to {embedding_tensor_path}")
    print(f"Saved node list to {embedding_nodes_path}")
### End - Create and Save Graph Embeddings ###

### Start - Load queries and answers ###
def load_queries_and_answers(dataset_dir: Path, dataset_name: str):
    queries_path = dataset_dir / f"{dataset_name}_query_train.jsonl"

    with open(queries_path, 'r') as f:
        data = [json.loads(line) for line in f]
        queries = [item['question'] for item in data]
        answers = {item['question']: item['answer'] for item in data}

    return queries, answers
### End of - Load queries and answers ###

### Start - Entity linking ###
def entity_linking(graph: nx.Graph, topic_entities: list):
    all_nodes = set()
    for entity in topic_entities:
        nodes = nx.ego_graph(graph, entity, radius=3).nodes()
        all_nodes.update(nodes)

    subgraph = graph.subgraph(all_nodes).copy()
    return subgraph
### End of - Entity linking ###

### Start - Find path-guided subgraph ###
def find_path_guided_subgraph(graph: nx.Graph, topic_entities: list, all_path_nodes: set, max_hops: int):
    """
    Builds a subgraph by expanding from topic entities hop by hop.
    At each hop, it includes all neighbors, but only continues to expand from the
    neighbors that lie on a known shortest path to an answer entity.
    """
    subgraph_nodes = set(topic_entities)
    subgraph_edges = set()
    
    nodes_to_expand = set(topic_entities)
    expanded_entities = set()

    for hop in range(max_hops):
        if not nodes_to_expand:
            break
            
        next_nodes_to_expand = set()
        
        expanded_entities.update(nodes_to_expand)
        
        for entity in nodes_to_expand:
            # entity -> hyperedge
            for hyperedge in graph.neighbors(entity):
                if hyperedge.startswith('<hyperedge>'):
                    subgraph_nodes.add(hyperedge)
                    subgraph_edges.add((entity, hyperedge))
                    
                    # hyperedge -> next_entity
                    for next_entity in graph.neighbors(hyperedge):
                        if not next_entity.startswith('<hyperedge>'):
                            subgraph_nodes.add(next_entity)
                            subgraph_edges.add((hyperedge, next_entity))
                            
                            if next_entity in all_path_nodes and next_entity not in expanded_entities:
                                next_nodes_to_expand.add(next_entity)

        nodes_to_expand = next_nodes_to_expand
            
    result_graph = nx.Graph()
    result_graph.add_nodes_from(list(subgraph_nodes))
    result_graph.add_edges_from(list(subgraph_edges))
    
    return result_graph
### End of - Find path-guided subgraph ###

### Start - Triplet extraction ###
def get_triplets_from_path(path: list):
    """
    Extracts (entity1, hyperedge, entity2) triplets from a given path.
    A path is expected to alternate between entity and hyperedge nodes.
    Example path: [entity_A, hyperedge_1, entity_B, hyperedge_2, entity_C]
    Triplets: (entity_A, hyperedge_1, entity_B), (entity_B, hyperedge_2, entity_C)
    """
    triplets = set()
    if len(path) < 3:
        return triplets

    for i in range(0, len(path) - 2, 2):
        entity1 = path[i]
        hyperedge = path[i+1]
        entity2 = path[i+2]
        # Ensure the nodes are of the expected type
        if not entity1.startswith('<hyperedge>') and \
           hyperedge.startswith('<hyperedge>') and \
           not entity2.startswith('<hyperedge>'):
            triplets.add((entity1, hyperedge, entity2))
    return triplets
### End of - Triplet extraction ###


### Start - Efficient Negative Sampling ###
def generate_negative_samples(subgraph: nx.Graph, positive_triplets: set, num_samples: int):
    """
    Efficiently generates the specified number of negative samples by randomly sampling
    from the subgraph structure.
    """
    negative_samples = set()
    # To enable quick lookup, create a set containing positive samples and their reverse
    positive_lookup = set(positive_triplets)
    for t in positive_triplets:
        positive_lookup.add((t[2], t[1], t[0]))

    # Identify all hyperedges in the subgraph
    hyperedges = [n for n in subgraph.nodes() if n.startswith('<hyperedge>')]
    if not hyperedges:
        return []

    # Set a maximum number of attempts to prevent infinite loops in certain scenarios
    max_attempts = num_samples * 20 
    attempts = 0

    while len(negative_samples) < num_samples and attempts < max_attempts:
        attempts += 1
        # 1. Randomly select a hyperedge
        hyperedge = random.choice(hyperedges)
        
        # 2. Retrieve all entities connected to this hyperedge
        connected_entities = list(subgraph.neighbors(hyperedge))
        
        # If there are fewer than 2 entities, a triplet cannot be formed, skip
        if len(connected_entities) < 2:
            continue
            
        # 3. Randomly select two distinct entities
        entity1, entity2 = random.sample(connected_entities, 2)
        
        # 4. Form a potential negative sample triplet
        triplet = (entity1, hyperedge, entity2)
        
        # 5. Check if it is a positive sample or already in the negative samples set
        if triplet not in positive_lookup and triplet not in negative_samples:
            negative_samples.add(triplet)
            
    return list(negative_samples)
### End of - Efficient Negative Sampling ###


def prepare_data(dataset_name: str):
    working_dir = Path("../expr") / dataset_name
    dataset_dir = Path("../dataset/open_domain_dataset/open_domain_splitted_query")
    output_path = working_dir / f"train/retrieval_samples.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = load_hypergraph(working_dir)
    queries, answers = load_queries_and_answers(dataset_dir, dataset_name)
    
    all_output = []

    query_topic_entity_dict = extract_entities(dataset_name, queries)
    for query, topic_entities_for_query in tqdm(query_topic_entity_dict.items(), desc="Processing queries"):
        filtered_topic_entities = [topic_entity for topic_entity in topic_entities_for_query if topic_entity in graph.nodes()]
        query_answers = answers.get(query, None)
        if query_answers is None:
            print(f"  No answers found for query '{query}'. Skipping.")
            continue
        filtered_answer_entities = [f"\"{answer.upper()}\"" for answer in answers[query] if f"\"{answer.upper()}\"" in graph.nodes()]

        if not filtered_topic_entities or not filtered_answer_entities:
            # print(f"  Skipping query '{query}': No valid topic entities or answer entities found in graph.")
            continue

        max_k_hops_for_query = 0
        all_positive_triplets_for_query = set()
        all_paths = []
        
        for topic_entity in filtered_topic_entities:
            for answer_entity in filtered_answer_entities:
                try:
                    path = nx.shortest_path(graph, topic_entity, answer_entity)
                    all_paths.append(path)
                    path_length = len(path) - 1
                    k_hops = (path_length + 1) // 2
                    max_k_hops_for_query = max(max_k_hops_for_query, k_hops)
                    
                    path_triplets = get_triplets_from_path(path)
                    all_positive_triplets_for_query.update(path_triplets)
                except nx.NetworkXNoPath:
                    # print(f"  No path found from {topic_entity} to {answer_entity} for query '{query}'.")
                    continue
            
        if max_k_hops_for_query > 0:
            all_path_nodes = {node for path in all_paths for node in path}
            query_subgraph = find_path_guided_subgraph(
                graph, 
                filtered_topic_entities, 
                all_path_nodes, 
                max_k_hops_for_query
            )
            
            positive_samples = list(all_positive_triplets_for_query)
            num_positive = len(positive_samples)

            if num_positive > 0:
                negative_samples = generate_negative_samples(query_subgraph, all_positive_triplets_for_query, num_positive)
            else:
                negative_samples = []

            sample_data = {
                "query": query,
                "topic_entities": filtered_topic_entities,
                "max_k_hops": max_k_hops_for_query,
                "positive_triplets": positive_samples,
                "negative_triplets": negative_samples
            }
            all_output.append(sample_data)
            
        else:
            print(f"  No paths found for query '{query}' to any answer. Skipping subgraph generation.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        config_path = Path("../config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config["hyperretriever_api_key"]

    parser = argparse.ArgumentParser(description="Prepare data for HyperGraph MLP retrieval.")
    parser.add_argument("domain", type=str, choices=["2wikimultihopqa", "hotpotqa", "musique"], help="Domain to prepare data for.")
    args = parser.parse_args()
    
    domain = args.domain
    working_dir = Path("../expr") / domain
    create_and_save_graph_embeddings(working_dir)
    prepare_data(domain)