import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model.dde import DDEEncoder
from model.emb import get_embedding

class RetrievalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_and_save_dataset(json_file_path, save_path):
    """
    Processes the raw data, creates the dataset by loading pre-computed graph embeddings
    and only computing embeddings for queries, then saves it to a file.
    """
    # Determine working directory from input file path
    working_dir = Path(json_file_path).parent.parent

    # Define paths for node and relation embeddings
    embedding_nodes_path = working_dir / "graph_nodes.json"
    embedding_nodes_tensor_path = working_dir / "graph_nodes_embeddings.pt"
    embedding_relations_path = working_dir / "graph_relations.json"
    embedding_relations_tensor_path = working_dir / "graph_relations_embeddings.pt"

    # Check if all embedding files exist
    if not all([embedding_nodes_path.exists(), 
                embedding_nodes_tensor_path.exists(), 
                embedding_relations_path.exists(), 
                embedding_relations_tensor_path.exists()]):
        raise FileNotFoundError(
            f"Pre-computed graph embeddings not found in {working_dir}. "
            "Please run prepare.py first to generate them."
        )

    print("Loading pre-computed graph embeddings...")
    # Load nodes
    with open(embedding_nodes_path, 'r', encoding='utf-8') as f:
        graph_nodes = json.load(f)
    graph_nodes_embeddings = torch.load(embedding_nodes_tensor_path)
    
    # Load relations
    with open(embedding_relations_path, 'r', encoding='utf-8') as f:
        graph_relations = json.load(f)
    graph_relations_embeddings = torch.load(embedding_relations_tensor_path)

    # Create a unified embedding map for both graph nodes and relations
    embedding_map = {text: emb for text, emb in zip(graph_nodes, graph_nodes_embeddings)}
    embedding_map.update({text: emb for text, emb in zip(graph_relations, graph_relations_embeddings)})
    print(f"Loaded {len(embedding_map)} graph node and relation embeddings.")

    # Load retrieval samples
    with open(json_file_path, "r", encoding="utf-8") as f:
        retrieval_samples = json.load(f)

    # 1. Collect all unique queries for embedding
    print("Collecting all unique queries...")
    all_queries = set()
    for sample in tqdm(retrieval_samples, desc="Scanning samples for queries"):
        all_queries.add(sample["query"])
    
    unique_queries_list = list(all_queries)
    print(f"Found {len(unique_queries_list)} unique queries to embed.")

    # 2. Batch Embedding Calculation for queries
    if unique_queries_list:
        print("Calculating query embeddings in batches...")
        embedding_batch_size = 1024 # You can adjust this value
        query_embeddings_tensor = get_embedding(unique_queries_list, batch_size=embedding_batch_size).cpu()

        # 3. Add query embeddings to the map
        for text, emb in zip(unique_queries_list, query_embeddings_tensor):
            embedding_map[text] = emb
        print("Query embeddings added to the map.")

    # 4. Second Pass: Build the dataset using the pre-computed embeddings
    processed_data = []
    dde_encoder = DDEEncoder(max_hops=3)

    for sample in tqdm(retrieval_samples, desc="Processing samples and building dataset"):
        query = sample["query"]
        topic_entities = sample["topic_entities"]
        positive_triplets = sample["positive_triplets"]
        negative_triplets = sample["negative_triplets"]
        subgraph = positive_triplets + negative_triplets

        dde_features = dde_encoder.compute_dde(subgraph, topic_entities)
        
        if query not in embedding_map:
            print(f"Warning: Query '{query}' not found in embedding map. Skipping sample.")
            continue
        query_embedding = embedding_map[query]

        for triplet in positive_triplets:
            head, relation, tail = triplet
            if (head, relation, tail) in dde_features and head in embedding_map and relation in embedding_map and tail in embedding_map:
                concatenated_features = torch.cat([
                    query_embedding,
                    embedding_map[head],
                    embedding_map[relation],
                    embedding_map[tail],
                    torch.tensor(dde_features[(head, relation, tail)], dtype=torch.float32)
                ])
                processed_data.append({'features': concatenated_features, 'label': torch.tensor(1, dtype=torch.float32)})

        for triplet in negative_triplets:
            head, relation, tail = triplet
            if (head, relation, tail) in dde_features and head in embedding_map and relation in embedding_map and tail in embedding_map:
                concatenated_features = torch.cat([
                    query_embedding,
                    embedding_map[head],
                    embedding_map[relation],
                    embedding_map[tail],
                    torch.tensor(dde_features[(head, relation, tail)], dtype=torch.float32)
                ])
                processed_data.append({'features': concatenated_features, 'label': torch.tensor(0, dtype=torch.float32)})

    # 5. Save the processed data
    print(f"Saving dataset to {save_path}...")
    torch.save(processed_data, save_path)
    print("Dataset saved.")
    
    return RetrievalDataset(processed_data)