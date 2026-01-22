import numpy as np
import torch
from typing import List, Dict, Tuple

class DDEEncoder:
    def __init__(self, max_hops: int = 3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.max_hops = max_hops
        self.device = device
        print(f"Using device: {self.device}")

    def prepare_data(self, subgraph: List[Tuple[str, str, str]], all_entities: List[str]) -> Tuple[Dict[str, int], int, torch.Tensor, torch.Tensor]:
        """
        Prepare graph structure data and directly convert it to Tensors, optimizing the edge creation process.
        """
        entity_to_idx = {entity: i for i, entity in enumerate(all_entities)}
        num_entities = len(all_entities)

        forward_edges = []
        backward_edges = []

        # Create edge lists in a single pass
        for head, _, tail in subgraph:
            head_idx = entity_to_idx.get(head)
            tail_idx = entity_to_idx.get(tail)
            if head_idx is not None and tail_idx is not None:
                # Forward: head -> tail
                forward_edges.append([head_idx, tail_idx])
                # Backward: head -> tail
                backward_edges.append([tail_idx, head_idx])
        
        # If there are no edges, create an empty tensor to avoid errors
        if not forward_edges:
            forward_edges_tensor = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            forward_edges_tensor = torch.tensor(forward_edges, dtype=torch.long, device=self.device).t()

        if not backward_edges:
            backward_edges_tensor = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            backward_edges_tensor = torch.tensor(backward_edges, dtype=torch.long, device=self.device).t()

        return entity_to_idx, num_entities, forward_edges_tensor, backward_edges_tensor

    def compute_dde(self, subgraph: List[Tuple[str, str, str]], topic_entities: List[str]) -> Dict[Tuple[str, str, str], np.ndarray]:
        """
        Compute DDE features (GPU version), optimized for the final feature extraction process.
        """
        if not subgraph:
            return {}

        all_entities = sorted(list(set(h for h, _, _ in subgraph) | set(t for _, _, t in subgraph)))
        entity_to_idx, num_entities, forward_edges, backward_edges = self.prepare_data(subgraph, all_entities)

        # 1. Initialize entity features
        initial_features = torch.zeros(num_entities, 1, device=self.device)
        topic_indices = [entity_to_idx[entity] for entity in topic_entities if entity in entity_to_idx]
        if topic_indices:
            initial_features[topic_indices] = 1.0
        
        entity_features = [initial_features]

        # 2. Multi-hop propagation
        current_features = initial_features
        for _ in range(self.max_hops):
            forward_features = self._propagate(current_features, forward_edges, num_entities)
            backward_features = self._propagate(current_features, backward_edges, num_entities)
            
            current_features = torch.cat([forward_features, backward_features], dim=1)
            entity_features.append(current_features)
        
        final_entity_features = torch.cat(entity_features, dim=1)

        # Vectorized feature extraction
        head_indices = [entity_to_idx[h] for h, _, t in subgraph if h in entity_to_idx and t in entity_to_idx]
        tail_indices = [entity_to_idx[t] for h, _, t in subgraph if h in entity_to_idx and t in entity_to_idx]
        
        valid_triples = [(h, r, t) for h, r, t in subgraph if h in entity_to_idx and t in entity_to_idx]

        if not valid_triples:
            return {}

        head_features = final_entity_features[head_indices]
        tail_features = final_entity_features[tail_indices]
        
        triple_features_tensor = torch.cat([head_features, tail_features], dim=1)
        triple_features_np = triple_features_tensor.cpu().numpy()

        triple_dde_features = {triple: feature for triple, feature in zip(valid_triples, triple_features_np)}
        
        return triple_dde_features

    def _propagate(self, features: torch.Tensor, edges: torch.Tensor, num_entities: int) -> torch.Tensor:
        if edges.numel() == 0:
            return torch.zeros(num_entities, features.size(1), device=self.device)

        source_features = features[edges[0]]
        
        new_features = torch.zeros(num_entities, features.size(1), device=self.device)
        new_features.scatter_add_(0, edges[1].unsqueeze(1).expand_as(source_features), source_features)
        
        in_degree = torch.zeros(num_entities, 1, device=self.device)
        in_degree.scatter_add_(0, edges[1].unsqueeze(1), torch.ones_like(edges[1], dtype=torch.float).unsqueeze(1))

        in_degree.clamp_(min=1.0)  # Avoid division by zero

        new_features /= in_degree
        
        return new_features