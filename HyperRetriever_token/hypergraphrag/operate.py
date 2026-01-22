import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union, List, Tuple, Dict, Set
from collections import Counter, defaultdict
import warnings
import torch
import numpy as np
import networkx as nx
from pathlib import Path
import itertools
from datetime import datetime
import os

import time

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from retrieve.model.emb import get_embedding
from retrieve.model.dde import DDEEncoder
from retrieve.model.mlp import MLP


def _load_graph_embeddings(domain: str, device) -> Dict[str, torch.Tensor]:
    """
    Loads pre-computed graph node embeddings and their corresponding node names.
    """
    embedding_nodes_path = Path(f"../expr/wikitopics/{domain}/graph_embedding_nodes.json")
    embedding_tensor_path = Path(f"../expr/wikitopics/{domain}/graph_embeddings.pt")

    if not embedding_nodes_path.exists() or not embedding_tensor_path.exists():
        logger.error(f"Pre-computed graph embedding files not found in ../expr/wikitopics/{domain}/")
        return {}

    logger.info("Loading pre-computed graph embeddings...")
    with open(embedding_nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    embeddings_tensor = torch.load(embedding_tensor_path, map_location=device)

    if len(nodes) != embeddings_tensor.shape[0]:
        logger.error("Mismatch between number of nodes and embeddings. Aborting.")
        return {}

    embedding_map = {node: embeddings_tensor[i] for i, node in enumerate(nodes)}
    logger.info(f"Loaded {len(embedding_map)} node embeddings into memory.")
    
    return embedding_map


async def _score_and_filter_triplets_single_hop(
    hop: int,
    query: str,
    nodes_to_expand: Set[str],
    scored_triplets_lookup: Set[Tuple[str, str, str]],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    embedding_map: Dict[str, torch.Tensor],
    model: MLP,
    full_graph: nx.Graph,
    dde_encoder: DDEEncoder,
    all_topic_entities: List[str],
    threshold: float = 0.5,
    top_k: int = None,
) -> Tuple[List[Tuple[Tuple[str, str, str], float]], List[Tuple[Tuple[str, str, str], float]], Set[Tuple[str, str, str]]]:
    """
    Performs a single hop of expansion from a given set of nodes, scores the resulting
    triplets, and partitions them. If top_k is given, it partitions into top-k and the rest.
    Otherwise, it partitions them based on a threshold.
    It returns passing triplets, failing triplets, and the updated set of all triplets that have been scored.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generate candidate triplets from the current frontier of nodes
    hop_candidate_triplets = set()
    initial_entities_count = len(nodes_to_expand)
    for entity in nodes_to_expand:
        if entity not in full_graph:
            continue
        for hyperedge in full_graph.neighbors(entity):
            if not hyperedge.startswith('<hyperedge>'):
                continue
            for next_entity in full_graph.neighbors(hyperedge):
                if next_entity.startswith('<hyperedge>') or next_entity == entity:
                    continue
                
                triplet = (entity, hyperedge, next_entity)
                reverse_triplet = (next_entity, hyperedge, entity)
                
                if triplet not in scored_triplets_lookup and reverse_triplet not in scored_triplets_lookup:
                    hop_candidate_triplets.add(triplet)
                    scored_triplets_lookup.add(triplet)
                    scored_triplets_lookup.add(reverse_triplet)

    # # Log expansion stats for analysis
    # _log_expansion_stats(hop, initial_entities_count, len(hop_candidate_triplets))

    if not hop_candidate_triplets:
        logger.info(f"Hop {hop} expansion: {initial_entities_count} entities expanded to 0 triplets.")
        return [], [], scored_triplets_lookup
        
    logger.info(f"Hop {hop} expansion: {initial_entities_count} entities expanded to {len(hop_candidate_triplets)} triplets.")

    # 2. DDE Calculation for the new triplets
    dde_features = dde_encoder.compute_dde(list(hop_candidate_triplets), all_topic_entities)

    # 3. Prepare batch for scoring
    batch_features = []
    valid_triplets_for_hop = []
    query_embedding = get_embedding([query])
    q_emb = query_embedding[0].to(device)

    for h, r, t in hop_candidate_triplets:
        triplet_key = (h, r, t)
        if triplet_key not in dde_features:
            continue

        h_emb, r_emb, t_emb = embedding_map.get(h), embedding_map.get(r), embedding_map.get(t)
        if h_emb is None or r_emb is None or t_emb is None:
            continue

        dde_feat = torch.tensor(dde_features[triplet_key], dtype=torch.float32).to(device)
        features = torch.cat([q_emb, h_emb, r_emb, t_emb, dde_feat])
        
        batch_features.append(features)
        valid_triplets_for_hop.append(triplet_key)

    if not valid_triplets_for_hop:
        return [], [], scored_triplets_lookup

    # 4. Scoring and Filtering for the current hop
    passing_triplets_with_scores = []
    failed_triplets_with_scores = []
    with torch.no_grad():
        features_tensor = torch.stack(batch_features)
        scores = model(features_tensor).view(-1)

        predictions = torch.sigmoid(scores) # Always compute predictions for thresholding

        # First, filter by threshold
        threshold_passed_triplets = []
        threshold_failed_triplets = []
        for i in range(predictions.shape[0]):
            triplet = valid_triplets_for_hop[i]
            score = scores[i].item()
            if predictions[i] > threshold:
                threshold_passed_triplets.append((triplet, score))
            else:
                threshold_failed_triplets.append((triplet, score))

        if top_k is not None:
            # If top_k is specified, sort the threshold-passed triplets and take top_k
            threshold_passed_triplets.sort(key=lambda x: x[1], reverse=True)
            
            passing_triplets_with_scores = threshold_passed_triplets[:top_k]
            # Remaining threshold-passed triplets and all threshold-failed triplets go to failed_triplets_with_scores
            failed_triplets_with_scores = threshold_passed_triplets[top_k:] + threshold_failed_triplets
            logger.info(f"Scored {len(valid_triplets_for_hop)} triplets. {len(threshold_passed_triplets)} passed threshold (>{threshold}), selected top {len(passing_triplets_with_scores)}.")
        else:
            # If top_k is not specified, just use the threshold filtering
            passing_triplets_with_scores = threshold_passed_triplets
            failed_triplets_with_scores = threshold_failed_triplets
            logger.info(f"Scored {len(valid_triplets_for_hop)} triplets, {len(passing_triplets_with_scores)} passed threshold (>{threshold}).")
    
    return passing_triplets_with_scores, failed_triplets_with_scores, scored_triplets_lookup


async def _build_query_context(
    query: str,
    topic_entities: List[str],
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    global_config: dict,
    embedding_map: Dict[str, torch.Tensor] = None,
    model: MLP = None,
    full_graph: nx.Graph = None,
    dde_encoder: DDEEncoder = None,
    entity_connectivity: Dict[str, int] = None,
    relation_connectivity: Dict[str, int] = None,
    graph_density: float = None,
):
    """
    Builds a token-optimized context using a phased approach with specific budgets and sorting rules.
    The retrieval strategy is chosen based on graph density.
    """
    # --- 0. Pre-computation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not embedding_map or model is None or full_graph is None or dde_encoder is None or graph_density is None:
        logger.error("One or more required assets (embedding_map, model, full_graph, dde_encoder, graph_density) were not provided.")
        return None, None, None, None

    density = graph_density
    logger.info(f"Graph density is {density:.2f}.")

    initial_topic_entities = [te for te in topic_entities if te in embedding_map]
    if not initial_topic_entities:
        logger.warning(f"None of the initial topic entities were found in the graph. Aborting.")
        return None, None, None, None
    
    # --- 1. Retrieval based on Density ---
    start = time.time()

    all_scored_triplets = []
    scored_triplets_lookup = set() # MODIFIED: Moved from iterative_expansion to here
    
    # This is the iterative expansion logic that will be used by density strategies
    async def iterative_expansion(start_nodes, initial_threshold, scored_triplets_lookup, top_k=None):
        final_passing_triplets = []
        final_failing_triplets = []
        
        nodes_to_expand = set(start_nodes)
        expanded_nodes = set()
        # scored_triplets_lookup = set() # REMOVED: Moved up

        hop = 0
        while nodes_to_expand:
            hop += 1
            # logger.info(f"--- Hop {hop} (Threshold: {initial_threshold:.2f}) ---")
            
            expanded_nodes.update(nodes_to_expand)
            
            passing_this_hop, failing_this_hop, scored_triplets_lookup = await _score_and_filter_triplets_single_hop(
                hop=hop,
                query=query,
                nodes_to_expand=nodes_to_expand,
                scored_triplets_lookup=scored_triplets_lookup,
                knowledge_graph_inst=knowledge_graph_inst,
                global_config=global_config,
                embedding_map=embedding_map,
                model=model,
                full_graph=full_graph,
                dde_encoder=dde_encoder,
                all_topic_entities=initial_topic_entities,
                threshold=initial_threshold,
                top_k=top_k,
            )
            
            final_passing_triplets.extend(passing_this_hop)
            final_failing_triplets.extend(failing_this_hop)

            next_frontier_nodes = set()
            for (h, _, t), score in passing_this_hop:
                if h not in expanded_nodes:
                    next_frontier_nodes.add(h)
                if t not in expanded_nodes:
                    next_frontier_nodes.add(t)
            
            nodes_to_expand = next_frontier_nodes
        
        logger.info(f"Iterative expansion from {start_nodes} found {len(final_passing_triplets)} passing and {len(final_failing_triplets)} triplets.")
        return final_passing_triplets, final_failing_triplets

    if density <= 2.35:
        logger.info("Using Strategy: Low Density (<= 2.35). Iterative expansion with simple supplementation.")
        MLP_threshold = 0.5
        edge_nums_threshold = 50
        max_attempts = 5

        # Initial full retrieval
        all_scored_triplets, failed_triplets_pool = await iterative_expansion(initial_topic_entities, MLP_threshold, scored_triplets_lookup, top_k=None)
        
        unique_hyperedges = {triplet[0][1] for triplet in all_scored_triplets}
        logger.info(f"Initial pass: Found {len(unique_hyperedges)} unique hyperedges with threshold {MLP_threshold:.2f}.")

        attempts = 0
        while len(unique_hyperedges) < edge_nums_threshold and attempts < max_attempts:
            attempts += 1
            MLP_threshold -= 0.1
            if MLP_threshold < 0:
                logger.warning("MLP threshold dropped below 0. Stopping supplementation.")
                break
            
            logger.info(f"Hyperedge count ({len(unique_hyperedges)}) is less than {edge_nums_threshold}. Reducing threshold to {MLP_threshold:.2f} (Attempt {attempts}).")
            
            # Promote from the pool of failed triplets
            promoted_triplets = []
            remaining_failed_triplets = []
            for triplet, score in failed_triplets_pool:
                if torch.sigmoid(torch.tensor(score)).item() > MLP_threshold:
                    promoted_triplets.append((triplet, score))
                else:
                    remaining_failed_triplets.append((triplet, score))
            
            failed_triplets_pool = remaining_failed_triplets
            
            if not promoted_triplets:
                logger.info("No additional triplets could be promoted from the pool. Stopping.")
                continue

            logger.info(f"Promoted {len(promoted_triplets)} triplets with new threshold. No recursive expansion will be performed.")
            all_scored_triplets.extend(promoted_triplets)
            
            # Recalculate unique hyperedges and continue loop if needed
            unique_hyperedges = {triplet[0][1] for triplet in all_scored_triplets}
            logger.info(f"After supplementation attempt {attempts}, we have {len(unique_hyperedges)} unique hyperedges.")

    elif 2.35 < density <= 5:
        logger.info("Using Strategy: Medium Density (2.35 < d <= 5). Iterative expansion with recursive supplementation.")
        MLP_threshold = 0.5
        edge_nums_threshold = 50
        max_attempts = 5
        
        # Initial full retrieval
        all_scored_triplets, failed_triplets_pool = await iterative_expansion(initial_topic_entities, MLP_threshold, scored_triplets_lookup, top_k=None)
        
        unique_hyperedges = {triplet[0][1] for triplet in all_scored_triplets}
        logger.info(f"Initial pass: Found {len(unique_hyperedges)} unique hyperedges with threshold {MLP_threshold:.2f}.")

        attempts = 0
        while len(unique_hyperedges) < edge_nums_threshold and attempts < max_attempts:
            attempts += 1
            MLP_threshold -= 0.1
            if MLP_threshold < 0:
                logger.warning("MLP threshold dropped below 0. Stopping supplementation.")
                break
            
            logger.info(f"Hyperedge count ({len(unique_hyperedges)}) is less than {edge_nums_threshold}. Reducing threshold to {MLP_threshold:.2f} (Attempt {attempts}).")
            
            # Promote from the pool of failed triplets
            promoted_triplets = []
            remaining_failed_triplets = []
            for triplet, score in failed_triplets_pool:
                if torch.sigmoid(torch.tensor(score)).item() > MLP_threshold:
                    promoted_triplets.append((triplet, score))
                else:
                    remaining_failed_triplets.append((triplet, score))
            
            failed_triplets_pool = remaining_failed_triplets
            
            if not promoted_triplets:
                logger.info("No additional triplets could be promoted from the pool. Stopping.")
                continue

            logger.info(f"Promoted {len(promoted_triplets)} triplets with new threshold.")
            all_scored_triplets.extend(promoted_triplets)
            
            # Get new nodes to expand from the newly promoted triplets
            new_nodes_to_expand = set()
            current_expanded_nodes = {h for h,r,t in [s[0] for s in all_scored_triplets]} | {t for h,r,t in [s[0] for s in all_scored_triplets]}

            for (h, _, t), score in promoted_triplets:
                if h not in current_expanded_nodes:
                    new_nodes_to_expand.add(h)
                if t not in current_expanded_nodes:
                    new_nodes_to_expand.add(t)

            if new_nodes_to_expand:
                logger.info(f"Recursively expanding from {len(new_nodes_to_expand)} new entities from promoted triplets.")
                # Recursively expand from these new nodes
                newly_passing, newly_failing = await iterative_expansion(list(new_nodes_to_expand), MLP_threshold, scored_triplets_lookup, top_k=None)
                
                all_scored_triplets.extend(newly_passing)
                failed_triplets_pool.extend(newly_failing)
                logger.info(f"Recursive expansion added {len(newly_passing)} passing triplets.")
            else:
                logger.info("No new nodes to expand from promoted triplets.")

            unique_hyperedges = {triplet[0][1] for triplet in all_scored_triplets}
            logger.info(f"After supplementation attempt {attempts}, we have {len(unique_hyperedges)} unique hyperedges.")

    else: # density > 5
        logger.info("Using Strategy: High Density (> 5). Iterative expansion with top-k and recursive supplementation.")
        top_k_initial = 100
        top_k_supplement = 50
        edge_nums_threshold = 50
        max_attempts = 5

        # Initial full retrieval with top-k
        all_scored_triplets, failed_triplets_pool = await iterative_expansion(initial_topic_entities, 0.5, scored_triplets_lookup, top_k=top_k_initial)
        
        # Sort failed_triplets_pool by score descending to easily promote later
        failed_triplets_pool.sort(key=lambda x: x[1], reverse=True)

        unique_hyperedges = {triplet[0][1] for triplet in all_scored_triplets}
        logger.info(f"Initial pass: Found {len(unique_hyperedges)} unique hyperedges with top_k={top_k_initial}.")

        attempts = 0
        while len(unique_hyperedges) < edge_nums_threshold and attempts < max_attempts:
            attempts += 1
            logger.info(f"Hyperedge count ({len(unique_hyperedges)}) is less than {edge_nums_threshold}. Supplementing with top {top_k_supplement} from failed pool (Attempt {attempts}).")
            
            if not failed_triplets_pool:
                logger.info("Failed triplets pool is empty. Stopping supplementation.")
                break

            # Promote from the pool of failed triplets
            promoted_triplets = failed_triplets_pool[:top_k_supplement]
            failed_triplets_pool = failed_triplets_pool[top_k_supplement:]
            
            if not promoted_triplets:
                logger.info("No additional triplets could be promoted from the pool. Stopping.")
                break

            logger.info(f"Promoted {len(promoted_triplets)} triplets.")
            all_scored_triplets.extend(promoted_triplets)
            
            # Get new nodes to expand from the newly promoted triplets
            new_nodes_to_expand = set()
            current_expanded_nodes = {h for h,r,t in [s[0] for s in all_scored_triplets]} | {t for h,r,t in [s[0] for s in all_scored_triplets]}

            for (h, _, t), score in promoted_triplets:
                if h not in current_expanded_nodes:
                    new_nodes_to_expand.add(h)
                if t not in current_expanded_nodes:
                    new_nodes_to_expand.add(t)

            if new_nodes_to_expand:
                logger.info(f"Recursively expanding from {len(new_nodes_to_expand)} new entities from promoted triplets.")
                # Recursively expand from these new nodes
                newly_passing, newly_failing = await iterative_expansion(list(new_nodes_to_expand), 0.5, scored_triplets_lookup, top_k=top_k_initial)
                
                all_scored_triplets.extend(newly_passing)
                # Add newly failing triplets to the pool and re-sort
                failed_triplets_pool.extend(newly_failing)
                failed_triplets_pool.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"Recursive expansion added {len(newly_passing)} passing triplets.")
            else:
                logger.info("No new nodes to expand from promoted triplets.")

            unique_hyperedges = {triplet[0][1] for triplet in all_scored_triplets}
            logger.info(f"After supplementation attempt {attempts}, we have {len(unique_hyperedges)} unique hyperedges.")

    if not all_scored_triplets:
        logger.warning("No triplets passed the scoring and filtering.")
        return None, None, None, None

    retrieve_time = time.time() - start

    # 2. Initialization and Budgeting
    tiktoken_model_name = global_config.get("tiktoken_model_name", "gpt-4o-mini")
    max_context_tokens = 127000
    
    # New token allocation
    relation_token_budget = int(max_context_tokens * 0.5)
    entity_token_budget = int(max_context_tokens * 0.3)
    source_token_budget = int(max_context_tokens * 0.2)

    # 3. Scoring and Sorting Preparations
    relation_max_scores = defaultdict(float)
    for (h, r, t), score in all_scored_triplets:
        if score > relation_max_scores[r]:
            relation_max_scores[r] = score
    
    entity_scores = defaultdict(float)
    for (h, r, t), score in all_scored_triplets:
        if score > entity_scores[h]: entity_scores[h] = score
        if score > entity_scores[t]: entity_scores[t] = score

    # Sort relations: Primary by score (desc), Secondary by connectivity (desc)
    sorted_relations = sorted(
        relation_max_scores.keys(), 
        key=lambda r: (relation_max_scores.get(r, 0.0), relation_connectivity.get(r, 0)), 
        reverse=True
    )

    # 4. Phase 1: Fill Relations (and collect entities)
    final_relations = set()
    final_triplets = []
    final_entities_from_relations = set()
    
    logger.info(f"--- Phase 1: Filling Relations (Budget: {relation_token_budget} tokens) ---")
    
    # --- Batch insertion attempt ---
    logger.info("Attempting to add all relations in a batch.")
    tentative_relations_batch = set(sorted_relations)
    tentative_triplets_batch = [st for st in all_scored_triplets if st[0][1] in tentative_relations_batch]
    tentative_entities_batch = {h for h, r, t in [st[0] for st in tentative_triplets_batch]} | \
                               {t for h, r, t in [st[0] for st in tentative_triplets_batch]}

    # Build context string for batch
    temp_relations_section_list_data_batch = []
    rel_map_batch = defaultdict(list)
    for h, r, t in [st[0] for st in tentative_triplets_batch]:
        if r in tentative_relations_batch:
            rel_map_batch[r].append(h)
            rel_map_batch[r].append(t)

    for i, r_name in enumerate(sorted(list(tentative_relations_batch))):
        related_nodes_in_context_batch = {node for node in rel_map_batch[r_name] if node in tentative_entities_batch}
        related_nodes_batch = "|".join(sorted(list(related_nodes_in_context_batch)))
        temp_relations_section_list_data_batch.append([str(i), r_name, related_nodes_batch])

    temp_relations_section_list_batch = [["id", "hyperedge", "related_entities"]] + temp_relations_section_list_data_batch
    relations_context_csv_batch = list_of_list_to_csv(temp_relations_section_list_batch)
    relations_context_str_batch = f"-----Relationships-----\n```csv\n{relations_context_csv_batch}\n```"
    tokens_batch = encode_string_by_tiktoken(relations_context_str_batch, model_name=tiktoken_model_name)

    if len(tokens_batch) <= relation_token_budget:
        logger.info(f"Batch insertion successful. All {len(tentative_relations_batch)} relations added.")
        final_relations = tentative_relations_batch
        final_triplets = tentative_triplets_batch
        final_entities_from_relations = tentative_entities_batch
    else:
        logger.info(f"Batch insertion exceeds budget: {relation_token_budget}. Falling back to one-by-one insertion.")
        # --- Fallback to one-by-one insertion (original logic) ---
        for rel in sorted_relations:
            triplets_for_rel = [st for st in all_scored_triplets if st[0][1] == rel]
            if not triplets_for_rel: continue

            new_entities_for_rel = {h for h, r, t in [st[0] for st in triplets_for_rel]} | \
                                   {t for h, r, t in [st[0] for st in triplets_for_rel]}
            
            # Tentatively add and check size
            tentative_relations = final_relations.union({rel})
            tentative_triplets = final_triplets + triplets_for_rel
            tentative_entities = final_entities_from_relations.union(new_entities_for_rel)

            temp_relations_section_list_data = []
            rel_map = defaultdict(list)
            for h, r, t in [st[0] for st in tentative_triplets]:
                if r in tentative_relations:
                    rel_map[r].append(h)
                    rel_map[r].append(t)

            for i, r_name in enumerate(sorted(list(tentative_relations))):
                related_nodes_in_context = {node for node in rel_map[r_name] if node in tentative_entities}
                related_nodes = "|".join(sorted(list(related_nodes_in_context)))
                temp_relations_section_list_data.append([str(i), r_name, related_nodes])
            
            temp_relations_section_list = [["id", "hyperedge", "related_entities"]] + temp_relations_section_list_data
            relations_context_csv = list_of_list_to_csv(temp_relations_section_list)
            relations_context_str = f"-----Relationships-----\n```csv\n{relations_context_csv}\n```"
            tokens = encode_string_by_tiktoken(relations_context_str, model_name=tiktoken_model_name)

            if len(tokens) <= relation_token_budget:
                final_relations.add(rel)
                final_triplets.extend(triplets_for_rel)
                final_entities_from_relations.update(new_entities_for_rel)
            else:
                logger.info(f"Relation '{rel}' exceeds budget. Stopping relation filling.")
                break
    
    if not final_relations:
        logger.warning("No relations could be added within the token budget.")
        return None, None, None, None

    # Fetch details for all entities that made it in
    final_entity_details_list = await asyncio.gather(*[knowledge_graph_inst.get_node(name) for name in final_entities_from_relations])
    final_entity_details = {name: data for name, data in zip(final_entities_from_relations, final_entity_details_list) if data}

    # Build final relation context and calculate used tokens
    relations_section_list_data = []
    rel_map = defaultdict(list)
    for h, r, t in [st[0] for st in final_triplets]:
        if r in final_relations:
            rel_map[r].append(h)
            rel_map[r].append(t)

    for i, r_name in enumerate(sorted(list(final_relations))):
        related_nodes_in_context = {node for node in rel_map[r_name] if node in final_entities_from_relations}
        related_nodes = "|".join(sorted(list(related_nodes_in_context)))
        relations_section_list_data.append([str(i), r_name, related_nodes])
    
    relations_section_list = [["id", "hyperedge", "related_entities"]] + relations_section_list_data
    final_relations_context_csv = list_of_list_to_csv(relations_section_list)
    relations_context_str = f"-----Relationships-----\n```csv\n{final_relations_context_csv}\n```"
    relations_tokens_used = len(encode_string_by_tiktoken(relations_context_str, model_name=tiktoken_model_name))
    entity_budget_spillover = max(0, relation_token_budget - relations_tokens_used)

    logger.info(f"Phase 1 complete. Added {len(final_relations)} relations. Tokens used: {relations_tokens_used}/{relation_token_budget}. Spillover to entities: {entity_budget_spillover}")

    # 5. Phase 2: Fill Entities
    entity_total_budget = entity_token_budget + entity_budget_spillover
    logger.info(f"--- Phase 2: Filling Entities (Budget: {entity_total_budget} tokens) ---")
    
    # Sort entities: Primary by score (desc), Secondary by connectivity (asc)
    sorted_entities_for_context = sorted(
        list(final_entities_from_relations),
        key=lambda e: (entity_scores.get(e, 0.0), -entity_connectivity.get(e, 99999)),
        reverse=True
    )

    final_entities_for_context = []
    final_entities_context_csv = "id,entity,type,description" # Header
    
    # --- Batch insertion attempt ---
    logger.info("Attempting to add all entities in a batch.")
    tentative_final_entities_batch = sorted_entities_for_context

    entites_section_list_data_batch = []
    for i, name in enumerate(tentative_final_entities_batch):
        details = final_entity_details.get(name, {})
        desc = details.get("description", "UNKNOWN")
        desc = desc if "<SEP>" in desc else desc.strip('"')
        entites_section_list_data_batch.append(
            [str(i), name, details.get("entity_type", "UNKNOWN"), desc]
        )
    entites_section_list_batch = [["id", "entity", "type", "description"]] + entites_section_list_data_batch
    temp_entities_context_csv_batch = list_of_list_to_csv(entites_section_list_batch)
    entities_context_str_batch = f"-----Entities-----\n```csv\n{temp_entities_context_csv_batch}\n```"
    entity_tokens_used_batch = len(encode_string_by_tiktoken(entities_context_str_batch, model_name=tiktoken_model_name))

    if entity_tokens_used_batch <= entity_total_budget:
        logger.info(f"Batch insertion successful. All {len(tentative_final_entities_batch)} entities added.")
        final_entities_for_context = tentative_final_entities_batch
        final_entities_context_csv = temp_entities_context_csv_batch
    else:
        logger.info(f"Batch insertion exceeds budget ({entity_tokens_used_batch} > {entity_total_budget}). Falling back to one-by-one insertion.")
        # --- Fallback to one-by-one insertion (original logic) ---
        for entity_name in sorted_entities_for_context:
            tentative_final_entities = final_entities_for_context + [entity_name]
            
            entites_section_list_data = []
            for i, name in enumerate(tentative_final_entities):
                details = final_entity_details.get(name, {})
                desc = details.get("description", "UNKNOWN")
                desc = desc if "<SEP>" in desc else desc.strip('"')
                entites_section_list_data.append(
                    [str(i), name, details.get("entity_type", "UNKNOWN"), desc]
                )
            entites_section_list = [["id", "entity", "type", "description"]] + entites_section_list_data
            temp_entities_context_csv = list_of_list_to_csv(entites_section_list)
            entities_context_str = f"-----Entities-----\n```csv\n{temp_entities_context_csv}\n```"
            
            entity_tokens_used = len(encode_string_by_tiktoken(entities_context_str, model_name=tiktoken_model_name))

            if entity_tokens_used <= entity_total_budget:
                final_entities_for_context.append(entity_name)
                final_entities_context_csv = temp_entities_context_csv
            else:
                logger.info(f"Entity '{entity_name}' exceeds budget. Stopping entity filling.")
                break
    
    entities_context_str = f"-----Entities-----\n```csv\n{final_entities_context_csv}\n```"
    entity_tokens_used = len(encode_string_by_tiktoken(entities_context_str, model_name=tiktoken_model_name))
    source_budget_spillover = max(0, entity_total_budget - entity_tokens_used)

    logger.info(f"Phase 2 complete. Added {len(final_entities_for_context)} entities. Tokens used: {entity_tokens_used}/{entity_total_budget}. Spillover to sources: {source_budget_spillover}")

    # 6. Phase 3: Fill Sources & Final Assembly
    source_total_budget = source_token_budget + source_budget_spillover
    logger.info(f"--- Phase 3: Filling Sources (Budget: {source_total_budget} tokens) ---")

    # First, assemble the core context string.
    final_core_context = f"""{entities_context_str}
{relations_context_str}"""

    # Now, get all the source chunks, in the correct sorted order.
    source_ids = set()
    source_scores = defaultdict(float)
    source_item_counts = defaultdict(int)

    relation_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(name) for name in final_relations])
    # Use sorted list of final_relations to ensure consistent ordering
    for rel_data, rel_name in zip(relation_datas, sorted(list(final_relations))):
        if rel_data and "source_id" in rel_data:
            ids = split_string_by_multi_markers(rel_data["source_id"], [GRAPH_FIELD_SEP])
            for sid in ids:
                if sid:
                    source_ids.add(sid)
                    source_scores[sid] += relation_max_scores.get(rel_name, 0)
                    source_item_counts[sid] += 1

    for ent_name in final_entities_for_context:
        ent_data = final_entity_details.get(ent_name, {})
        if ent_data and "source_id" in ent_data:
            ids = split_string_by_multi_markers(ent_data["source_id"], [GRAPH_FIELD_SEP])
            for sid in ids:
                if sid:
                    source_ids.add(sid)
                    source_scores[sid] += entity_scores.get(ent_name, 0)
                    source_item_counts[sid] += 1

    sorted_source_ids = sorted(
        list(source_ids),
        key=lambda sid: (source_scores.get(sid, 0.0), source_item_counts.get(sid, 0)),
        reverse=True
    )

    # Now, build the untruncated sources section using these sorted IDs.
    text_chunk_data = await asyncio.gather(*[text_chunks_db.get_by_id(sid) for sid in sorted_source_ids])
    
    text_units_section_list_data = []
    seen_content = set()
    # The order is preserved here because we iterate over text_chunk_data which was fetched from sorted_source_ids.
    for i, chunk in enumerate(filter(None, text_chunk_data)):
        content = chunk.get('content', 'N/A')
        if content not in seen_content:
            text_units_section_list_data.append([f"{i}", content])
            seen_content.add(content)

    untruncated_sources_context_csv = ""
    if text_units_section_list_data:
        text_units_section_list = [["id", "content"]] + text_units_section_list_data
        untruncated_sources_context_csv = list_of_list_to_csv(text_units_section_list)

    # Combine core context with untruncated sources to check total size
    untruncated_sources_context = ""
    if untruncated_sources_context_csv:
        untruncated_sources_context = f"""
-----Sources-----
```csv
{untruncated_sources_context_csv}
```"""
    
    full_context_untruncated = final_core_context + untruncated_sources_context
    full_tokens = encode_string_by_tiktoken(full_context_untruncated, model_name=tiktoken_model_name)

    # Now, check if truncation is needed.
    if len(full_tokens) <= max_context_tokens:
        final_context = full_context_untruncated
    else:
        logger.warning(f"Final context token count ({len(full_tokens)}) exceeds limit ({max_context_tokens}). Truncating sources.")
        
        core_tokens_len = len(encode_string_by_tiktoken(final_core_context, model_name=tiktoken_model_name))
        # The budget for sources is the total allowed minus what the core context already took.
        source_token_budget = max_context_tokens - core_tokens_len

        if source_token_budget <= 0:
            logger.warning("No token budget remaining for sources. Returning core context only.")
            final_context = final_core_context
        else:
            # Truncate the sources CSV we built. This logic is from operate_old2.py.
            csv_lines = untruncated_sources_context_csv.split('\n')
            csv_header = csv_lines[0]
            csv_data_lines = csv_lines[1:]

            truncated_csv_data_lines = []
            # The shell includes the core context, which is already accounted for in the budget.
            # We just need to calculate the token cost of the source section shell + each line.
            source_shell_str = f"""
-----Sources-----
```csv
{csv_header}
```"""
            tokens_so_far = len(encode_string_by_tiktoken(source_shell_str, model_name=tiktoken_model_name))

            for line in csv_data_lines:
                # Add 1 for the newline character that was split on
                line_token_cost = len(encode_string_by_tiktoken(line + '\n', model_name=tiktoken_model_name))
                if tokens_so_far + line_token_cost > source_token_budget:
                    logger.info(f"Source section truncated. Used {tokens_so_far}/{source_token_budget} tokens.")
                    break
                
                truncated_csv_data_lines.append(line)
                tokens_so_far += line_token_cost
            
            if truncated_csv_data_lines:
                final_sources_csv = csv_header + '\n' + '\n'.join(truncated_csv_data_lines)
                final_sources_context = f"""
-----Sources-----
```csv
{final_sources_csv}
```"""
                final_context = final_core_context + final_sources_context
            else:
                final_context = final_core_context

    final_tokens = len(encode_string_by_tiktoken(final_context, model_name=tiktoken_model_name))
    logger.info(f"Final context assembled. Total tokens: {final_tokens}/{max_context_tokens}")

    context_token_count = final_tokens
    entities_count = len(final_entities_for_context)

    return final_context, context_token_count, entities_count, retrieve_time

# --- End of new retrieval logic functions ---


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
        ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_relation = await _handle_single_hyperrelation_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(
                    if_relation
                )
                now_hyper_relation = if_relation["hyper_relation"]
                
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
            
    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any hyperedges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    embedding_map: Dict[str, torch.Tensor] = None,
    model: MLP = None,
    full_graph: nx.Graph = None,
    dde_encoder: DDEEncoder = None,
    hashing_kv: BaseKVStorage = None,
    entity_connectivity: Dict[str, int] = None,
    relation_connectivity: Dict[str, int] = None,
    graph_density: float = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response, None, None, None
    
    # --- Start of new retrieval logic ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = global_config["addon_params"].get("domain", "art")
    k_hops = 3 # This should be consistent with DDEEncoder initialization

    # Load pre-computed assets if they are not provided, with a warning.
    if embedding_map is None:
        logger.warning("Embedding map not pre-loaded, loading now. For better performance, pre-load during initialization.")
        embedding_map = _load_graph_embeddings(domain, device)
    
    if model is None:
        logger.warning("MLP model not pre-loaded, loading now. For better performance, pre-load during initialization.")
        model_path = f"../expr/wikitopics/{domain}/train/best_retrieval_model.pth"
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            pred_in_size = checkpoint['pred_in_size']
            emb_size = checkpoint['emb_size']
            model = MLP(pred_in_size=pred_in_size, emb_size=emb_size).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            logger.error(f"MLP model file not found at {model_path}. Aborting.")
            return PROMPTS["fail_response"]

    if full_graph is None:
        logger.warning("Full graph not pre-loaded, loading now. For better performance, pre-load during initialization.")
        graph_path = f"../expr/wikitopics/{domain}/graph_chunk_entity_relation.graphml"
        if Path(graph_path).exists():
            full_graph = nx.read_graphml(graph_path)
        else:
            logger.error(f"GraphML file not found at {graph_path}. Aborting.")
            return PROMPTS["fail_response"]

    if dde_encoder is None:
        logger.warning("DDEEncoder not pre-loaded, creating now. For better performance, pre-load during initialization.")
        dde_encoder = DDEEncoder(max_hops=k_hops, device=device)
    # --- End of new retrieval logic ---

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )
    
    hint_prompt = entity_extract_prompt.format(
        **context_base, input_text="{input_text}"
    ).format(**context_base, input_text=query)

    final_result = await use_model_func(hint_prompt)

    logger.info("kw_prompt result:")
    print(final_result)
    topic_entities = []
    try:
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                continue
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                topic_entities.append(clean_str(record_attributes[1]).upper())
            else:
                continue
    # Handle parsing error
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {final_result}")
        return PROMPTS["fail_response"], None, None, None

    # Handle keywords missing
    if not topic_entities:
        logger.warning("No topic entities extracted from the query.")
        return PROMPTS["fail_response"], None, None, None
    
    logger.info(f"Extracted Topic Entities: {topic_entities}")

    # Build context using the new MLP-based retriever
    context, context_token_count, entities_count, retrieve_time = await _build_query_context(
        query,
        topic_entities,
        knowledge_graph_inst,
        text_chunks_db,
        global_config,
        embedding_map=embedding_map,
        model=model,
        full_graph=full_graph,
        dde_encoder=dde_encoder,
        entity_connectivity=entity_connectivity,
        relation_connectivity=relation_connectivity,
        graph_density=graph_density,
    )
    
    if query_param.only_need_context:
        return context
    if context is None:
        logger.warning("Failed to build context using the new retriever.")
        return PROMPTS["fail_response"], None, None, None
    # with open("context.txt", "w", encoding="utf-8") as f:
    #     f.write(context)
    
    sys_prompt_temp = PROMPTS["rag_response_entities"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response, context_token_count, entities_count, retrieve_time