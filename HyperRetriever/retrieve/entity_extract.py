import asyncio
import re
import os
import sys
import json
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from hypergraphrag.prompt import PROMPTS
from hypergraphrag.utils import clean_str, split_string_by_multi_markers
from hypergraphrag.llm import gpt_4o_mini_complete
from hypergraphrag.hypergraphrag import HyperGraphRAG

async def extract_entities_by_hypergraph(user_query: str, hgrag_instance: HyperGraphRAG):
    """
    Use the LLM function configured in the HyperGraphRAG instance to extract entities and hyper-relations from the query.
    """

    # Retrieve global_config and llm_model_func from the HyperGraphRAG instance
    global_config = asdict(hgrag_instance)
    use_llm_func = hgrag_instance.llm_model_func

    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])

    # Construct example context
    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = "\n".join(PROMPTS["entity_extraction_examples"]).format(**example_context_base)

    # Construct main prompt context
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )

    # Construct the final prompt
    entity_extract_prompt = PROMPTS["entity_extraction"]
    hint_prompt = entity_extract_prompt.format(
        **context_base, input_text="{input_text}"
    ).format(input_text=user_query)

    # Call the LLM function configured in the HyperGraphRAG instance
    final_result = await use_llm_func(hint_prompt)

    ll_keywords, hl_keywords = [], []
    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )

    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match:
            record = match.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                hl_keywords.append("<hyperedge>"+clean_str(record_attributes[1]))
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                ll_keywords.append(clean_str(record_attributes[1]).upper())
            else:
                continue

    return ll_keywords

async def aextract_entities_single(query: str, hgrag_instance: HyperGraphRAG):
    """
    Use the LLM function configured in the HyperGraphRAG instance to extract entities and hyper-relations from a single query.
    """
    entities = await extract_entities_by_hypergraph(query, hgrag_instance)
    return entities

def extract_entities(domain: str, query_list: list[str], enable_llm_cache: bool = True):
    """
    Extract entities from the given list of queries.
    """
    working_dir_for_hgrag = Path("../expr/wikitopics") / domain / "train"
    working_dir_for_hgrag.mkdir(parents=True, exist_ok=True)
    cache_file_path = working_dir_for_hgrag / "extracted_entities_cache.json"

    cached_data = {}
    if enable_llm_cache and cache_file_path.exists():
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from cache file {cache_file_path}. Starting with empty cache.")
            cached_data = {}

    unprocessed_queries = []
    processed_results = {}

    for query in query_list:
        if enable_llm_cache and query in cached_data:
            processed_results[query] = cached_data[query]
        else:
            unprocessed_queries.append(query)

    if not unprocessed_queries:
        print("All queries already processed and cached. No new extraction needed.")
        return processed_results

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Instantiate HyperGraphRAG here only if there are unprocessed queries
    hgrag_instance = HyperGraphRAG(
        working_dir=working_dir_for_hgrag,
        llm_model_func=gpt_4o_mini_complete, # use gpt-4o-mini
        llm_model_name="gpt-4o-mini"
    )

    for query in tqdm(unprocessed_queries, desc="Extracting entities"):
        # Use asyncio event loop to run the coroutine for extracting entities
        newly_extracted_entities = loop.run_until_complete(aextract_entities_single(query, hgrag_instance))
        cached_data[query] = newly_extracted_entities

        # Update cache with newly extracted data after each query
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=2)

    # Combine processed and newly extracted data
    processed_results.update(cached_data)
    return processed_results