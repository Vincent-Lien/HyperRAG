import os
import json
from pathlib import Path
from tqdm import tqdm
from hypergraphrag import HyperGraphRAG
import argparse

import time

if os.environ.get("OPENAI_API_KEY") is None:
    with open("../config_mine.json", "r") as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config["hypermemory_api_key"]

parser = argparse.ArgumentParser(description="Run HyperMemory on WikiTopics queries")
parser.add_argument("--domain", type=str, help="Domain to process (e.g. art, sci, etc.)")
args = parser.parse_args()

domain = args.domain
target = "('e', ('r', 'r', 'r'))"
# query_upper = 1000

dataset_dir = Path("../dataset/wikitopics_test_sampled") / domain
output_dir = Path("../results/HyperMemory/wikitopics_sampled") / domain
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{domain}_output.jsonl"

# queries
with open(dataset_dir / "test_queries.json", mode="r") as f:
    queries = json.load(f)[target]
# easy answers
with open(dataset_dir / "test_answers_easy.json", mode="r") as f:
    easy_answers = json.load(f)[target]
# hard answers
with open(dataset_dir / "test_answers_hard.json", mode="r") as f:
    hard_answers = json.load(f)[target]

rag = HyperGraphRAG(working_dir=f"../expr/wikitopics/{domain}", enable_llm_cache=False)

for query in tqdm(queries, desc=f"Processing {domain} queries"):
    response, context_token_count, entities_count, retrieve_time = rag.query(query)

    # Store metrics for each query
    if 'metrics' not in locals():
        metrics = []

    # Create a new metrics dictionary for each iteration
    metric_data = {
        "query": query,
        "context_token_count": context_token_count,
        "entity_count": entities_count,
        "elapsed_time": retrieve_time
    }

    # Directly append this new dictionary to the file
    with open(output_dir / f"{domain}_metrics.jsonl", "a", encoding="utf-8") as mf:
        json.dump(metric_data, mf, ensure_ascii=False)
        mf.write("\n")
    
    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        cleaned_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        cleaned_response = []
        print("Invalid JSON response")
    output = {
        "query": query,
        "response": cleaned_response,
        "easy_answer": easy_answers.get(query, []),
        "hard_answer": hard_answers.get(query, [])
    }
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")
