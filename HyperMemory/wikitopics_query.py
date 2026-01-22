import os
import json
from pathlib import Path
from tqdm import tqdm
from hypergraphrag import HyperGraphRAG
import argparse

# Set up openai api key
if os.environ.get("OPENAI_API_KEY") is None:
    with open("../config.json", "r") as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config.get("hypermemory_api_key", "your_openai_api_key_here")

parser = argparse.ArgumentParser(description="Run Hypergraph on WikiTopics queries")
parser.add_argument("domain", type=str, choices=['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax'],
                    help="Domain to process (e.g. art, sci, etc.)")
args = parser.parse_args()

domain = args.domain
target = "('e', ('r', 'r', 'r'))"

dataset_dir = Path("../dataset/WikiTopicsQE_NLG") / domain
output_dir = Path("../results/HyperMemory/wikitopics")
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
    response = rag.query(query)
    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        cleaned_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        cleaned_response = "Invalid JSON response"
    output = {
        "query": query,
        "response": cleaned_response,
        "easy_answer": easy_answers.get(query, []),
        "hard_answer": hard_answers.get(query, [])
    }
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")
