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

parser = argparse.ArgumentParser(description="Run Hypergraph on open domain queries")
parser.add_argument('--dataset', type=str, required=True, choices=['2wikimultihopqa', 'hotpotqa', 'musique'], 
                    help='Dataset name, e.g., 2wikimultihopqa, hotpotqa, musique')
args = parser.parse_args()

dataset = args.dataset

dataset_dir = Path("../dataset/open_domain_sampled")
output_dir = Path("../results/HyperMemory/open_domain")
output_dir.mkdir(parents=True, exist_ok=True)
output_file_path = output_dir / f"{dataset}_output.jsonl"

# read queries in jsonl
with open(dataset_dir / f"{dataset}_query_test.jsonl", mode="r", encoding="utf-8") as f:
    queries = [json.loads(line)["question"] for line in f]


rag = HyperGraphRAG(working_dir=f"../expr/{dataset}", enable_llm_cache=False)

for query in tqdm(queries, desc=f"Processing {dataset} queries"):
    response = rag.query(query)

    # response parsing
    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        cleaned_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        cleaned_response = []
        print("Invalid JSON response")
    if len(cleaned_response) == 0:
        cleaned_response = ""
    else:
        cleaned_response = cleaned_response[0]

    # prepare output format
    output = {
        "query": query,
        "response": cleaned_response,
    }

    # write output to file
    with open(output_file_path, "a") as f:
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")
