import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from hypergraphrag import HyperGraphRAG


if os.environ.get("OPENAI_API_KEY") is None:
    with open("../config.json", "r") as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

parser = argparse.ArgumentParser(description="Run HyperRetriever on WikiTopics queries")
parser.add_argument("domain", type=str, choices=["2wikimultihopqa", "hotpotqa", "musique"], 
                    help="Domain to run the queries on")
args = parser.parse_args()

domain = args.domain
target = "('e', ('r', 'r', 'r'))"

dataset_dir = Path("../dataset/open_domain_splitted")
output_dir = Path("../results/HyperRetriever/open_domain") / domain
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"test_results_cleaned.jsonl"

# queries
with open(dataset_dir / f"{domain}_query_test.jsonl", mode="r") as f:
    items = [json.loads(line) for line in f]
    queries = [item['question'] for item in items]
    easy_answers = {item['question']: item['answer'] for item in items}

rag = HyperGraphRAG(working_dir=f"../expr/{domain}", enable_llm_cache=False)

for query in tqdm(queries, desc=f"Processing {domain} queries"):
    response = rag.query(query)
    cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        cleaned_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        cleaned_response = ""
        print("Invalid JSON Error")
    if len(cleaned_response) == 0:
        cleaned_response = ""
    else:
        cleaned_response = cleaned_response[0]

    output = {
        "question": query,
        "output": cleaned_response,
    }
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")
