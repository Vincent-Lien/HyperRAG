import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from hypergraphrag import HyperGraphRAG

if os.environ.get("OPENAI_API_KEY") is None:
    with open("../config.json", "r") as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config.get("hyperretriever_api_key", "your_api_key_here")

parser = argparse.ArgumentParser(description="Run HyperRetriever on WikiTopics queries")
parser.add_argument("domain", type=str, help="Domain to process")
args = parser.parse_args()

domain = args.domain
target = "('e', ('r', 'r', 'r'))"

dataset_dir = Path("../dataset/wikitopics_test_sampled") / domain
output_dir = Path("../results/HyperRetriever/wikitopics_sampled")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{domain}_output.jsonl"

# queries
with open(dataset_dir / "test_queries.json", "r") as f:
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
        cleaned_response = []
    output = {
        "query": query,
        "response": cleaned_response,
        "easy_answer": easy_answers.get(query, []),
        "hard_answer": hard_answers.get(query, [])
    }
    with open(output_file, "a") as f:
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")
    # print(f"Processed query: {query}")
    # print(f"Response: {cleaned_response}")
    # for answer in cleaned_response:
    #     for easy_answer in easy_answers.get(query, []):
    #         if answer.lower() == easy_answer.lower():
    #             print(f"Easy answer found: {easy_answer}")
    # input("Press enter to continue...")  # Uncomment to stop after the first query