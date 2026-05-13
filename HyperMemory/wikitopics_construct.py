import os
import json
import argparse
from pathlib import Path

from hypergraphrag import HyperGraphRAG

# Set up openai api key
if os.environ.get("OPENAI_API_KEY") is None:
    with open("../config.json", "r") as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config.get("openai_api_key")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Construct Hypergraph for WikiTopics.")
parser.add_argument("domain", type=str, choices=['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax'],
                    help="Domain to construct Hypergraph for.")
args = parser.parse_args()

domain = args.domain
dataset_dir = Path(f"../dataset/WikiTopicsQE_NLG") / domain

train_document_path = dataset_dir / "train_sentences.txt"
test_document_path = dataset_dir / "test_sentences.txt"

rag = HyperGraphRAG(working_dir=f"../expr/wikitopics/{domain}")

with open(train_document_path, mode="r") as f:
    train_contexts = [train_context.strip() for train_context in f.readlines()]
with open(test_document_path, mode="r") as f:
    test_contexts = [test_context.strip() for test_context in f.readlines()]
unique_contexts = list(set(train_contexts + test_contexts))

rag.insert(unique_contexts)