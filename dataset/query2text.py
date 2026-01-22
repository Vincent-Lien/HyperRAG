import os
import argparse
from collections import defaultdict
from vllm import LLM, SamplingParams
import time
from tqdm import tqdm
import shutil
import json
from huggingface_hub import login
login(token="your token here")  # Replace with your Hugging Face token

# Initialize vLLM model
llm = LLM(
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",  # 使用 Instruct 版本
    # quantization="bitsandbytes",  # 使用 bitsandbytes 進行量化
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,  # 使用 GPU 張數
    max_model_len=16384,  # 設定最大序列長度
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=200,
    top_p=0.9,
)

def build_prompt(query):
    """Format prompt for Mistral-Small-3.1-24B-Instruct-2503 chat format."""
    # --- SYSTEM PROMPTS ---
    SYSTEM_PROMPT_3HOP = """
    You are a specialized assistant for converting structured three-hop queries into natural language questions. A three-hop query follows the pattern: (start_entity, (relation1, relation2, relation3)), which represents a logical path to find target entities.

    Your task is to convert three-hop queries into clear, natural English questions that accurately capture the intended logical relationship without revealing intermediate entities.

    Key principles:
    1. Preserve the logical structure: The question must reflect the three-hop reasoning path
    2. Use only given information: Only use the start entity and the three relations provided
    3. No intermediate entities: Never mention specific intermediate entities that would appear in the actual query execution
    4. Express "shared property" relationships: Most three-hop queries seek entities connected through shared properties
    5. Natural language: Generate fluent, grammatically correct questions

    Three-hop logic pattern:
    - Start with the given entity
    - Relation 1: Find intermediate entities A connected to start entity
    - Relation 2: Find intermediate entities B connected to entities A
    - Relation 3: Find target entities connected to entities B
    - The goal is usually to find entities that share some property with the start entity through this path

    Response format: Provide only the natural language question. Do not include explanations, analysis, or alternative phrasings unless specifically requested.    
    """

    # --- USER PROMPT TEMPLATE ---
    USER_PROMPT_TEMPLATE = """
    Convert the following three-hop query into a natural language question:

    Query: {three_hop_query}

    Generate a clear, natural English question that captures the logical intent of this three-hop query.
    """

    system_prompt = SYSTEM_PROMPT_3HOP
    user_prompt = USER_PROMPT_TEMPLATE.format(three_hop_query=query)

    return f"<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]"

# 原版的prompt
def old_build_prompt(query):
    """Format prompt for Mistral-Small-3.1-24B-Instruct-2503 chat format."""
    # --- SYSTEM PROMPTS ---
    SYSTEM_PROMPT_3HOP = """
    You are a knowledgeable assistant that helps convert structured multi-hop knowledge graph queries into fluent and natural English questions.

    Each query follows a 3-hop structure:
    - A head entity (e.g., a person, place, or object),
    - Followed by a sequence of three relations, possibly including "inverse" directions.

    Your job:
    - Convert this structure into a natural, fluent, and **concise** English question.
    - The question must follow the logic of the 3-hop path exactly and preserve the direction of each relation (inverse = subject/object is reversed).
    - Keep the output question **short and focused** — ideally under **100 tokens**, and never more than **200 tokens**.
    - Avoid unnecessary elaboration or repetition.

    Guidelines:
    - Use plain, human-readable English.
    - Avoid long-winded or overly descriptive phrasing.
    - It's okay to omit trivial words if it keeps the meaning clear.
    """

    # --- USER PROMPT TEMPLATE ---
    USER_PROMPT_TEMPLATE = """
    Convert the following structured 3-hop query into a concise and natural English question: {query}
    """

    system_prompt = SYSTEM_PROMPT_3HOP
    user_prompt = USER_PROMPT_TEMPLATE.format(query=query)

    return f"<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]"

def call_batch(queries):
    """Call the language model in batches to process the queries."""

    prompts = [build_prompt(query) for query in queries]
    
    # Generate responses in batch
    outputs = llm.generate(prompts, sampling_params)
    
    results = [output.outputs[0].text.strip() for output in outputs]

    return results

def process_query_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)["('e', ('r', 'r', 'r'))"]
    
    # Prepare queries for processing
    print(f"Loaded {len(queries)} queries from {input_file}")
    query_sentences = call_batch(queries)

    # Write to output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"('e', ('r', 'r', 'r'))": query_sentences}, f, indent=2, ensure_ascii=False)

    return {query: query_sentence for query, query_sentence in zip(queries, query_sentences)}

def process_answer_file(input_file, output_file, query_pairs):
    with open(input_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)["('e', ('r', 'r', 'r'))"]
    
    # Prepare answers for processing
    print(f"Loaded {len(answers)} answers from {input_file}")

    answer_pairs = {}
    for query, query_sentence in query_pairs.items():
        answer_pairs[query_sentence] = answers[query]

    # Write to output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"('e', ('r', 'r', 'r'))": answer_pairs}, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Convert triples to natural sentences in two stages using vLLM.")
    parser.add_argument('domain', help='Domain name corresponding to input folder')
    args = parser.parse_args()

    domain = args.domain

    print(f"\nProcessing domain: {domain}")

    dataset_dir = f"WikiTopicsQE_decoded/{domain}"
    output_dir = f"WikiTopicsQE_NLG_new/{domain}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    query_files = ["train_queries.json", "valid_queries.json", "test_queries.json"]
    for query_file in query_files:
        print(f"Processing file: {query_file}")
        query_input_file = os.path.join(dataset_dir, query_file)
        base_name = query_file.split('_')[0] # Extract base name (train, valid, test)
        query_output_file = os.path.join(output_dir, f"{base_name}_queries.json")

        if os.path.exists(query_input_file):
            query_pairs = process_query_file(query_input_file, query_output_file)
            print(f"Converted sentences written to {query_output_file}")

            difficulties = ['easy', 'hard'] if base_name != 'train' else ['hard']
            for difficulty in difficulties:
                answer_input_file = os.path.join(dataset_dir, f"{base_name}_answers_{difficulty}.json")
                answer_output_file = os.path.join(output_dir, f"{base_name}_answers_{difficulty}.json")
                if os.path.exists(answer_input_file):
                    process_answer_file(answer_input_file, answer_output_file, query_pairs)
                    print(f"Converted answers written to {answer_output_file}")
                else:
                    print(f"Input file not found: {answer_input_file}")
        else:
            print(f"Input file not found: {query_input_file}")

if __name__ == '__main__':
    main()