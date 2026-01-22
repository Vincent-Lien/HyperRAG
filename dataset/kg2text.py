import os
import argparse
from collections import defaultdict
from vllm import LLM, SamplingParams
import time
from tqdm import tqdm
import shutil
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

def parse_triples(filepath):
    """Parse tab-separated triples, skipping header."""
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    # Skip header if present
    if lines and lines[0].split('\t')[0].lower() == 'head':
        lines = lines[1:]
    for line in lines:
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        head, relation, tail = parts
        triples.append((head, relation, tail))
    return triples

def group_by_head(triples):
    """Group triples by their head."""
    grouped = defaultdict(list)
    for head, rel, tail in triples:
        grouped[head].append((rel, tail))
    return grouped

def format_chat_prompt(system_prompt, user_prompt):
    """Format prompt for Mistral-Small-3.1-24B-Instruct-2503 chat format."""
    return f"<s>[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]"

def call_stage1_batch(triples_data):
    """Stage 1: Convert individual triples to base sentences using batch processing."""
    sys_prompt = "You are a linguistic robot that translates messages in the form of triples into text. You may only return a single sentence and you can't use semicolons as part of your answer."
    
    prompts = []
    for head, rel, tail in triples_data:
        user_prompt = f"Please convert this triple into a single sentence. Do not insert any other information or commentary.\nTriple: {head}\t{rel}\t{tail}"
        full_prompt = format_chat_prompt(sys_prompt, user_prompt)
        prompts.append(full_prompt)
    
    # Generate responses in batch
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        # Clean up the response - take only the first sentence
        sentences = generated_text.split('.')
        if sentences[0]:
            results.append(sentences[0].strip() + '.')
        else:
            results.append(generated_text)
    
    return results

def call_stage2_batch(base_sentences, grouped_triples):
    """Stage 2: Insert additional triples into base sentences using batch processing."""
    sys_prompt = "You are a linguistic robot that translates messages in the form of triples into text. You may only return a single sentence and you can't use semicolons as part of your answer."
    
    prompts = []
    for base_sentence, triples in zip(base_sentences, grouped_triples):
        if len(triples) <= 1:
            # No additional triples to insert
            prompts.append(None)
            continue
            
        # Build the list of additional triples (skip the first one as it's already in base_sentence)
        extras = [f"{h}\t{r}\t{t}" for h, r, t in triples[1:]]
        user_prompt = (
            f"Now insert all of the following triples into this sentence: {base_sentence}. "
            "Keep the length as short as possible. Do not insert any other information or commentary than these triples and the previous triple I gave you:\n"
            + "\n".join(extras)
        )
        full_prompt = format_chat_prompt(sys_prompt, user_prompt)
        prompts.append(full_prompt)
    
    # Filter out None prompts and keep track of indices
    valid_prompts = []
    valid_indices = []
    for i, prompt in enumerate(prompts):
        if prompt is not None:
            valid_prompts.append(prompt)
            valid_indices.append(i)
    
    if not valid_prompts:
        return base_sentences
    
    # Generate responses for valid prompts
    outputs = llm.generate(valid_prompts, sampling_params)
    
    # Build final results
    results = base_sentences.copy()
    for i, output in enumerate(outputs):
        original_index = valid_indices[i]
        generated_text = output.outputs[0].text.strip()
        # Clean up the response
        sentences = generated_text.split('.')
        if sentences[0]:
            results[original_index] = sentences[0].strip() + '.'
        else:
            results[original_index] = generated_text
    
    return results

def process_file(input_file, output_file):
    triples = parse_triples(input_file)
    groups = group_by_head(triples)
    
    print(f"Processing {len(groups)} groups of triples...")
    
    # Prepare data for batch processing
    first_triples = []
    grouped_triples_list = []
    heads = []
    
    for head, items in groups.items():
        heads.append(head)
        # First triple for stage 1
        rel0, tail0 = items[0]
        first_triples.append((head, rel0, tail0))
        # All triples for stage 2
        full_triples = [(head, rel, tail) for rel, tail in items]
        grouped_triples_list.append(full_triples)
    
    # Stage 1: Generate base sentences
    print("Stage 1: Generating base sentences...")
    base_sentences = call_stage1_batch(first_triples)
    
    # Stage 2: Insert additional triples
    print("Stage 2: Inserting additional triples...")
    final_sentences = call_stage2_batch(base_sentences, grouped_triples_list)
    
    # Write to output
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in final_sentences:
            f.write(sentence + '\n')

def main():
    parser = argparse.ArgumentParser(description="Convert triples to natural sentences in two stages using vLLM.")
    parser.add_argument('domain', help='Domain name corresponding to input folder')
    args = parser.parse_args()

    domain = args.domain

    print(f"\nProcessing domain: {domain}")

    dataset_dir = f"WikiTopicsQE_decoded/{domain}"
    output_dir = f"WikiTopicsQE_NLG/{domain}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    graph_files = ["train_graph.txt", "test_inference.txt"]
    for graph_file in graph_files:
        print(f"Processing file: {graph_file}")
        input_file = os.path.join(dataset_dir, graph_file)
        if "train" in graph_file:
            output_file = os.path.join(output_dir, "train_sentences.txt")
        else:
            output_file = os.path.join(output_dir, "test_sentences.txt")

        if os.path.exists(input_file):
            process_file(input_file, output_file)
            print(f"Converted sentences written to {output_file}")
        else:
            print(f"Input file not found: {input_file}")

    # Copy train_sentences.txt to val_sentences.txt
    train_file = os.path.join(output_dir, "train_sentences.txt")
    val_file = os.path.join(output_dir, "val_sentences.txt")
    if os.path.exists(train_file):
        shutil.copy2(train_file, val_file)
        print(f"Copied {train_file} to {val_file}")

if __name__ == '__main__':
    main()