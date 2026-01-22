import json
import string
import re
from typing import List
from datetime import datetime
import argparse
from pathlib import Path

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_hits_at_k(gold_answers: List[List[str]], pred_ranked_list: List[List[str]], k: int = 10) -> float:
    """Computes Hits@k."""
    assert len(gold_answers) == len(pred_ranked_list)
    hits = 0
    for gold, preds in zip(gold_answers, pred_ranked_list):
        norm_gold = set([normalize_answer(g) for g in gold])
        norm_preds = [normalize_answer(p) for p in preds[:k]]
        if any(p in norm_gold for p in norm_preds):
            hits += 1
    return 100.0 * hits / len(gold_answers)

def compute_mrr(gold_answers: List[List[str]], pred_ranked_list: List[List[str]]) -> float:
    """Computes Mean Reciprocal Rank (MRR)."""
    assert len(gold_answers) == len(pred_ranked_list)
    total_rr = 0.0
    for gold, preds in zip(gold_answers, pred_ranked_list):
        norm_gold = set([normalize_answer(g) for g in gold])
        norm_preds = [normalize_answer(p) for p in preds]
        rr = 0.0
        for i, p in enumerate(norm_preds):
            if p in norm_gold:
                rr = 1.0 / (i + 1)
                break
        total_rr += rr
    return 100.0 * total_rr / len(gold_answers)

def evaluate_qa(gold_answers: List[List[str] | str], pred_answers: List[str | List[str]]) -> dict:
    assert len(gold_answers) == len(pred_answers)

    # For Hits@k & MRR, require list-of-lists
    ranked_preds = [p if isinstance(p, list) else [p] for p in pred_answers]

    # hits_1 = compute_hits_at_k(gold_answers, ranked_preds, k=1)
    hits_10 = compute_hits_at_k(gold_answers, ranked_preds, k=10)
    mrr = compute_mrr(gold_answers, ranked_preds)

    return {
        "MRR": mrr,
        "Hits@10": hits_10
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA results for Wikitopics dataset.")
    parser.add_argument('--model', type=str, help='Name of the model to evaluate (HyperRetriever, HyperMemory, etc.)')
    parser.add_argument('--sampled', action='store_true', help='Use sampled dataset')
    parser.add_argument('domain', type=str, nargs='?', default='all', help='Domain to evaluate on (default: all)')
    args = parser.parse_args()

    model_name = args.model
    result_dir = f"results/{model_name}/wikitopics_sampled" if args.sampled else f"results/{model_name}/wikitopics"
    
    if args.domain != 'all':
        domains = [args.domain]
    else:
        domains = ['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax']

    eval_log_dir = Path(f"results/log/{model_name}")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_result_path = f"results/log/{model_name}/qa_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    eval_result = {}

    for domain in domains:
        output_path = f"{result_dir}/{domain}_output.jsonl"
        
        easy_answers = []
        hard_answers = []
        pred_answers_for_easy = []
        pred_answers_for_hard = []

        try:
            with open(output_path, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        easy_answer = data.get('easy_answer', [])
                        hard_answer = data.get('hard_answer', [])
                        pred_answer = data.get('response', [])

                        # assume predictions are strings or ranked list of strings
                        pred = pred_answer if isinstance(pred_answer, list) else [pred_answer]

                        easy_answers.append(easy_answer)
                        pred_answers_for_easy.append(pred)

                        hard_answers.append(hard_answer)
                        pred_answers_for_hard.append(pred)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in {output_path}, skipping line.")
                        continue
        except FileNotFoundError:
            print(f"Error: File {output_path} not found, skipping domain {domain}.")
            continue

        eval_result["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        eval_result["model"] = model_name
        eval_result["data"] = "full dataset" if not args.sampled else "sampled dataset"

        print(f"Domain: {domain}")
        eval_result[domain] = {}
        if easy_answers:
            easy_scores = evaluate_qa(easy_answers, pred_answers_for_easy)
            for key, value in easy_scores.items():
                print(f"Easy - {key}: {value:.2f}%")
            eval_result[domain]['easy'] = easy_scores
        else:
            print("Easy - No valid answers to evaluate.")
            eval_result[domain]['easy'] = {}

        print("")
        if hard_answers:
            hard_scores = evaluate_qa(hard_answers, pred_answers_for_hard)
            for key, value in hard_scores.items():
                print(f"Hard - {key}: {value:.2f}%")
            eval_result[domain]['hard'] = hard_scores
        else:
            print("Hard - No valid answers to evaluate.")
            eval_result[domain]['hard'] = {}
        print("")
        print("-" * 40)

    # Save evaluation results to a JSON file
    with open(eval_result_path, 'w') as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)