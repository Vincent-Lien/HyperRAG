import json
import string
import re
import numpy as np
from typing import List, Tuple
from datetime import datetime
import argparse
from pathlib import Path
from scipy import stats

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

def compute_hits_at_k_per_sample(gold_answers: List[List[str]], pred_ranked_list: List[List[str]], k: int = 10) -> np.ndarray:
    """Computes Hits@k for each sample, returns array of 0s and 1s."""
    assert len(gold_answers) == len(pred_ranked_list)
    hits = []
    for gold, preds in zip(gold_answers, pred_ranked_list):
        norm_gold = set([normalize_answer(g) for g in gold])
        norm_preds = [normalize_answer(p) for p in preds[:k]]
        if any(p in norm_gold for p in norm_preds):
            hits.append(1.0)
        else:
            hits.append(0.0)
    return np.array(hits)

def compute_mrr_per_sample(gold_answers: List[List[str]], pred_ranked_list: List[List[str]]) -> np.ndarray:
    """Computes MRR for each sample."""
    assert len(gold_answers) == len(pred_ranked_list)
    rr_scores = []
    for gold, preds in zip(gold_answers, pred_ranked_list):
        norm_gold = set([normalize_answer(g) for g in gold])
        norm_preds = [normalize_answer(p) for p in preds]
        rr = 0.0
        for i, p in enumerate(norm_preds):
            if p in norm_gold:
                rr = 1.0 / (i + 1)
                break
        rr_scores.append(rr)
    return np.array(rr_scores)

def get_significance_level(p_value: float) -> str:
    """
    Determine significance level based on p-value.
    Returns: 'extremely significant', 'highly significant', 'significant', 
             'marginally significant', or 'not significant'
    """
    if p_value < 0.001:
        return 'extremely significant'
    elif p_value < 0.01:
        return 'highly significant'
    elif p_value < 0.05:
        return 'significant'
    elif p_value < 0.1:
        return 'marginally significant'
    else:
        return 'not significant'

def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> dict:
    """Perform paired t-test."""
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    mean_diff = float(np.mean(scores1) - np.mean(scores2))
    
    return {
        'mean_diff': mean_diff * 100,  # Convert to percentage
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significance_level': get_significance_level(float(p_value))
    }

def load_predictions(output_path: str) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """Load predictions and gold answers from output file."""
    easy_answers = []
    hard_answers = []
    pred_answers = []
    
    try:
        with open(output_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    easy_answer = data.get('easy_answer', [])
                    hard_answer = data.get('hard_answer', [])
                    pred_answer = data.get('response', [])
                    
                    # Convert predictions to list format
                    pred = pred_answer if isinstance(pred_answer, list) else [pred_answer]
                    
                    easy_answers.append(easy_answer)
                    hard_answers.append(hard_answer)
                    pred_answers.append(pred)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in {output_path}, skipping line.")
                    continue
    except FileNotFoundError:
        print(f"Error: File {output_path} not found.")
        return [], [], []
    
    return easy_answers, hard_answers, pred_answers

def evaluate_significance_all_domains(model1: str, model2: str, domains: List[str], 
                                      sampled: bool = False, difficulty: str = 'easy') -> dict:
    """
    Compare two models across all domains combined.
    """
    result_dir_suffix = "wikitopics_sampled" if sampled else "wikitopics"
    
    # Collect all data across domains
    all_gold1 = []
    all_gold2 = []
    all_pred1 = []
    all_pred2 = []
    domain_sample_counts = {}
    
    for domain in domains:
        model1_path = f"results/{model1}/{result_dir_suffix}/{domain}_output.jsonl"
        model2_path = f"results/{model2}/{result_dir_suffix}/{domain}_output.jsonl"
        
        # Load data for both models
        easy1, hard1, pred1 = load_predictions(model1_path)
        easy2, hard2, pred2 = load_predictions(model2_path)
        
        if not easy1 or not easy2:
            print(f"Warning: Failed to load data for domain {domain}, skipping")
            continue
        
        # Select difficulty level
        if difficulty == 'easy':
            gold1, gold2 = easy1, easy2
        else:  # hard
            gold1, gold2 = hard1, hard2
        
        # Ensure same number of samples for this domain
        min_samples = min(len(pred1), len(pred2))
        domain_sample_counts[domain] = min_samples
        
        # Append to combined lists
        all_gold1.extend(gold1[:min_samples])
        all_gold2.extend(gold2[:min_samples])
        all_pred1.extend(pred1[:min_samples])
        all_pred2.extend(pred2[:min_samples])
    
    if not all_gold1:
        return {
            'error': 'No valid data loaded for any domain'
        }
    
    # Compute per-sample scores for all combined data
    mrr1 = compute_mrr_per_sample(all_gold1, all_pred1)
    mrr2 = compute_mrr_per_sample(all_gold2, all_pred2)
    
    hits1 = compute_hits_at_k_per_sample(all_gold1, all_pred1, k=10)
    hits2 = compute_hits_at_k_per_sample(all_gold2, all_pred2, k=10)
    
    # Statistical tests
    mrr_ttest = paired_t_test(mrr1, mrr2)
    hits_ttest = paired_t_test(hits1, hits2)
    
    results = {
        'difficulty': difficulty,
        'total_samples': int(len(all_pred1)),
        'domain_sample_counts': domain_sample_counts,
        'model1': model1,
        'model2': model2,
        'model1_mrr_mean': float(np.mean(mrr1) * 100),
        'model2_mrr_mean': float(np.mean(mrr2) * 100),
        'model1_hits10_mean': float(np.mean(hits1) * 100),
        'model2_hits10_mean': float(np.mean(hits2) * 100),
        'MRR': mrr_ttest,
        'Hits@10': hits_ttest
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical significance testing for QA results.")
    parser.add_argument('--model1', type=str, required=True, help='First model name')
    parser.add_argument('--model2', type=str, required=True, help='Second model name')
    parser.add_argument('--sampled', action='store_true', help='Use sampled dataset')
    parser.add_argument('--difficulty', type=str, default='both', choices=['easy', 'hard', 'both'],
                       help='Difficulty level to test (default: both)')
    args = parser.parse_args()
    
    domains = ['art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax']
    difficulties = ['easy', 'hard'] if args.difficulty == 'both' else [args.difficulty]
    
    # Create log directory
    eval_log_dir = Path(f"results/log/significance")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_result_path = f"results/log/significance/stat_test_{args.model1}_vs_{args.model2}_{timestamp}.json"
    
    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model1': args.model1,
        'model2': args.model2,
        'dataset': 'sampled' if args.sampled else 'full',
        'results': {}
    }
    
    print(f"Comparing {args.model1} vs {args.model2}")
    print(f"Dataset: {'sampled' if args.sampled else 'full'}")
    print(f"Domains: {', '.join(domains)}")
    print("=" * 80)
    
    for difficulty in difficulties:
        print(f"\nDifficulty: {difficulty}")
        print("-" * 80)
        
        results = evaluate_significance_all_domains(
            args.model1, args.model2, domains,
            sampled=args.sampled,
            difficulty=difficulty
        )
        
        if 'error' in results:
            print(f"  {results['error']}")
            all_results['results'][difficulty] = results
            continue
        
        all_results['results'][difficulty] = results
        
        # Print results
        print(f"\nTotal samples: {results['total_samples']}")
        print(f"Samples per domain: {results['domain_sample_counts']}")
        print(f"\n{args.model1} MRR: {results['model1_mrr_mean']:.2f}%")
        print(f"{args.model2} MRR: {results['model2_mrr_mean']:.2f}%")
        print(f"{args.model1} Hits@10: {results['model1_hits10_mean']:.2f}%")
        print(f"{args.model2} Hits@10: {results['model2_hits10_mean']:.2f}%")
        
        # MRR significance
        mrr_test = results['MRR']
        print(f"\nMRR Difference: {mrr_test['mean_diff']:.2f}%")
        print(f"  t-statistic: {mrr_test['t_statistic']:.4f}")
        print(f"  p-value: {mrr_test['p_value']:.4f} ({mrr_test['significance_level']})")
        
        # Hits@10 significance
        hits_test = results['Hits@10']
        print(f"\nHits@10 Difference: {hits_test['mean_diff']:.2f}%")
        print(f"  t-statistic: {hits_test['t_statistic']:.4f}")
        print(f"  p-value: {hits_test['p_value']:.4f} ({hits_test['significance_level']})")
        
        print("\n" + "=" * 80)
    
    # Save results
    with open(eval_result_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {eval_result_path}")
    print("\nSignificance levels:")
    print("  p < 0.001: extremely significant")
    print("  p < 0.01:  highly significant")
    print("  p < 0.05:  significant")
    print("  p < 0.1:   marginally significant")
    print("  p >= 0.1:  not significant")
