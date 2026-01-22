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

def bootstrap_test(scores1: np.ndarray, scores2: np.ndarray, n_bootstrap: int = 10000, confidence_level: float = 0.95) -> dict:
    """
    Perform bootstrap significance test.
    Returns p-value and confidence interval for the difference.
    """
    n = len(scores1)
    assert n == len(scores2), "Sample sizes must be equal"
    
    # Observed difference
    obs_diff = np.mean(scores1) - np.mean(scores2)
    
    # Bootstrap resampling
    diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        boot_diff = np.mean(scores1[indices]) - np.mean(scores2[indices])
        diffs.append(boot_diff)
    
    diffs = np.array(diffs)
    
    # Two-tailed p-value
    p_value = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))
    
    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    
    return {
        'observed_diff': float(obs_diff * 100),  # Convert to percentage
        'p_value': p_value,
        'ci_lower': ci_lower * 100,
        'ci_upper': ci_upper * 100,
        'significance_level': get_significance_level(p_value)
    }

def paired_t_test(scores1: np.ndarray, scores2: np.ndarray) -> dict:
    """Perform paired t-test."""
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    return {
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

def evaluate_significance(model1: str, model2: str, domain: str, sampled: bool = False, 
                         n_bootstrap: int = 10000, difficulty: str = 'easy') -> dict:
    """
    Compare two models on a specific domain and difficulty level.
    """
    result_dir_suffix = "wikitopics_sampled" if sampled else "wikitopics"
    
    model1_path = f"results/{model1}/{result_dir_suffix}/{domain}_output.jsonl"
    model2_path = f"results/{model2}/{result_dir_suffix}/{domain}_output.jsonl"
    
    # Load data for both models
    easy1, hard1, pred1 = load_predictions(model1_path)
    easy2, hard2, pred2 = load_predictions(model2_path)
    
    if not easy1 or not easy2:
        return {
            'error': f'Failed to load data for domain {domain}'
        }
    
    # Select difficulty level
    if difficulty == 'easy':
        gold1, gold2 = easy1, easy2
    else:  # hard
        gold1, gold2 = hard1, hard2
    
    # Ensure same number of samples
    min_samples = min(len(pred1), len(pred2))
    gold1 = gold1[:min_samples]
    gold2 = gold2[:min_samples]
    pred1 = pred1[:min_samples]
    pred2 = pred2[:min_samples]
    
    # Compute per-sample scores
    mrr1 = compute_mrr_per_sample(gold1, pred1)
    mrr2 = compute_mrr_per_sample(gold2, pred2)
    
    hits1 = compute_hits_at_k_per_sample(gold1, pred1, k=10)
    hits2 = compute_hits_at_k_per_sample(gold2, pred2, k=10)
    
    # Statistical tests
    mrr_bootstrap = bootstrap_test(mrr1, mrr2, n_bootstrap=n_bootstrap)
    mrr_ttest = paired_t_test(mrr1, mrr2)
    
    hits_bootstrap = bootstrap_test(hits1, hits2, n_bootstrap=n_bootstrap)
    hits_ttest = paired_t_test(hits1, hits2)
    
    results = {
        'domain': domain,
        'difficulty': difficulty,
        'n_samples': int(min_samples),
        'model1': model1,
        'model2': model2,
        'model1_mrr_mean': float(np.mean(mrr1) * 100),
        'model2_mrr_mean': float(np.mean(mrr2) * 100),
        'model1_hits10_mean': float(np.mean(hits1) * 100),
        'model2_hits10_mean': float(np.mean(hits2) * 100),
        'MRR': {
            'bootstrap': mrr_bootstrap,
            't_test': mrr_ttest
        },
        'Hits@10': {
            'bootstrap': hits_bootstrap,
            't_test': hits_ttest
        }
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical significance testing for QA results.")
    parser.add_argument('--model1', type=str, required=True, help='First model name')
    parser.add_argument('--model2', type=str, required=True, help='Second model name')
    parser.add_argument('--sampled', action='store_true', help='Use sampled dataset')
    parser.add_argument('--domain', type=str, default='all', help='Domain to evaluate on (default: all)')
    parser.add_argument('--difficulty', type=str, default='both', choices=['easy', 'hard', 'both'],
                       help='Difficulty level to test (default: both)')
    parser.add_argument('--n_bootstrap', type=int, default=10000, help='Number of bootstrap iterations')
    args = parser.parse_args()
    
    if args.domain != 'all':
        domains = [args.domain]
    else:
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
        'n_bootstrap': args.n_bootstrap,
        'domains': {}
    }
    
    print(f"Comparing {args.model1} vs {args.model2}")
    print(f"Dataset: {'sampled' if args.sampled else 'full'}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print("=" * 80)
    
    for domain in domains:
        print(f"\nDomain: {domain}")
        all_results['domains'][domain] = {}
        
        for difficulty in difficulties:
            print(f"\n  Difficulty: {difficulty}")
            results = evaluate_significance(
                args.model1, args.model2, domain, 
                sampled=args.sampled, 
                n_bootstrap=args.n_bootstrap,
                difficulty=difficulty
            )
            
            if 'error' in results:
                print(f"    {results['error']}")
                all_results['domains'][domain][difficulty] = results
                continue
            
            all_results['domains'][domain][difficulty] = results
            
            # Print results
            print(f"    Samples: {results['n_samples']}")
            print(f"    {args.model1} MRR: {results['model1_mrr_mean']:.2f}%")
            print(f"    {args.model2} MRR: {results['model2_mrr_mean']:.2f}%")
            print(f"    {args.model1} Hits@10: {results['model1_hits10_mean']:.2f}%")
            print(f"    {args.model2} Hits@10: {results['model2_hits10_mean']:.2f}%")
            
            # MRR significance
            mrr_boot = results['MRR']['bootstrap']
            print(f"\n    MRR Difference: {mrr_boot['observed_diff']:.2f}%")
            print(f"      Bootstrap p-value: {mrr_boot['p_value']:.4f} ({mrr_boot['significance_level']})")
            print(f"      95% CI: [{mrr_boot['ci_lower']:.2f}%, {mrr_boot['ci_upper']:.2f}%]")
            print(f"      Paired t-test p-value: {results['MRR']['t_test']['p_value']:.4f} ({results['MRR']['t_test']['significance_level']})")
            
            # Hits@10 significance
            hits_boot = results['Hits@10']['bootstrap']
            print(f"\n    Hits@10 Difference: {hits_boot['observed_diff']:.2f}%")
            print(f"      Bootstrap p-value: {hits_boot['p_value']:.4f} ({hits_boot['significance_level']})")
            print(f"      95% CI: [{hits_boot['ci_lower']:.2f}%, {hits_boot['ci_upper']:.2f}%]")
            print(f"      Paired t-test p-value: {results['Hits@10']['t_test']['p_value']:.4f} ({results['Hits@10']['t_test']['significance_level']})")
        
        print("\n" + "-" * 80)
    
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
