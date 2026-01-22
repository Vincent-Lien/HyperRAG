import json
import os
import re
import string
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description="Evaluate QA results")
parser.add_argument("--model_name", type=str, help="Path to the folder containing results and questions")
parser.add_argument("dataset", type=str, help="Name of the dataset to evaluate")
args = parser.parse_args()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def precision_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    return num_same / len(pred_tokens) if pred_tokens else 0.0


def recall_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    return num_same / len(gt_tokens) if gt_tokens else 0.0


def exact_match(response, answers):
    clean_result = response.strip().replace(" ", "").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ", "").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_folder(folder_path, dataset):
    result_data = load_jsonl(os.path.join(folder_path, "test_results_cleaned.jsonl"))
    question_data = load_jsonl(f"dataset/open_domain_splitted/{dataset}_query_test.jsonl")

    # Build question -> ground_truths mapping
    question2answers = {}
    for item in question_data:
        q = item["question"]
        # ground_truths: string or list
        gt = item["answer"]
        question2answers[q] = gt if isinstance(gt, list) else [gt]

    f1_list, precision_list, recall_list = [], [], []
    num_right = 0
    wrong_list = []

    for item in result_data:
        question = item["question"]
        response = item.get("output", item.get("answer", ""))  # output or answer
        ground_truths = question2answers.get(question, [])

        if not response or not ground_truths:
            wrong_list.append(item)
            continue

        best_f1 = max(f1_score(response, gt) for gt in ground_truths)
        best_precision = max(precision_score(response, gt) for gt in ground_truths)
        best_recall = max(recall_score(response, gt) for gt in ground_truths)

        f1_list.append(best_f1)
        precision_list.append(best_precision)
        recall_list.append(best_recall)

        if exact_match(response, ground_truths):
            num_right += 1
        else:
            wrong_list.append(item)

    total = len(result_data)
    return {
        "exact_match": num_right / total,
        "f1_score": sum(f1_list) / total,
        "precision": sum(precision_list) / total,
        "recall": sum(recall_list) / total,
        "wrong_list": wrong_list,
    }


# main
model_name = args.model_name
dataset = args.dataset

base_dir = f"results/{model_name}/open_domain"
print(f"==={dataset}===")
folder = os.path.join(base_dir, dataset)
result = evaluate_folder(folder, dataset)
print(json.dumps({k: v for k, v in result.items() if k != "wrong_list"}, indent=4))