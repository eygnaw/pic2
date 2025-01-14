import json
import pandas as pd
from llama_cpp import Llama
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Compute BLEU
def compute_bleu(predictions):
    scores = []
    for pred in predictions:
        reference = [pred["reference"].split()]  # BLEU expects tokenized references
        hypothesis = pred["prediction"].split()
        scores.append(sentence_bleu(reference, 
                                    hypothesis, 
                                    smoothing_function=SmoothingFunction().method1))
    return sum(scores) / len(scores) if scores else 0

# Load the test dataset
def load_test_dataset(file_path):
    with open(file_path, 'r') as f:
        return [
            {
                "input": next((c["value"] for c in data.get("conversations", []) if c["from"] == "human"), ""),
                "reference": next((c["value"] for c in data.get("conversations", []) if c["from"] == "gpt"), ""),
            }
            for data in map(json.loads, f)
        ]

# Evaluate a model
def evaluate_model(model, dataset):
    return [
        {
            "prediction": model(example["input"], max_tokens=128, stop=["\n"])["choices"][0]["text"].strip(),
            "reference": example["reference"]
        }
        for example in dataset
    ]

# Compute Metrics
def compute_metrics(predictions, model=None, dataset=None):
    rouge = load("rouge")
    bertscore = load("bertscore")
    preds = [p["prediction"] for p in predictions]
    refs = [p["reference"] for p in predictions]

    # ROUGE
    rouge_scores = rouge.compute(predictions=preds, references=refs)

    # Exact Match
    exact_matches = sum(p == r for p, r in zip(preds, refs)) / len(predictions)

    # F1 Score
    def compute_f1(p, r):
        p_tokens, r_tokens = set(p.split()), set(r.split())
        tp = len(p_tokens & r_tokens)
        if tp == 0:
            return 0
        precision = tp / len(p_tokens)
        recall = tp / len(r_tokens)
        return 2 * precision * recall / (precision + recall)

    f1_scores = [compute_f1(p, r) for p, r in zip(preds, refs)]

    # BLEU
    bleu_score = compute_bleu(predictions)

    # BERTScore
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")
    bert_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"]) if bert_scores["f1"] else 0

    return {
        "rouge": rouge_scores,
        "exact_match": exact_matches,
        "f1": sum(f1_scores) / len(f1_scores),
        "bleu": bleu_score,
        "bertscore": bert_f1
    }

# Main Workflow
def main():
    dataset = load_test_dataset("test.json")

    # Initialize models
    model1 = Llama(model_path="llama-3.2-1b-instruct-q8_0.gguf", verbose=False)
    model2 = Llama(model_path="unsloth.F16.gguf", verbose=False)

    try:
        # Evaluate models
        predictions1 = evaluate_model(model1, dataset)
        predictions2 = evaluate_model(model2, dataset)

        # Compute metrics
        metrics1 = compute_metrics(predictions1, model1, dataset)
        metrics2 = compute_metrics(predictions2, model2, dataset)

        # Display results
        comparison = pd.DataFrame({
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Exact Match", "F1", "BLEU", "BERTScore"],
            "Model 1": [
                metrics1["rouge"]["rouge1"],
                metrics1["rouge"]["rouge2"],
                metrics1["rouge"]["rougeL"],
                metrics1["exact_match"],
                metrics1["f1"],
                metrics1["bleu"],
                metrics1["bertscore"]
            ],
            "Model 2": [
                metrics2["rouge"]["rouge1"],
                metrics2["rouge"]["rouge2"],
                metrics2["rouge"]["rougeL"],
                metrics2["exact_match"],
                metrics2["f1"],
                metrics2["bleu"],
                metrics2["bertscore"]
            ]
        })

        print(comparison)

    finally:
        # Cleanup models
        model1.close()
        model2.close()

if __name__ == "__main__":
    main()
