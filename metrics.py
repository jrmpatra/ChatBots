# llm_metrics.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import textstat

# ---------------- Perplexity ----------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def calculate_perplexity_llama(model_name: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity


# ---------------- Accuracy / F1 ----------------
def calculate_f1_accuracy(y_true: list, y_pred: list):
    """
    y_true: list of reference labels
    y_pred: list of predicted labels
    Returns: dict with 'f1' and 'accuracy'
    """
    f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    return {"f1": f1, "accuracy": acc}

# ---------------- BLEU ----------------
def calculate_bleu(reference: str, candidate: str):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    return score

# ---------------- ROUGE ----------------
def calculate_rouge(reference: str, candidate: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# ---------------- Bias / Toxicity ----------------
def calculate_toxicity(text: str):
    """
    Returns a dictionary with toxicity predictions
    Uses HuggingFace 'unitary/toxic-bert' model
    """
    classifier = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)
    results = classifier(text)
    # results: list of list of dicts [{'label': 'toxic', 'score': 0.99}, ...]
    return results[0]

# ---------------- Readability ----------------
def calculate_readability(text: str):
    """
    Returns Flesch Reading Ease score (higher = easier to read)
    """
    return textstat.flesch_reading_ease(text)

# ---------------- Example Combined Function ----------------
def evaluate_generated_text(model_name: str, reference: str, candidate: str, y_true=None, y_pred=None):
    """
    Computes all metrics for convenience
    """
    metrics = {}
   # metrics["perplexity"] = calculate_perplexity_llama(model_name, candidate)
    metrics["bleu"] = calculate_bleu(reference, candidate)
    metrics["rouge"] = calculate_rouge(reference, candidate)
   # metrics["readability"] = calculate_readability(candidate)
    
    if y_true and y_pred:
        acc_f1 = calculate_f1_accuracy(y_true, y_pred)
        metrics.update(acc_f1)
    
    metrics["toxicity"] = calculate_toxicity(candidate)
    
    return metrics

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    ref = "The cat sat on the mat."
    pred = "The cat is sitting on the mat."
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    metrics = evaluate_generated_text(model_name, ref, pred)
    print(metrics)
