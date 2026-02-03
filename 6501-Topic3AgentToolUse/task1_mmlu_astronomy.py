"""
Task 1: MMLU Evaluation - Astronomy Topic
Modified to use Ollama server for Llama 3.2-1B

Usage:
1. Start Ollama server: ollama serve
2. Pull model: ollama pull llama3.2:1b
3. Run: python task1_mmlu_astronomy.py
"""

import requests
import json
from datasets import load_dataset
from tqdm.auto import tqdm
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"
SUBJECT = "astronomy"


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer with just the letter (A, B, C, or D):"
    return prompt


def get_ollama_prediction(prompt):
    """Get model's prediction from Ollama server"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 5,
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        generated_text = result.get("response", "").strip()

        # Extract answer letter
        answer = generated_text[:1].upper()
        if answer not in ["A", "B", "C", "D"]:
            for char in generated_text.upper():
                if char in ["A", "B", "C", "D"]:
                    answer = char
                    break
            else:
                answer = "A"
        return answer
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "A"


def evaluate_subject(subject):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_ollama_prediction(prompt)

        if predicted_answer == correct_answer:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Result: {correct}/{total} correct = {accuracy:.2f}%")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


def main():
    print("\n" + "="*70)
    print(f"Llama 3.2-1B MMLU Evaluation via Ollama - {SUBJECT.upper()}")
    print("="*70 + "\n")

    start_time = datetime.now()
    result = evaluate_subject(SUBJECT)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Subject: {SUBJECT}")
    print(f"Model: {MODEL_NAME} (via Ollama)")
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Duration: {duration:.1f} seconds")
    print("="*70)

    return result


if __name__ == "__main__":
    main()
