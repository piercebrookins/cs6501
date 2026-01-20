#!/usr/bin/env python3
"""
Enhanced MMLU Evaluation Script - Multi-Model with Timing

Features:
- Supports multiple models (Llama 3.2-1B, Qwen2-0.5B, TinyLlama-1.1B)
- Detailed timing info (real time, CPU time, GPU time)
- Verbose mode to show questions, answers, and correctness
- Configurable quantization and device settings
- Pickle checkpointing for restartability
- Exports results for graphing

Usage:
    python enhanced_mmlu_eval.py [--verbose] [--cpu] [--quant 4|8] [--resume]
"""

import argparse
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models to evaluate (small models that can run on laptop)
MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# 10 MMLU subjects for evaluation
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "computer_security",
    "conceptual_physics",
    "econometrics",
]

MAX_NEW_TOKENS = 1
CHECKPOINT_FILE = "mmlu_checkpoint.pkl"
RESULTS_FILE = "mmlu_results.json"


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class Timer:
    """Context manager for detailed timing measurements."""

    def __init__(self):
        self.real_time = 0.0
        self.cpu_time = 0.0
        self.gpu_time = 0.0
        self._start_real = None
        self._start_cpu = None
        self._cuda_available = torch.cuda.is_available()

    def __enter__(self):
        if self._cuda_available:
            torch.cuda.synchronize()
        self._start_real = time.perf_counter()
        self._start_cpu = time.process_time()
        return self

    def __exit__(self, *args):
        if self._cuda_available:
            torch.cuda.synchronize()
        self.real_time = time.perf_counter() - self._start_real
        self.cpu_time = time.process_time() - self._start_cpu
        # GPU time approximation (real - cpu when using GPU)
        if self._cuda_available:
            self.gpu_time = max(0, self.real_time - self.cpu_time)

    def to_dict(self):
        return {
            "real_time_sec": round(self.real_time, 4),
            "cpu_time_sec": round(self.cpu_time, 4),
            "gpu_time_sec": round(self.gpu_time, 4),
        }


class CumulativeTimer:
    """Accumulates timing across multiple operations."""

    def __init__(self):
        self.total_real = 0.0
        self.total_cpu = 0.0
        self.total_gpu = 0.0

    def add(self, timer: Timer):
        self.total_real += timer.real_time
        self.total_cpu += timer.cpu_time
        self.total_gpu += timer.gpu_time

    def to_dict(self):
        return {
            "total_real_time_sec": round(self.total_real, 2),
            "total_cpu_time_sec": round(self.total_cpu, 2),
            "total_gpu_time_sec": round(self.total_gpu, 2),
        }


# ============================================================================
# DEVICE & MODEL UTILITIES
# ============================================================================

def detect_device(use_gpu: bool = True) -> str:
    """Detect best available device."""
    if not use_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_quantization_config(quant_bits: int | None, device: str):
    """Create quantization config if applicable."""
    if quant_bits is None or device != "cuda":
        return None

    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    return None


def load_model(model_name: str, model_path: str, device: str, quant_bits: int | None):
    """Load model and tokenizer with optional quantization."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Path: {model_path}")
    print(f"Device: {device}, Quantization: {quant_bits or 'None'}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quantization_config(quant_bits, device)

    load_kwargs = {"low_cpu_mem_usage": True}

    if quant_config:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
    elif device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if device in ["mps", "cpu"] and not quant_config:
        model = model.to(device)

    model.eval()
    print(f"✓ Model loaded on {next(model.parameters()).device}")

    return model, tokenizer


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """Format MMLU question as multiple choice."""
    labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_prediction(model, tokenizer, prompt: str) -> str:
    """Get model prediction for a question."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # Extract answer letter
    answer = generated.strip()[:1].upper()
    if answer not in "ABCD":
        for char in generated.upper():
            if char in "ABCD":
                return char
        return "A"  # Default fallback
    return answer


def evaluate_subject(
    model,
    tokenizer,
    subject: str,
    verbose: bool = False,
) -> dict:
    """Evaluate model on a single MMLU subject."""
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Failed to load {subject}: {e}")
        return None

    correct = 0
    total = 0
    question_results = []
    timer = CumulativeTimer()

    desc = f"Evaluating {subject}"
    for example in tqdm(dataset, desc=desc, leave=False):
        question = example["question"]
        choices = example["choices"]
        correct_idx = example["answer"]
        correct_answer = "ABCD"[correct_idx]

        prompt = format_mmlu_prompt(question, choices)

        # Time the prediction
        t = Timer()
        with t:
            predicted = get_prediction(model, tokenizer, prompt)
        timer.add(t)

        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Store result for analysis
        result = {
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted": predicted,
            "is_correct": is_correct,
        }
        question_results.append(result)

        # Verbose output
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Q: {question[:80]}...")
            print(f"   Correct: {correct_answer}, Predicted: {predicted}")

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "timing": timer.to_dict(),
        "questions": question_results,
    }


def evaluate_model(
    model_name: str,
    model_path: str,
    subjects: list[str],
    device: str,
    quant_bits: int | None,
    verbose: bool = False,
    checkpoint: dict | None = None,
) -> dict:
    """Evaluate a model on all subjects."""
    # Check checkpoint for completed subjects
    completed_subjects = {}
    if checkpoint and model_name in checkpoint:
        completed_subjects = {
            s["subject"]: s for s in checkpoint[model_name].get("subjects", [])
        }
        print(f"📂 Resuming {model_name}: {len(completed_subjects)} subjects done")

    # Load model
    model, tokenizer = load_model(model_name, model_path, device, quant_bits)

    results = []
    model_timer = CumulativeTimer()

    for subject in subjects:
        # Skip if already completed
        if subject in completed_subjects:
            print(f"⏭️  Skipping {subject} (already completed)")
            results.append(completed_subjects[subject])
            continue

        result = evaluate_subject(model, tokenizer, subject, verbose)
        if result:
            results.append(result)
            model_timer.total_real += result["timing"]["total_real_time_sec"]
            model_timer.total_cpu += result["timing"]["total_cpu_time_sec"]
            model_timer.total_gpu += result["timing"]["total_gpu_time_sec"]

            print(
                f"✓ {subject}: {result['accuracy']:.1f}% "
                f"({result['timing']['total_real_time_sec']:.1f}s)"
            )

    # Calculate totals
    total_correct = sum(r["correct"] for r in results)
    total_questions = sum(r["total"] for r in results)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions else 0

    # Clean up model to free memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "model_path": model_path,
        "device": device,
        "quantization": quant_bits,
        "subjects": results,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "overall_accuracy": overall_accuracy,
        "timing": model_timer.to_dict(),
    }


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(data: dict, filepath: str = CHECKPOINT_FILE):
    """Save checkpoint for restartability."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str = CHECKPOINT_FILE) -> dict | None:
    """Load checkpoint if exists."""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"📂 Loaded checkpoint from {filepath}")
        return data
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced MMLU Evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show Q&A details")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--quant", type=int, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Enhanced MMLU Evaluation - Multi-Model with Timing")
    print("=" * 70)

    # Setup
    device = detect_device(not args.cpu)
    print(f"\n📱 Device: {device}")
    print(f"📊 Subjects: {len(MMLU_SUBJECTS)}")
    print(f"🤖 Models: {len(MODELS)}")

    # Load checkpoint if resuming
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint()

    # Filter models if specified
    models_to_run = MODELS
    if args.models:
        models_to_run = {k: v for k, v in MODELS.items() if k in args.models}

    # Run evaluation
    all_results = checkpoint or {}
    start_time = datetime.now()

    for model_name, model_path in models_to_run.items():
        try:
            result = evaluate_model(
                model_name=model_name,
                model_path=model_path,
                subjects=MMLU_SUBJECTS,
                device=device,
                quant_bits=args.quant,
                verbose=args.verbose,
                checkpoint=checkpoint,
            )
            all_results[model_name] = result

            # Save checkpoint after each model
            save_checkpoint(all_results)

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for model_name, result in all_results.items():
        if isinstance(result, dict) and "overall_accuracy" in result:
            timing = result["timing"]
            print(f"\n🤖 {model_name}:")
            print(f"   Accuracy: {result['overall_accuracy']:.2f}%")
            print(f"   Real Time: {timing['total_real_time_sec']:.1f}s")
            print(f"   CPU Time:  {timing['total_cpu_time_sec']:.1f}s")
            print(f"   GPU Time:  {timing['total_gpu_time_sec']:.1f}s")

    print(f"\n⏱️  Total Duration: {total_duration/60:.1f} minutes")

    # Save final results
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "quantization": args.quant,
        "total_duration_sec": total_duration,
        "models": all_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n📄 Results saved to {RESULTS_FILE}")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"🗑️  Checkpoint removed")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - checkpoint saved for resume")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
