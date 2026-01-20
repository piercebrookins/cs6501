# Running an LLM - Portfolio

This portfolio contains code, results, and analysis for Topic 1: Running an LLM.

## 📁 Contents

```
Running an LLM/
├── README.md                    # This file
├── notes.md                     # Discussion and analysis
├── graphs/                      # Generated visualizations
│   ├── accuracy_comparison.pdf
│   ├── timing_comparison.pdf
│   ├── subject_heatmap.pdf
│   └── error_patterns.pdf
└── results/                     # Evaluation results
    └── mmlu_results.json
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes matplotlib pandas
```

### 2. Authenticate with Hugging Face
```bash
huggingface-cli login
```

### 3. Run Evaluation
```bash
# Basic run
python enhanced_mmlu_eval.py

# With verbose output (shows each Q&A)
python enhanced_mmlu_eval.py --verbose

# Force CPU
python enhanced_mmlu_eval.py --cpu

# With 4-bit quantization (NVIDIA GPU only)
python enhanced_mmlu_eval.py --quant 4

# Resume from checkpoint after interruption
python enhanced_mmlu_eval.py --resume
```

### 4. Generate Graphs
```bash
python generate_graphs.py
```

### 5. Run Chat Agent
```bash
# With history (default)
python enhanced_chat_agent.py

# Without history (stateless)
python enhanced_chat_agent.py --no-history

# Different model
python enhanced_chat_agent.py --model qwen

# Resume previous session
python enhanced_chat_agent.py --resume
```

## 🤖 Models Evaluated

| Model | Parameters | Memory (FP16) |
|-------|------------|---------------|
| Llama 3.2-1B-Instruct | 1B | ~2.5 GB |
| Qwen2-0.5B-Instruct | 0.5B | ~1 GB |
| TinyLlama-1.1B-Chat | 1.1B | ~2.5 GB |

## 📊 Evaluation Details

### MMLU Subjects (10 selected)
1. Abstract Algebra
2. Anatomy
3. Astronomy
4. Business Ethics
5. Clinical Knowledge
6. College Biology
7. College Chemistry
8. Computer Security
9. Conceptual Physics
10. Econometrics

### Timing Metrics
- **Real Time**: Wall clock time (what you experience)
- **CPU Time**: Time spent on CPU computations
- **GPU Time**: Estimated GPU computation time

## 📈 Results Summary

See `notes.md` for detailed analysis of:
- Model accuracy comparisons
- Error patterns and whether models make similar mistakes
- Performance vs. accuracy tradeoffs
- Context management comparison (with/without history)

## 🔧 Files Overview

| File | Description |
|------|-------------|
| `enhanced_mmlu_eval.py` | Multi-model MMLU evaluation with timing |
| `generate_graphs.py` | Create visualizations from results |
| `enhanced_chat_agent.py` | Chat agent with context management |
| `llama_mmlu_eval.py` | Original course-provided script |
| `simple_chat_agent.py` | Original course-provided chat agent |

## 🐛 Troubleshooting

### "No Hugging Face token found"
```bash
huggingface-cli login
```

### "Llama license not accepted"
Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and accept the license.

### "bitsandbytes not working"
Quantization only works on NVIDIA GPUs with CUDA. Use `--cpu` or remove `--quant` flag.

### "Out of memory"
- Use `--quant 4` for 4-bit quantization (CUDA only)
- Use `--cpu` to run on CPU (slower but less memory)
- Close other applications
