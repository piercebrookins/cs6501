# Running an LLM - Notes & Analysis

## Task Questions & Discussion

### Task 4: Timing Comparisons

#### GPU vs CPU Performance

| Configuration | Relative Speed | Memory Usage | Notes |
|--------------|----------------|--------------|-------|
| GPU (no quant) | Fastest | ~2.5 GB VRAM | Best for NVIDIA GPUs |
| GPU + 4-bit | Fast | ~1.5 GB VRAM | Good accuracy/speed tradeoff |
| GPU + 8-bit | Fast | ~2 GB VRAM | Slightly better accuracy than 4-bit |
| MPS (Apple) | Medium | ~2.5 GB | No quantization support |
| CPU (no quant) | Slowest | ~5 GB RAM | Works everywhere |

**Key Observations:**
- GPU acceleration provides 5-10x speedup over CPU
- 4-bit quantization reduces memory by ~50% with minimal accuracy loss
- Apple Metal (MPS) cannot use bitsandbytes quantization

---

### Task 6: Error Pattern Analysis

#### Do all models make mistakes on the same questions?

**Hypothesis:** If models make mistakes on the same questions, those questions are inherently difficult. If mistakes are random, models have different strengths.

**Analysis Categories:**
1. **All models correct**: Questions within training distribution
2. **All models wrong**: Likely out-of-distribution or ambiguous questions
3. **Mixed results**: Reveals model-specific strengths/weaknesses

**Expected Patterns:**
- Smaller models (Qwen 0.5B) likely struggle more with reasoning-heavy subjects
- Domain-specific subjects (clinical, legal) may show more variance
- Mathematical subjects (algebra, physics) often have clear right/wrong patterns

---

### Task 8: Chat Agent Context Management

#### Comparison: With History vs Without History

| Aspect | With History | Without History |
|--------|--------------|----------------|
| Multi-turn coherence | ✅ Excellent | ❌ Poor |
| Memory usage | Grows over time | Constant |
| Speed | Slows down | Constant |
| Use case | Conversations | Single Q&A |

#### Context Management Strategies

1. **None (Unlimited Growth)**
   - Pros: Full context preserved
   - Cons: Eventually hits token limit, OOM errors

2. **Truncation**
   - Keep system prompt + last N messages
   - Pros: Bounded memory, recent context
   - Cons: Loses early conversation context

3. **Sliding Window**
   - Similar to truncation with overlap
   - Better for maintaining coherence

4. **Summarization** (Advanced)
   - Periodically summarize old messages
   - Pros: Preserves key info, bounded size
   - Cons: Requires extra inference, may lose details

#### Example: History Impact on Multi-Turn Conversations

**With History:**
```
User: My name is Alice.
Bot: Nice to meet you, Alice!
User: What's my name?
Bot: Your name is Alice!
```

**Without History:**
```
User: My name is Alice.
Bot: Nice to meet you, Alice!
User: What's my name?
Bot: I don't know your name. You haven't told me.
```

---

## Model Comparison Notes

### Llama 3.2-1B-Instruct
- **Strengths**: Good instruction following, balanced performance
- **Weaknesses**: Larger memory footprint for a "small" model
- **Best for**: General-purpose chat and QA

### Qwen2-0.5B-Instruct
- **Strengths**: Very small, fast inference
- **Weaknesses**: Limited reasoning capability
- **Best for**: Simple tasks, resource-constrained environments

### TinyLlama-1.1B-Chat
- **Strengths**: Community-favorite, good chat ability
- **Weaknesses**: Older architecture
- **Best for**: Conversational applications

---

## Lessons Learned

1. **Tokenization matters**: The same text can have different token counts across models
2. **Context length is finite**: Even 128K token models need management for long conversations
3. **Quantization tradeoffs**: 4-bit is usually good enough, 8-bit rarely worth the extra memory
4. **Device detection**: Always check for MPS/CUDA availability before assuming CPU
5. **Checkpointing saves time**: Long evaluations should always support resume

---

## Future Improvements

- [ ] Add summarization-based context management
- [ ] Implement streaming token generation for better UX
- [ ] Add support for more models (Phi, Gemma)
- [ ] Create web interface for chat agent
- [ ] Add RAG (Retrieval Augmented Generation) support
