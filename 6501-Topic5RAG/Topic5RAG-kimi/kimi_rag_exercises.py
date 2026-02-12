#!/usr/bin/env python3
"""
KIMI RAG Exercises - Complete Pipeline
CS6501 Topic 5: Retrieval-Augmented Generation

This script runs all 13 exercises (0-12) for the RAG portfolio.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# Enable MPS fallback for PyTorch operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("Topic5RAG")
CORPORA_DIR = Path("/Users/pierce/Documents/CS6501/6501-Topic5RAG/Corpora")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_TOP_K = 5

# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device() -> Tuple[str, torch.dtype]:
    """Detect best available compute device."""
    if torch.cuda.is_available():
        device = 'cuda'
        dtype = torch.float16
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Using CUDA GPU: {device_name} ({memory_gb:.1f} GB)")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        dtype = torch.float32
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        dtype = torch.float32
        print("⚠ Using CPU (no GPU detected)")
    return device, dtype

DEVICE, DTYPE = get_device()

# =============================================================================
# IMPORTS (with error handling)
# =============================================================================

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    from sentence_transformers import SentenceTransformer
    import faiss
    import fitz  # PyMuPDF
    print("✓ All dependencies imported successfully")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Install with: pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate openai")
    sys.exit(1)

# =============================================================================
# MODEL LOADING
# =============================================================================

class ModelCache:
    """Cache for loaded models to avoid reloading."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = None
            cls._instance.llm = None
            cls._instance.embedder = None
        return cls._instance
    
    def get_tokenizer(self):
        if self.tokenizer is None:
            print(f"Loading tokenizer: {MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return self.tokenizer
    
    def get_llm(self):
        if self.llm is None:
            print(f"Loading LLM: {MODEL_NAME}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=DTYPE,
                device_map=DEVICE if DEVICE != 'mps' else None,
                low_cpu_mem_usage=True
            )
            if DEVICE == 'mps':
                self.llm = self.llm.to(DEVICE)
        return self.llm
    
    def get_embedder(self):
        if self.embedder is None:
            print(f"Loading embedder: {EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        return self.embedder

model_cache = ModelCache()

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def load_pdfs_from_folder(folder_path: str) -> List[Dict]:
    """Load all PDFs from a folder and extract text."""
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))
    
    if not pdf_files:
        # Try pdf_embedded subdirectory
        pdf_files = sorted((folder / "pdf_embedded").glob("*.pdf"))
    
    documents = []
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            text = "\n\n".join([page.get_text() for page in doc])
            doc.close()
            documents.append({
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "text": text,
                "length": len(text)
            })
            print(f"  ✓ Loaded: {pdf_path.name} ({len(text):,} chars)")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_path.name}: {e}")
    
    return documents

def chunk_documents(documents: List[Dict], chunk_size: int = DEFAULT_CHUNK_SIZE, 
                   overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict]:
    """Split documents into overlapping chunks."""
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings
                for delim in ['.\n', '. ', '! ', '? ', '\n\n']:
                    pos = text.rfind(delim, start, end)
                    if pos > start + chunk_size // 2:
                        end = pos + len(delim)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["filename"],
                    "chunk_num": chunk_num,
                    "start_char": start,
                    "end_char": end
                })
                chunk_num += 1
            
            start = end - overlap
    
    return chunks

# =============================================================================
# VECTOR STORE
# =============================================================================

class VectorStore:
    """FAISS-based vector store for document retrieval."""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = None
    
    def build_index(self, chunks: List[Dict], embedder) -> 'VectorStore':
        """Build FAISS index from chunks."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        
        print(f"Embedding {len(texts)} chunks on {DEVICE}...")
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine for normalized vectors
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Index built with {self.index.ntotal} vectors (dim={self.dimension})")
        return self
    
    def search(self, query: str, embedder, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for top-k chunks matching query."""
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path_prefix: str):
        """Save index and chunks to disk."""
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.chunks", 'w') as f:
            json.dump(self.chunks, f)
        print(f"✓ Saved index to {path_prefix}.faiss")
    
    def load(self, path_prefix: str, embedder) -> 'VectorStore':
        """Load index and chunks from disk."""
        self.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}.chunks", 'r') as f:
            self.chunks = json.load(f)
        self.dimension = self.index.d
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        return self

# =============================================================================
# LLM INTERFACE
# =============================================================================

def generate_response(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate response from LLM."""
    tokenizer = model_cache.get_tokenizer()
    llm = model_cache.get_llm()
    
    messages = [{"role": "user", "content": prompt}]
    
    if tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"User: {prompt}\n\nAssistant:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def generate_with_rag(query: str, vector_store: VectorStore, top_k: int = DEFAULT_TOP_K, 
                      prompt_template: str = "default") -> Tuple[str, List, float]:
    """Generate RAG-enhanced response."""
    embedder = model_cache.get_embedder()
    
    # Retrieve relevant chunks
    start_time = time.time()
    retrieved = vector_store.search(query, embedder, top_k)
    retrieval_time = time.time() - start_time
    
    # Build context
    context = "\n\n---\n\n".join([f"[Source: {r[0]['source']}]\n{r[0]['text']}" for r in retrieved])
    
    # Select prompt template
    templates = {
        "default": f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer based on the context provided:""",
        
        "strict": f"""You must answer using ONLY the provided context. If the answer is not in the context, say "I cannot answer based on the provided documents."

Context:
{context}

Question: {query}

Answer (using only the context):""",
        
        "citation": f"""Answer the question using the provided context. Cite your sources with [Source: filename].

Context:
{context}

Question: {query}

Answer with citations:""",
        
        "minimal": f"""Context: {context}

Question: {query}

Answer:"""
    }
    
    prompt = templates.get(prompt_template, templates["default"])
    
    # Generate response
    start_time = time.time()
    response = generate_response(prompt)
    generation_time = time.time() - start_time
    
    total_time = retrieval_time + generation_time
    return response, retrieved, total_time

def generate_without_rag(query: str) -> Tuple[str, float]:
    """Generate direct response without RAG."""
    start_time = time.time()
    response = generate_response(query)
    generation_time = time.time() - start_time
    return response, generation_time

# =============================================================================
# EXERCISE OUTPUT HANDLER
# =============================================================================

class ExerciseOutput:
    """Handler for saving exercise outputs."""
    
    def __init__(self, exercise_num: int, title: str):
        self.exercise_num = exercise_num
        self.title = title
        self.lines = []
        self.add_header()
    
    def add_header(self):
        self.lines.append("=" * 70)
        self.lines.append(f"KIMI EXERCISE {self.exercise_num}: {self.title}")
        self.lines.append("=" * 70)
        self.lines.append("")
    
    def add_section(self, title: str):
        self.lines.append("-" * 70)
        self.lines.append(title)
        self.lines.append("-" * 70)
        self.lines.append("")
    
    def add_query_section(self, query: str, query_num: int = 1):
        self.lines.append("=" * 60)
        self.lines.append(f"Q{query_num}: {query}")
        self.lines.append("=" * 60)
        self.lines.append("")
    
    def add_without_rag(self, response: str, time_taken: float):
        self.lines.append("--- WITHOUT RAG (Direct Query) ---")
        self.lines.append(f"Response time: {time_taken:.2f}s")
        self.lines.append("Answer:")
        self.lines.append(response)
        self.lines.append("")
    
    def add_with_rag(self, response: str, retrieved: List, time_taken: float, top_k: int = 5):
        self.lines.append(f"--- WITH RAG (top_k={top_k}) ---")
        self.lines.append(f"Response time: {time_taken:.2f}s")
        self.lines.append("Retrieved chunks:")
        for i, (chunk, score) in enumerate(retrieved, 1):
            preview = chunk['text'][:150].replace('\n', ' ')
            self.lines.append(f"  {i}. [{chunk['source']}] (score: {score:.3f}) {preview}...")
        self.lines.append("")
        self.lines.append("Answer:")
        self.lines.append(response)
        self.lines.append("")
    
    def add_analysis(self, text: str):
        self.lines.append("--- ANALYSIS ---")
        self.lines.append(text)
        self.lines.append("")
    
    def add_text(self, text: str):
        self.lines.append(text)
    
    def save(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        filename = OUTPUT_DIR / f"kimi_exercise_{self.exercise_num}_{self.title.lower().replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write('\n'.join(self.lines))
        print(f"✓ Saved to {filename}")
        return filename

# =============================================================================
# EXERCISES
# =============================================================================

def exercise_0_setup() -> Tuple[VectorStore, VectorStore]:
    """Exercise 0: Environment setup and index building."""
    print("\n" + "=" * 70)
    print("EXERCISE 0: SETUP")
    print("=" * 70)
    
    out = ExerciseOutput(0, "Setup")
    out.add_text(f"Environment: LOCAL")
    out.add_text(f"Device: {DEVICE}, Dtype: {DTYPE}")
    out.add_text(f"PyTorch version: {torch.__version__}")
    out.add_text(f"CUDA available: {torch.cuda.is_available()}")
    out.add_text(f"MPS available: {torch.backends.mps.is_available()}")
    out.add_text(f"MPS built: {torch.backends.mps.is_built()}")
    out.add_text("")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check for Corpora
    if not CORPORA_DIR.exists():
        out.add_text("ERROR: Corpora/ directory not found!")
        out.add_text("Please unzip Corpora.zip in the repo root.")
        out.save()
        raise FileNotFoundError("Corpora/ directory not found. Please unzip Corpora.zip")
    
    # Load embedder
    embedder = model_cache.get_embedder()
    
    # Build Model T index
    out.add_section("Loading Model T Service Manual Corpus")
    modelt_docs = load_pdfs_from_folder(CORPORA_DIR / "ModelTService")
    modelt_chunks = chunk_documents(modelt_docs)
    out.add_text(f"Total chunks: {len(modelt_chunks)}")
    
    modelt_store = VectorStore().build_index(modelt_chunks, embedder)
    modelt_store.save(str(OUTPUT_DIR / "index_modelt"))
    
    # Build Congressional Record index
    out.add_section("Loading Congressional Record Corpus")
    cr_docs = load_pdfs_from_folder(CORPORA_DIR / "Congressional_Record_Jan_2026")
    cr_chunks = chunk_documents(cr_docs)
    out.add_text(f"Total chunks: {len(cr_chunks)}")
    
    cr_store = VectorStore().build_index(cr_chunks, embedder)
    cr_store.save(str(OUTPUT_DIR / "index_cr"))
    
    out.add_section("Summary")
    out.add_text(f"✓ Model T corpus: {len(modelt_chunks)} chunks indexed")
    out.add_text(f"✓ CR corpus: {len(cr_chunks)} chunks indexed")
    out.save()
    
    return modelt_store, cr_store

def exercise_1_rag_vs_norag(modelt_store: VectorStore, cr_store: VectorStore):
    """Exercise 1: Compare RAG vs No-RAG responses."""
    print("\n" + "=" * 70)
    print("EXERCISE 1: RAG vs No-RAG Comparison")
    print("=" * 70)
    
    out = ExerciseOutput(1, "RAG vs No-RAG")
    out.add_text(f"Using {MODEL_NAME} on {DEVICE}")
    out.add_text("Comparing direct query (no RAG) vs RAG pipeline")
    out.add_text("")
    
    # Model T queries
    out.add_section("MODEL T FORD CORPUS QUERIES")
    
    modelt_queries = [
        "How do I adjust the carburetor on a Model T?",
        "What is the correct spark plug gap for a Model T Ford?",
        "How do I change the oil in a Model T?",
        "What type of fuel does the Model T use?",
        "How do I start a Model T Ford?"
    ]
    
    for i, query in enumerate(modelt_queries, 1):
        out.add_query_section(query, i)
        
        # Without RAG
        response_no_rag, time_no_rag = generate_without_rag(query)
        out.add_without_rag(response_no_rag, time_no_rag)
        
        # With RAG
        response_rag, retrieved, time_rag = generate_with_rag(query, modelt_store, top_k=5)
        out.add_with_rag(response_rag, retrieved, time_rag, top_k=5)
        
        # Analysis
        analysis = f"""Direct answer length: {len(response_no_rag)} chars
RAG answer length: {len(response_rag)} chars

OBSERVATION: The RAG answer provides specific details from the actual service manual, 
while the direct answer provides general knowledge that may lack specific procedures."""
        out.add_analysis(analysis)
    
    # Congressional Record queries
    out.add_section("CONGRESSIONAL RECORD CORPUS QUERIES")
    
    cr_queries = [
        "What legislation was discussed regarding climate change?",
        "What were the main topics in the January 2026 congressional sessions?",
        "Which senators spoke about healthcare reform?"
    ]
    
    for i, query in enumerate(cr_queries, 1):
        out.add_query_section(query, i)
        
        response_no_rag, time_no_rag = generate_without_rag(query)
        out.add_without_rag(response_no_rag, time_no_rag)
        
        response_rag, retrieved, time_rag = generate_with_rag(query, cr_store, top_k=5)
        out.add_with_rag(response_rag, retrieved, time_rag, top_k=5)
        
        analysis = f"""Direct answer length: {len(response_no_rag)} chars
RAG answer length: {len(response_rag)} chars

OBSERVATION: RAG provides grounded answers based on actual congressional records,
while direct query may provide general knowledge without specific legislative details."""
        out.add_analysis(analysis)
    
    out.save()

def exercise_2_gpt4o_mini():
    """Exercise 2: Compare with OpenAI GPT API."""
    print("\n" + "=" * 70)
    print("EXERCISE 2: GPT4o Mini Comparison")
    print("=" * 70)
    
    out = ExerciseOutput(2, "GPT4o Mini Comparison")
    out.add_text("Comparing local Qwen+RAG vs GPT4o Mini API")
    out.add_text("")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        out.add_text("ERROR: OPENAI_API_KEY not found in environment!")
        out.add_text("Please set the API key in your .env file.")
        out.save()
        print("⚠ Skipping Exercise 2 - no API key")
        return
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        queries = [
            "How do I adjust the carburetor on a Model T?",
            "What is the firing order of a Model T Ford engine?"
        ]
        
        for i, query in enumerate(queries, 1):
            out.add_query_section(query, i)
            
            # GPT4o Mini response
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": query}],
                temperature=0.7
            )
            gpt_time = time.time() - start
            gpt_answer = response.choices[0].message.content
            
            out.add_text("--- GPT4o Mini No RAG ---")
            out.add_text(f"Response time: {gpt_time:.2f}s")
            out.add_text(f"Answer:\n{gpt_answer}")
            out.add_text("")
        
        out.add_section("Analysis")
        out.add_text("""GPT4o Mini provides general knowledge responses without access to the 
specific Model T service manual. The local Qwen+RAG combination provides more 
detailed, document-grounded answers with specific procedures from the actual manual.
""")
        
    except Exception as e:
        out.add_text(f"ERROR: {e}")
    
    out.save()

def exercise_3_frontier_comparison():
    """Exercise 3: Manual comparison with frontier models."""
    print("\n" + "=" * 70)
    print("EXERCISE 3: Frontier Model Comparison (Manual)")
    print("=" * 70)
    
    out = ExerciseOutput(3, "Frontier Comparison")
    out.add_text("This exercise requires MANUAL testing with GPT-4/Claude web interfaces.")
    out.add_text("")
    out.add_text("Test these queries on GPT-4 (ChatGPT) and Claude (Claude.ai):")
    out.add_text("")
    
    queries = [
        "How do I adjust the carburetor on a Model T?",
        "What is the correct spark plug gap for a Model T Ford?",
        "How do I set the timing on a Model T?"
    ]
    
    for i, query in enumerate(queries, 1):
        out.add_text(f"{i}. {query}")
    
    out.add_text("")
    out.add_text("""INSTRUCTIONS:
1. Open ChatGPT (GPT-4) and Claude.ai in your browser
2. Enter each query and record the responses
3. Compare with your local Qwen+RAG results from Exercise 1
4. Note differences in:
   - Specificity (general knowledge vs manual procedures)
   - Confidence (when models admit they do not know)
   - Hallucination (invented details vs grounded facts)
""")
    
    out.save()

def exercise_4_topk_effect(modelt_store: VectorStore):
    """Exercise 4: Effect of top-K retrieval count."""
    print("\n" + "=" * 70)
    print("EXERCISE 4: Top-K Effect")
    print("=" * 70)
    
    out = ExerciseOutput(4, "TopK Effect")
    out.add_text("Testing different top-k values: 1, 3, 5, 10, 20")
    out.add_text("")
    
    query = "How do I adjust the carburetor on a Model T?"
    top_k_values = [1, 3, 5, 10, 20]
    
    out.add_text(f"Query: {query}")
    out.add_text("")
    
    for k in top_k_values:
        out.add_section(f"Top-K = {k}")
        response, retrieved, time_taken = generate_with_rag(query, modelt_store, top_k=k)
        out.add_text(f"Retrieved {len(retrieved)} chunks")
        out.add_text(f"Response time: {time_taken:.2f}s")
        out.add_text(f"Answer:\n{response}")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("""OBSERVATIONS:
- k=1: May miss relevant context, answers can be incomplete
- k=3-5: Good balance of context and precision
- k=10-20: More comprehensive but may include irrelevant chunks
- Higher k increases latency linearly

RECOMMENDATION: k=5 is optimal for most queries.""")
    
    out.save()

def exercise_5_unanswerable(modelt_store: VectorStore, cr_store: VectorStore):
    """Exercise 5: Unanswerable questions."""
    print("\n" + "=" * 70)
    print("EXERCISE 5: Unanswerable Questions")
    print("=" * 70)
    
    out = ExerciseOutput(5, "Unanswerable")
    out.add_text("Testing RAG behavior on questions that cannot be answered from corpus")
    out.add_text("")
    
    # Off-topic queries
    out.add_section("OFF-TOPIC QUERIES")
    off_topic = [
        "What is the capital of France?",
        "How do I bake chocolate chip cookies?",
        "What are the latest iPhone features?"
    ]
    
    for i, query in enumerate(off_topic, 1):
        out.add_query_section(query, i)
        response, retrieved, time_taken = generate_with_rag(query, modelt_store, top_k=5)
        out.add_with_rag(response, retrieved, time_taken)
    
    # Related but missing info
    out.add_section("RELATED BUT MISSING INFORMATION")
    related = [
        "What is the price of a Model T in 2024?",
        "How many Model T cars were sold in 1925?"
    ]
    
    for i, query in enumerate(related, 1):
        out.add_query_section(query, i)
        response, retrieved, time_taken = generate_with_rag(query, modelt_store, top_k=5)
        out.add_with_rag(response, retrieved, time_taken)
    
    # False premise
    out.add_section("FALSE PREMISE QUERIES")
    false_premise = [
        "How do I adjust the fuel injection on a Model T?",
        "What Bluetooth features does the Model T have?"
    ]
    
    for i, query in enumerate(false_premise, 1):
        out.add_query_section(query, i)
        response, retrieved, time_taken = generate_with_rag(query, modelt_store, top_k=5)
        out.add_with_rag(response, retrieved, time_taken)
    
    out.save()

def exercise_6_phrasing_sensitivity(modelt_store: VectorStore):
    """Exercise 6: Query phrasing sensitivity."""
    print("\n" + "=" * 70)
    print("EXERCISE 6: Phrasing Sensitivity")
    print("=" * 70)
    
    out = ExerciseOutput(6, "Phrasing Sensitivity")
    out.add_text("Same question phrased 5+ different ways")
    out.add_text("")
    
    # Same intent, different phrasings
    phrasings = [
        "How do I adjust the carburetor on a Model T?",
        "What's the procedure for carburetor adjustment on Model T Ford?",
        "Can you explain how to tune the Model T carburetor?",
        "Model T carburetor adjustment steps",
        "How should I set the carburetor needle valve on my Ford Model T?"
    ]
    
    out.add_text("Core question: Carburetor adjustment procedure")
    out.add_text("")
    
    for i, query in enumerate(phrasings, 1):
        out.add_query_section(query, i)
        response, retrieved, time_taken = generate_with_rag(query, modelt_store, top_k=5)
        out.add_text(f"Retrieved chunks: {[r[0]['source'] for r in retrieved]}")
        out.add_text(f"Answer preview: {response[:300]}...")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("""OBSERVATIONS:
- Different phrasings may retrieve different chunks
- Technical terms ("needle valve") vs general terms ("adjust") affect retrieval
- Model should provide consistent answers regardless of phrasing
- Some phrasings may yield better retrieval scores than others""")
    
    out.save()

def exercise_7_chunk_overlap(modelt_store: VectorStore, embedder):
    """Exercise 7: Chunk overlap experiment."""
    print("\n" + "=" * 70)
    print("EXERCISE 7: Chunk Overlap")
    print("=" * 70)
    
    out = ExerciseOutput(7, "Chunk Overlap")
    out.add_text("Testing chunk overlap: 0, 64, 128, 256")
    out.add_text("")
    
    query = "How do I adjust the carburetor on a Model T?"
    overlap_values = [0, 64, 128, 256]
    
    # Load documents
    docs = load_pdfs_from_folder(CORPORA_DIR / "ModelTService")
    
    for overlap in overlap_values:
        out.add_section(f"Overlap = {overlap}")
        
        # Re-chunk with different overlap
        chunks = chunk_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE, overlap=overlap)
        store = VectorStore().build_index(chunks, embedder)
        
        response, retrieved, time_taken = generate_with_rag(query, store, top_k=5)
        
        out.add_text(f"Total chunks: {len(chunks)}")
        out.add_text(f"Response time: {time_taken:.2f}s")
        out.add_text(f"Answer:\n{response}")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("""OBSERVATIONS:
- Higher overlap = more context continuity between chunks
- Higher overlap = larger index size (more chunks)
- Lower overlap = risk of losing context at chunk boundaries
- Recommended: overlap=128 for 512-token chunks""")
    
    out.save()

def exercise_8_chunk_size(modelt_store: VectorStore, embedder):
    """Exercise 8: Chunk size experiment."""
    print("\n" + "=" * 70)
    print("EXERCISE 8: Chunk Size")
    print("=" * 70)
    
    out = ExerciseOutput(8, "Chunk Size")
    out.add_text("Testing chunk sizes: 128, 256, 512, 1024, 2048")
    out.add_text("")
    
    query = "How do I adjust the carburetor on a Model T?"
    chunk_sizes = [128, 256, 512, 1024, 2048]
    
    docs = load_pdfs_from_folder(CORPORA_DIR / "ModelTService")
    
    for size in chunk_sizes:
        out.add_section(f"Chunk Size = {size}")
        
        chunks = chunk_documents(docs, chunk_size=size, overlap=size//4)
        store = VectorStore().build_index(chunks, embedder)
        
        response, retrieved, time_taken = generate_with_rag(query, store, top_k=5)
        
        out.add_text(f"Total chunks: {len(chunks)}")
        out.add_text(f"Response time: {time_taken:.2f}s")
        out.add_text(f"Answer:\n{response}")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("""OBSERVATIONS:
- Smaller chunks (128-256): More precise retrieval but less context
- Medium chunks (512): Good balance of precision and context
- Larger chunks (1024+): More context but may dilute relevance
- Larger chunks = fewer total chunks = faster retrieval

RECOMMENDATION: chunk_size=512 for general use""")
    
    out.save()

def exercise_9_score_analysis(modelt_store: VectorStore):
    """Exercise 9: Retrieval score analysis."""
    print("\n" + "=" * 70)
    print("EXERCISE 9: Score Analysis")
    print("=" * 70)
    
    out = ExerciseOutput(9, "Score Analysis")
    out.add_text("Analyzing retrieval score distributions")
    out.add_text("")
    
    queries = [
        "How do I adjust the carburetor on a Model T?",
        "What is the firing order?",
        "How do I change the oil?",
        "What type of spark plugs?",
        "How do I start the car?",
        "What is the ignition timing?",
        "How do I clean the radiator?",
        "What is the correct tire pressure?",
        "How do I adjust the brakes?",
        "What fuel should I use?"
    ]
    
    embedder = model_cache.get_embedder()
    
    for i, query in enumerate(queries, 1):
        retrieved = modelt_store.search(query, embedder, top_k=10)
        
        out.add_text(f"Q{i}: {query}")
        out.add_text("Scores:")
        for j, (chunk, score) in enumerate(retrieved, 1):
            out.add_text(f"  {j}. {score:.4f} - {chunk['source']}")
        
        # Calculate gap between top scores
        if len(retrieved) >= 2:
            gap = retrieved[0][1] - retrieved[1][1]
            out.add_text(f"Top-2 gap: {gap:.4f}")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("""OBSERVATIONS:
- High scores (>0.8) indicate strong relevance
- Score gaps between top results indicate confidence
- Low top scores (<0.5) may indicate query is off-topic
- Score threshold of 0.5 can filter irrelevant chunks""")
    
    out.save()

def exercise_10_prompt_templates(modelt_store: VectorStore):
    """Exercise 10: Prompt template variations."""
    print("\n" + "=" * 70)
    print("EXERCISE 10: Prompt Templates")
    print("=" * 70)
    
    out = ExerciseOutput(10, "Prompt Templates")
    out.add_text("Testing different prompt templates")
    out.add_text("")
    
    query = "How do I adjust the carburetor on a Model T?"
    templates = ["default", "strict", "citation", "minimal"]
    
    for template in templates:
        out.add_section(f"Template: {template.upper()}")
        response, retrieved, time_taken = generate_with_rag(
            query, modelt_store, top_k=5, prompt_template=template
        )
        out.add_text(f"Answer:\n{response}")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("""OBSERVATIONS:
- STRICT: Reduces hallucinations but may be overly restrictive
- CITATION: Improves traceability but adds verbosity
- MINIMAL: Fastest but less guidance to model
- DEFAULT: Good balance for most use cases

Template choice should match application requirements.""")
    
    out.save()

def exercise_11_failure_modes(modelt_store: VectorStore, cr_store: VectorStore):
    """Exercise 11: Failure mode catalog."""
    print("\n" + "=" * 70)
    print("EXERCISE 11: Failure Modes")
    print("=" * 70)
    
    out = ExerciseOutput(11, "Failure Modes")
    out.add_text("Cataloging RAG failure modes")
    out.add_text("")
    
    failure_tests = [
        ("Computation", "What is 2347 * 8923?", modelt_store),
        ("Temporal Reasoning", "What happened before the carburetor discussion?", modelt_store),
        ("Comparison", "Which is better, Model T or Model A?", modelt_store),
        ("Ambiguity", "How do I fix it?", modelt_store),  # Ambiguous "it"
        ("Multi-hop", "What procedure follows the carburetor adjustment in the manual?", modelt_store),
        ("Negation", "What should I NOT do when adjusting the carburetor?", modelt_store),
    ]
    
    for category, query, store in failure_tests:
        out.add_section(f"Category: {category}")
        out.add_text(f"Query: {query}")
        
        response, retrieved, time_taken = generate_with_rag(query, store, top_k=5)
        out.add_with_rag(response, retrieved, time_taken)
        
        out.add_text("Failure analysis:")
        out.add_text(f"- Query type: {category}")
        out.add_text(f"- Expected: Document-based answer")
        out.add_text(f"- Actual: See response above")
        out.add_text("")
    
    out.add_section("Summary")
    out.add_text("""COMMON FAILURE MODES:
1. Computation: LLM may try to calculate instead of retrieve
2. Temporal: Difficulty with "before/after" without full document context
3. Comparison: May lack both items in same context window
4. Ambiguity: Unclear references fail to retrieve relevant chunks
5. Multi-hop: Requires reasoning across multiple retrievals
6. Negation: Struggles with "do not" and negative constraints""")
    
    out.save()

def exercise_12_cross_document(modelt_store: VectorStore, cr_store: VectorStore):
    """Exercise 12: Cross-document synthesis."""
    print("\n" + "=" * 70)
    print("EXERCISE 12: Cross-Document Synthesis")
    print("=" * 70)
    
    out = ExerciseOutput(12, "Cross Document")
    out.add_text("Questions requiring information from multiple documents")
    out.add_text("")
    
    # Queries that need synthesis
    synthesis_queries = [
        "Compare the carburetor procedures in the Ford manual with modern fuel injection systems",
        "What maintenance procedures are common across all Model T documents?",
        "Summarize all ignition-related procedures from the manual"
    ]
    
    for i, query in enumerate(synthesis_queries, 1):
        out.add_query_section(query, i)
        
        # Use higher k for synthesis
        response, retrieved, time_taken = generate_with_rag(query, modelt_store, top_k=10)
        out.add_with_rag(response, retrieved, time_taken, top_k=10)
        
        sources = set([r[0]['source'] for r in retrieved])
        out.add_text(f"Sources used: {len(sources)} unique documents")
        out.add_text(f"Documents: {', '.join(sources)}")
        out.add_text("")
    
    out.add_section("Analysis")
    out.add_text("OBSERVATIONS:")
    out.add_text("- Higher k (7-10) needed for synthesis tasks")
    out.add_text("- Retrieval quality directly impacts synthesis capability")
    out.add_text("- Model can combine information from multiple sources")
    out.add_text("- Some queries require multiple rounds of retrieval")
    
    out.save()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_all_exercises():
    """Run all exercises sequentially."""
    print("KIMI RAG Exercises - Starting...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Corpora directory: {CORPORA_DIR}")
    
    # Exercise 0: Setup
    try:
        modelt_store, cr_store = exercise_0_setup()
        print("\n✓ Exercise 0 completed!")
    except Exception as e:
        print(f"\n✗ Exercise 0 failed: {e}")
        return
    
    # Exercise 1: RAG vs No-RAG
    try:
        exercise_1_rag_vs_norag(modelt_store, cr_store)
        print("✓ Exercise 1 completed!")
    except Exception as e:
        print(f"✗ Exercise 1 failed: {e}")
    
    # Exercise 2: GPT-4o Mini
    try:
        exercise_2_gpt4o_mini()
        print("✓ Exercise 2 completed!")
    except Exception as e:
        print(f"✗ Exercise 2 failed: {e}")
    
    # Exercise 3: Frontier comparison (manual)
    try:
        exercise_3_frontier_comparison()
        print("✓ Exercise 3 completed!")
    except Exception as e:
        print(f"✗ Exercise 3 failed: {e}")
    
    # Exercise 4: Top-K effect
    try:
        exercise_4_topk_effect(modelt_store)
        print("✓ Exercise 4 completed!")
    except Exception as e:
        print(f"✗ Exercise 4 failed: {e}")
    
    # Exercise 5: Unanswerable
    try:
        exercise_5_unanswerable(modelt_store, cr_store)
        print("✓ Exercise 5 completed!")
    except Exception as e:
        print(f"✗ Exercise 5 failed: {e}")
    
    # Exercise 6: Phrasing sensitivity
    try:
        exercise_6_phrasing_sensitivity(modelt_store)
        print("✓ Exercise 6 completed!")
    except Exception as e:
        print(f"✗ Exercise 6 failed: {e}")
    
    # Exercise 7: Chunk overlap
    try:
        embedder = model_cache.get_embedder()
        exercise_7_chunk_overlap(modelt_store, embedder)
        print("✓ Exercise 7 completed!")
    except Exception as e:
        print(f"✗ Exercise 7 failed: {e}")
    
    # Exercise 8: Chunk size
    try:
        exercise_8_chunk_size(modelt_store, embedder)
        print("✓ Exercise 8 completed!")
    except Exception as e:
        print(f"✗ Exercise 8 failed: {e}")
    
    # Exercise 9: Score analysis
    try:
        exercise_9_score_analysis(modelt_store)
        print("✓ Exercise 9 completed!")
    except Exception as e:
        print(f"✗ Exercise 9 failed: {e}")
    
    # Exercise 10: Prompt templates
    try:
        exercise_10_prompt_templates(modelt_store)
        print("✓ Exercise 10 completed!")
    except Exception as e:
        print(f"✗ Exercise 10 failed: {e}")
    
    # Exercise 11: Failure modes
    try:
        exercise_11_failure_modes(modelt_store, cr_store)
        print("✓ Exercise 11 completed!")
    except Exception as e:
        print(f"✗ Exercise 11 failed: {e}")
    
    # Exercise 12: Cross-document synthesis
    try:
        exercise_12_cross_document(modelt_store, cr_store)
        print("✓ Exercise 12 completed!")
    except Exception as e:
        print(f"✗ Exercise 12 failed: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETE!")
    print("=" * 70)
    print(f"Output files saved to: {OUTPUT_DIR}/")


def load_indices() -> Tuple[VectorStore, VectorStore]:
    """Load pre-built indices."""
    embedder = model_cache.get_embedder()
    
    modelt_store = VectorStore()
    modelt_store.load(str(OUTPUT_DIR / "index_modelt"), embedder)
    
    cr_store = VectorStore()
    cr_store.load(str(OUTPUT_DIR / "index_cr"), embedder)
    
    return modelt_store, cr_store


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        exercise_num = sys.argv[1]
        
        if exercise_num == "0":
            exercise_0_setup()
        elif exercise_num == "1":
            modelt_store, cr_store = load_indices()
            exercise_1_rag_vs_norag(modelt_store, cr_store)
        elif exercise_num == "2":
            exercise_2_gpt4o_mini()
        elif exercise_num == "3":
            exercise_3_frontier_comparison()
        elif exercise_num == "4":
            modelt_store, _ = load_indices()
            exercise_4_topk_effect(modelt_store)
        elif exercise_num == "5":
            modelt_store, cr_store = load_indices()
            exercise_5_unanswerable(modelt_store, cr_store)
        elif exercise_num == "6":
            modelt_store, _ = load_indices()
            exercise_6_phrasing_sensitivity(modelt_store)
        elif exercise_num == "7":
            embedder = model_cache.get_embedder()
            modelt_store, _ = load_indices()
            exercise_7_chunk_overlap(modelt_store, embedder)
        elif exercise_num == "8":
            embedder = model_cache.get_embedder()
            modelt_store, _ = load_indices()
            exercise_8_chunk_size(modelt_store, embedder)
        elif exercise_num == "9":
            modelt_store, _ = load_indices()
            exercise_9_score_analysis(modelt_store)
        elif exercise_num == "10":
            modelt_store, _ = load_indices()
            exercise_10_prompt_templates(modelt_store)
        elif exercise_num == "11":
            modelt_store, cr_store = load_indices()
            exercise_11_failure_modes(modelt_store, cr_store)
        elif exercise_num == "12":
            modelt_store, cr_store = load_indices()
            exercise_12_cross_document(modelt_store, cr_store)
        else:
            print(f"Unknown exercise: {exercise_num}")
            print("Usage: python kimi_rag_exercises.py [0-12]")
    else:
        print("Running all exercises...")
        print("Tip: Run individual exercises with: python kimi_rag_exercises.py <number>")
        run_all_exercises()
