#!/usr/bin/env python3
"""
Manual RAG Pipeline - Standalone Python Script

This script builds a complete RAG (Retrieval-Augmented Generation) pipeline
from scratch, demonstrating each step explicitly.

Usage:
    python manual_rag_pipeline.py

The script will open a folder picker dialog to select your documents folder.
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Enable MPS fallback BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np


def get_device() -> Tuple[str, torch.dtype]:
    """
    Detect the best available compute device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        Tuple of (device_string, recommended_dtype)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        dtype = torch.float16
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Using CUDA GPU: {device_name} ({memory_gb:.1f} GB)")
        
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        dtype = torch.float32  # float32 is often faster on Apple Silicon
        print("✓ Using Apple Silicon GPU (MPS)")
        print("  Note: Using float32 (faster than float16 on Apple Silicon)")
        
    else:
        device = 'cpu'
        dtype = torch.float32
        print("⚠ Using CPU (no GPU detected)")
        print("  Tip: For faster processing, use a machine with a GPU")
    
    return device, dtype


# =============================================================================
# FOLDER SELECTION
# =============================================================================

def select_folder_with_picker() -> Optional[str]:
    """
    Open a folder picker dialog using tkinter.
    
    Note: tkinter's askdirectory() is BLOCKING - the script pauses until
    the user selects a folder and closes the dialog. This is different from
    ipyfilechooser in Jupyter notebooks, which is non-blocking.
    
    For environments without tkinter (some servers), falls back to manual input.
    
    Returns:
        Selected folder path, or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create and hide the root window
        root = tk.Tk()
        root.withdraw()
        
        # Bring the dialog to the front (especially important on macOS)
        root.attributes('-topmost', True)
        
        # macOS-specific: ensure dialog appears
        if sys.platform == 'darwin':
            root.call('wm', 'attributes', '.', '-topmost', True)
        
        print("📁 Opening folder picker dialog...")
        print("   (If you don't see it, check behind other windows)")
        
        # This call BLOCKS until user selects or cancels
        folder_path = filedialog.askdirectory(
            title='Select your documents folder',
            initialdir=str(Path.home())
        )
        
        root.destroy()
        
        if folder_path:
            return folder_path
        else:
            print("   No folder selected (dialog was cancelled)")
            return None
            
    except ImportError:
        print("tkinter not available - falling back to manual input")
        return None
    except Exception as e:
        print(f"Error opening folder picker: {e}")
        return None


def select_folder_manual() -> str:
    """Prompt user to enter folder path manually."""
    print("\nEnter the path to your documents folder:")
    print("  (or press Enter to use './documents')")
    
    user_input = input("> ").strip()
    
    if not user_input:
        return "documents"
    else:
        return user_input


def select_document_folder() -> str:
    """
    Select document folder using picker with fallback to manual entry.
    
    Returns:
        Path to the selected documents folder
    """
    print("\n" + "=" * 60)
    print("DOCUMENT FOLDER SELECTION")
    print("=" * 60)
    
    # Try the GUI picker first
    folder = select_folder_with_picker()
    
    # Fall back to manual entry if picker failed or was cancelled
    if not folder:
        folder = select_folder_manual()
    
    # Validate the folder exists
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"\n⚠ Folder does not exist: {folder}")
        create = input("Create it? (y/n): ").strip().lower()
        if create == 'y':
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created folder: {folder}")
        else:
            print("Please add documents to the folder and run again.")
            sys.exit(1)
    
    print(f"\n✓ Using folder: {folder}")
    return str(folder_path.absolute())


def list_documents(doc_folder: str) -> List[Path]:
    """List supported documents in the folder."""
    doc_path = Path(doc_folder)
    supported_extensions = ['.pdf', '.txt', '.md', '.text']
    
    files_found = [f for f in doc_path.glob('*') if f.is_file()]
    supported = [f for f in files_found if f.suffix.lower() in supported_extensions]
    
    print(f"\nDocuments in '{doc_folder}':")
    print("-" * 50)
    
    if supported:
        for f in supported:
            size_kb = f.stat().st_size / 1024
            print(f"  ✓ {f.name} ({size_kb:.1f} KB)")
        print(f"\nTotal: {len(supported)} supported file(s)")
    else:
        print("  (no PDF or TXT files found)")
        if files_found:
            print(f"  Found {len(files_found)} other file(s) - only .pdf, .txt, .md supported")
    
    return supported


# =============================================================================
# STAGE 1: DOCUMENT LOADING
# =============================================================================

def load_text_file(filepath: str) -> str:
    """Load a plain text file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_pdf_file(filepath: str) -> str:
    """
    Extract text from a PDF with embedded text.
    
    Uses PyMuPDF (fitz) to read the text layer directly.
    For scanned PDFs without embedded text, you'd need OCR.
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(filepath)
    text_parts = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"\n[Page {page_num + 1}]\n{text}")
    
    doc.close()
    return "\n".join(text_parts)


def load_documents(doc_folder: str) -> List[Tuple[str, str]]:
    """Load all documents from a folder. Returns list of (filename, content)."""
    print("\n" + "=" * 60)
    print("STAGE 1: DOCUMENT LOADING")
    print("=" * 60)
    
    documents = []
    folder = Path(doc_folder)
    
    for filepath in folder.rglob("*"):
        if filepath.is_file():
            try:
                if filepath.suffix.lower() == '.pdf':
                    content = load_pdf_file(str(filepath))
                elif filepath.suffix.lower() in ['.txt', '.md', '.text']:
                    content = load_text_file(str(filepath))
                else:
                    continue
                
                if content.strip():
                    documents.append((filepath.name, content))
                    print(f"✓ Loaded: {filepath.name} ({len(content):,} chars)")
            except Exception as e:
                print(f"✗ Error loading {filepath}: {e}")
    
    print(f"\nTotal: {len(documents)} document(s) loaded")
    return documents


# =============================================================================
# STAGE 2: CHUNKING
# =============================================================================

@dataclass
class Chunk:
    """A chunk of text with metadata for tracing back to source."""
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    We try to break at sentence or paragraph boundaries
    to avoid cutting mid-thought.
    """
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a good boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind('\n\n', start + chunk_size // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', start + chunk_size // 2, end)
                if sentence_break != -1:
                    end = sentence_break + 2
        
        chunk_text_str = text[start:end].strip()
        
        if chunk_text_str:
            chunks.append(Chunk(
                text=chunk_text_str,
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end
            ))
            chunk_index += 1
        
        # Move forward, accounting for overlap
        start = end - chunk_overlap
        if chunks and start <= chunks[-1].start_char:
            start = end  # Safety: ensure progress
    
    return chunks


def chunk_documents(
    documents: List[Tuple[str, str]],
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> List[Chunk]:
    """Chunk all documents."""
    print("\n" + "=" * 60)
    print("STAGE 2: CHUNKING")
    print("=" * 60)
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    all_chunks = []
    for filename, content in documents:
        doc_chunks = chunk_text(content, filename, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
        print(f"  {filename}: {len(doc_chunks)} chunks")
    
    print(f"\nTotal: {len(all_chunks)} chunks")
    return all_chunks


# =============================================================================
# STAGE 3: EMBEDDING
# =============================================================================

def load_embedding_model(device: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the embedding model."""
    print("\n" + "=" * 60)
    print("STAGE 3: EMBEDDING")
    print("=" * 60)
    
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading embedding model: {model_name}")
    print(f"Device: {device}")
    
    # Must explicitly pass device for MPS support
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    
    return model, embedding_dim


def embed_chunks(chunks: List[Chunk], embed_model) -> np.ndarray:
    """Embed all chunks."""
    print(f"\nEmbedding {len(chunks)} chunks...")
    
    chunk_texts = [c.text for c in chunks]
    embeddings = embed_model.encode(chunk_texts, show_progress_bar=True)
    embeddings = embeddings.astype('float32')  # FAISS wants float32
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# =============================================================================
# STAGE 4: VECTOR INDEX (FAISS)
# =============================================================================

def build_faiss_index(embeddings: np.ndarray, embedding_dim: int):
    """Build FAISS index for similarity search."""
    print("\n" + "=" * 60)
    print("STAGE 4: VECTOR INDEX (FAISS)")
    print("=" * 60)
    
    import faiss
    
    # IndexFlatIP = Inner Product (for cosine similarity on normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Normalize vectors so inner product = cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add vectors to index
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors")
    
    return index


# =============================================================================
# STAGE 5: RETRIEVAL
# =============================================================================

class Retriever:
    """Retrieves relevant chunks for a query."""
    
    def __init__(self, embed_model, index, chunks: List[Chunk]):
        self.embed_model = embed_model
        self.index = index
        self.chunks = chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the top-k most relevant chunks for a query.
        
        Returns: List of (chunk, similarity_score) tuples
        """
        import faiss
        
        # Embed the query
        query_embedding = self.embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))
        
        return results


# =============================================================================
# STAGE 6: LLM GENERATION
# =============================================================================

def load_llm(device: str, dtype: torch.dtype, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the language model for generation."""
    print("\n" + "=" * 60)
    print("STAGE 6: LLM GENERATION")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading LLM: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("This may take a few minutes on first run...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load with appropriate settings for each device type
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        print("Model loaded on CUDA")
        
    elif device == 'mps':
        # For MPS, load to CPU first, then move to MPS
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model = model.to(device)
        print("Model loaded on MPS (Apple Silicon)")
        
    else:
        # CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        print("Model loaded on CPU (this will be slow)")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3
) -> str:
    """Generate a response from the LLM."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the correct device
    if device == 'cuda':
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


# =============================================================================
# STAGE 7: RAG PIPELINE
# =============================================================================

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question based ONLY on the information in the context above
- If the context doesn't contain enough information to answer, say so
- Quote relevant parts of the context to support your answer
- Be concise and direct

ANSWER:"""


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(self, retriever: Retriever, model, tokenizer, device: str):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        show_context: bool = False
    ) -> str:
        """
        Answer a question using RAG.
        
        1. Retrieve relevant chunks
        2. Build prompt with context
        3. Generate answer
        """
        # Step 1: Retrieve
        results = self.retriever.retrieve(question, top_k)
        
        # Format context
        context_parts = []
        for chunk, score in results:
            context_parts.append(f"[Source: {chunk.source_file}, Relevance: {score:.3f}]\n{chunk.text}")
        context = "\n\n---\n\n".join(context_parts)
        
        if show_context:
            print("=" * 60)
            print("RETRIEVED CONTEXT:")
            print("=" * 60)
            print(context)
            print("=" * 60 + "\n")
        
        # Step 2: Build prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Step 3: Generate
        answer = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            self.device
        )
        
        return answer


# =============================================================================
# INTERACTIVE QUERY LOOP
# =============================================================================

def interactive_loop(pipeline: RAGPipeline):
    """Run an interactive query loop."""
    print("\n" + "=" * 60)
    print("RAG PIPELINE READY")
    print("=" * 60)
    print("Enter your questions below. Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'debug' - Toggle showing retrieved context")
    print("  'topk N' - Set number of chunks to retrieve (e.g., 'topk 10')")
    print("=" * 60)
    
    show_context = False
    top_k = 5
    
    while True:
        try:
            print()
            question = input("Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'debug':
                show_context = not show_context
                print(f"Context display: {'ON' if show_context else 'OFF'}")
                continue
            
            if question.lower().startswith('topk '):
                try:
                    top_k = int(question.split()[1])
                    print(f"Top-K set to: {top_k}")
                except (IndexError, ValueError):
                    print("Usage: topk N (e.g., 'topk 10')")
                continue
            
            print("\nGenerating answer...")
            start_time = time.time()
            
            answer = pipeline.query(question, top_k=top_k, show_context=show_context)
            
            elapsed = time.time() - start_time
            print(f"\nAnswer ({elapsed:.1f}s):")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("MANUAL RAG PIPELINE")
    print("=" * 60)
    
    # Detect device
    device, dtype = get_device()
    
    # Select document folder
    doc_folder = select_document_folder()
    
    # List documents
    supported_files = list_documents(doc_folder)
    if not supported_files:
        print("\n⚠ No supported documents found. Please add PDF or TXT files.")
        sys.exit(1)
    
    input("\nPress Enter to continue with pipeline setup...")
    
    # Stage 1: Load documents
    documents = load_documents(doc_folder)
    if not documents:
        print("\n⚠ No documents could be loaded.")
        sys.exit(1)
    
    # Stage 2: Chunk documents
    chunks = chunk_documents(documents)
    
    # Stage 3: Load embedding model and embed chunks
    embed_model, embedding_dim = load_embedding_model(device)
    embeddings = embed_chunks(chunks, embed_model)
    
    # Stage 4: Build FAISS index
    index = build_faiss_index(embeddings, embedding_dim)
    
    # Stage 5: Create retriever
    retriever = Retriever(embed_model, index, chunks)
    
    # Stage 6: Load LLM
    model, tokenizer = load_llm(device, dtype)
    
    # Stage 7: Create RAG pipeline
    pipeline = RAGPipeline(retriever, model, tokenizer, device)
    
    # Run interactive loop
    interactive_loop(pipeline)


if __name__ == "__main__":
    main()
