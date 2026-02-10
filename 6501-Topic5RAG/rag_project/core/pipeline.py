"""
Core RAG Pipeline Implementation

This module implements the main RAG pipeline from the course notebook,
refactored for clarity and reusability across exercises.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline.
    
    Pipeline flow:
    Documents → Chunking → Embedding → Index (FAISS)
                                            ↓
    User Query → Embed Query → Similarity Search → Top-K Chunks
                                                        ↓
                                    Prompt Assembly → LLM → Answer
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        top_k: int = 5,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the sentence transformer model
            llm_model_name: Name of the LLM to use
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
            top_k: Number of chunks to retrieve
            device: Compute device ('cuda', 'mps', 'cpu', or None for auto)
            dtype: PyTorch dtype (None for auto)
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Auto-detect device if not specified
        if device is None:
            self.device, self.dtype = self._detect_device()
        else:
            self.device = device
            self.dtype = dtype if dtype else torch.float32
        
        # Components (initialized on demand)
        self._embedding_model: Optional[SentenceTransformer] = None
        self._llm = None
        self._tokenizer = None
        self._index: Optional[faiss.Index] = None
        self._chunks: List[str] = []
        self._chunk_metadata: List[Dict] = []
        
    def _detect_device(self) -> Tuple[str, torch.dtype]:
        """Detect best available device."""
        if torch.cuda.is_available():
            device = 'cuda'
            dtype = torch.float16
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
            dtype = torch.float32  # float32 is often faster on Apple Silicon
        else:
            device = 'cpu'
            dtype = torch.float32
        return device, dtype
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
        return self._embedding_model
    
    def load_documents(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Load PDF documents from a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            List of document dicts with 'text', 'source', and 'page' keys
        """
        folder = Path(folder_path)
        documents = []
        
        pdf_files = list(folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {folder}")
        
        for pdf_path in pdf_files:
            try:
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        documents.append({
                            'text': text,
                            'source': pdf_path.name,
                            'page': page_num + 1,
                            'path': str(pdf_path)
                        })
                doc.close()
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
        
        print(f"Loaded {len(documents)} pages from {len(pdf_files)} documents")
        return documents
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of document dictionaries
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            Tuple of (chunks, metadata)
        """
        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        
        chunks = []
        metadata = []
        
        for doc in documents:
            text = doc['text']
            source = doc['source']
            page = doc.get('page', 1)
            
            # Simple sliding window chunking
            start = 0
            chunk_idx = 0
            
            while start < len(text):
                end = min(start + size, len(text))
                
                # Try to break at sentence boundary
                if end < len(text):
                    # Look for period, question mark, or newline
                    for delim in ['. ', '? ', '! ', '\n', ' ']:
                        pos = text.rfind(delim, start, end)
                        if pos != -1 and pos > start + size // 2:
                            end = pos + len(delim)
                            break
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    metadata.append({
                        'source': source,
                        'page': page,
                        'chunk_index': chunk_idx,
                        'start_char': start,
                        'end_char': end
                    })
                    chunk_idx += 1
                
                # Move start forward by chunk_size - overlap
                start += size - overlap
                
                # Prevent infinite loop when overlap >= size
                if start <= 0 or (overlap >= size and end >= len(text)):
                    break
        
        print(f"Created {len(chunks)} chunks (size={size}, overlap={overlap})")
        self._chunks = chunks
        self._chunk_metadata = metadata
        return chunks, metadata
    
    def build_index(self, chunks: Optional[List[str]] = None) -> faiss.Index:
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of text chunks (uses self._chunks if None)
            
        Returns:
            FAISS index
        """
        if chunks is None:
            chunks = self._chunks
        
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
        index.add(embeddings)
        
        self._index = index
        print(f"Built index with {index.ntotal} vectors (dim={dimension})")
        return index
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            return_scores: Whether to include similarity scores
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        if self._index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        k = top_k if top_k is not None else self.top_k
        
        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self._index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._chunks):
                continue
            result = {
                'text': self._chunks[idx],
                'metadata': self._chunk_metadata[idx],
                'rank': i + 1,
                'index': int(idx)
            }
            if return_scores:
                result['score'] = float(score)
            results.append(result)
        
        return results
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string for LLM."""
        context_parts = []
        for chunk in retrieved_chunks:
            meta = chunk['metadata']
            source = f"[Source: {meta['source']}, Page: {meta['page']}]"
            context_parts.append(f"{source}\n{chunk['text']}")
        return "\n\n---\n\n".join(context_parts)
    
    def create_prompt(
        self,
        query: str,
        context: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Create prompt for LLM.
        
        Args:
            query: User query
            context: Retrieved context
            prompt_template: Custom prompt template (uses default if None)
            
        Returns:
            Formatted prompt
        """
        if prompt_template is None:
            prompt_template = """You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain the answer, say "I cannot answer this from the available documents."

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt_template.format(context=context, query=query)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        prompt_template: Optional[str] = None,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Execute full RAG query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            prompt_template: Custom prompt template
            return_context: Whether to return retrieved context
            
        Returns:
            Dict with 'query', 'prompt', 'retrieved_chunks', and optionally 'context'
        """
        # Retrieve
        retrieved = self.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.format_context(retrieved)
        
        # Create prompt
        prompt = self.create_prompt(query, context, prompt_template)
        
        result = {
            'query': query,
            'prompt': prompt,
            'retrieved_chunks': retrieved,
        }
        
        if return_context:
            result['context'] = context
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'chunks': len(self._chunks),
            'index_size': self._index.ntotal if self._index else 0,
            'embedding_model': self.embedding_model_name,
            'llm_model': self.llm_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k': self.top_k,
            'device': self.device
        }
