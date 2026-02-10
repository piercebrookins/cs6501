"""
Exercise 8: Chunk Size Experiment

Test how chunk size affects retrieval precision and answer quality.
Chunk sizes: 128, 256, 512, 1024, 2048 characters.
"""

from typing import List, Dict, Any, Tuple
import json
import time
from dataclasses import dataclass


@dataclass
class ChunkSizeResult:
    """Result for a single chunk size configuration."""
    chunk_size: int
    num_chunks: int
    query: str
    avg_score: float
    precision: float  # Relevance of top results
    coverage: float   # Whether answer is in retrieved chunks
    index_time: float
    query_time: float


class Exercise8_ChunkSize:
    """
    Exercise 8: Chunk Size Experiment
    
    Tests chunk sizes: 128, 256, 512, 1024, 2048 characters
    
    For each configuration:
    1. Rebuild the index
    2. Run the same set of 5 queries
    3. Examine retrieved chunks and final answers
    
    Questions to explore:
    - How does chunk size affect retrieval precision?
    - How does it affect answer completeness?
    - Is there a sweet spot for your corpus?
    - Does optimal size depend on the type of question?
    """
    
    CHUNK_SIZES = [128, 256, 512, 1024, 2048]
    
    def __init__(self, base_pipeline, documents: List[Dict]):
        """
        Initialize Exercise 8.
        
        Args:
            base_pipeline: RAGPipeline instance to use as template
            documents: Pre-loaded documents to chunk and index
        """
        self.base_pipeline = base_pipeline
        self.documents = documents
        self.results: Dict[int, List[ChunkSizeResult]] = {
            size: [] for size in self.CHUNK_SIZES
        }
    
    def run_experiment(
        self,
        queries: List[str],
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 0
    ) -> Dict[int, List[ChunkSizeResult]]:
        """
        Run chunk size experiment.
        
        Args:
            queries: List of test queries
            chunk_sizes: Override default chunk sizes
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dict mapping chunk size to list of results
        """
        sizes = chunk_sizes if chunk_sizes is not None else self.CHUNK_SIZES
        
        print(f"\n{'='*60}")
        print("Exercise 8: Chunk Size Experiment")
        print(f"{'='*60}")
        print(f"Testing chunk sizes: {sizes}")
        print(f"Overlap: {chunk_overlap}")
        print(f"Queries: {len(queries)}")
        
        for size in sizes:
            print(f"\n{'-'*40}")
            print(f"Testing chunk_size={size}")
            print(f"{'-'*40}")
            
            # Create pipeline with this chunk size
            from ..core.pipeline import RAGPipeline
            
            start_total = time.time()
            
            pipeline = RAGPipeline(
                embedding_model=self.base_pipeline.embedding_model_name,
                llm_model_name=self.base_pipeline.llm_model_name,
                chunk_size=size,
                chunk_overlap=chunk_overlap,
                top_k=self.base_pipeline.top_k,
                device=self.base_pipeline.device,
                dtype=self.base_pipeline.dtype
            )
            
            # Chunk and index
            start_index = time.time()
            chunks, metadata = pipeline.chunk_documents(self.documents)
            pipeline.build_index(chunks)
            index_time = time.time() - start_index
            
            print(f"  Created {len(chunks)} chunks in {index_time:.2f}s")
            
            # Test each query
            for query in queries:
                result = self._test_single_query(
                    pipeline, query, size, len(chunks), index_time
                )
                self.results[size].append(result)
                
                print(f"  Q: {query[:50]}...")
                print(f"     Avg score: {result.avg_score:.3f}, "
                      f"Query time: {result.query_time:.3f}s")
        
        return self.results
    
    def _test_single_query(
        self,
        pipeline: 'RAGPipeline',
        query: str,
        chunk_size: int,
        num_chunks: int,
        index_time: float
    ) -> ChunkSizeResult:
        """Test a single query with specific chunk size."""
        start_time = time.time()
        
        # Retrieve chunks
        retrieved = pipeline.retrieve(query)
        
        query_time = time.time() - start_time
        
        # Calculate statistics
        scores = [c.get('score', 0) for c in retrieved]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Estimate precision (how many of top 3 are relevant)
        # This is a heuristic - in practice, you'd evaluate manually
        precision = self._estimate_precision(retrieved, query)
        
        # Estimate coverage (whether retrieved chunks likely contain answer)
        coverage = self._estimate_coverage(retrieved, query)
        
        return ChunkSizeResult(
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            query=query,
            avg_score=avg_score,
            precision=precision,
            coverage=coverage,
            index_time=index_time,
            query_time=query_time
        )
    
    def _estimate_precision(self, retrieved: List[Dict], query: str) -> float:
        """
        Estimate precision of retrieved chunks.
        
        This is a simple heuristic based on keyword overlap.
        In practice, you'd want manual evaluation.
        """
        query_terms = set(query.lower().split())
        scores = []
        
        for chunk in retrieved[:3]:  # Top 3
            chunk_text = chunk['text'].lower()
            matches = sum(1 for term in query_terms if term in chunk_text)
            scores.append(min(matches / max(len(query_terms), 1), 1.0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _estimate_coverage(self, retrieved: List[Dict], query: str) -> float:
        """
        Estimate whether answer is likely in retrieved chunks.
        
        Based on query term coverage across all retrieved chunks.
        """
        query_terms = set(query.lower().split())
        all_text = ' '.join(c['text'].lower() for c in retrieved)
        
        matches = sum(1 for term in query_terms if term in all_text)
        return matches / max(len(query_terms), 1)
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze chunk size experiment results.
        
        Returns:
            Analysis dict with findings about optimal chunk size.
        """
        if not any(self.results.values()):
            return {"error": "No results to analyze. Run experiment first."}
        
        analysis = {
            "per_size_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        for size, results in self.results.items():
            if not results:
                continue
            
            avg_precision = sum(r.precision for r in results) / len(results)
            avg_coverage = sum(r.coverage for r in results) / len(results)
            avg_query_time = sum(r.query_time for r in results) / len(results)
            total_chunks = results[0].num_chunks if results else 0
            
            analysis["per_size_analysis"][size] = {
                "avg_precision": avg_precision,
                "avg_coverage": avg_coverage,
                "avg_query_time": avg_query_time,
                "total_chunks": total_chunks
            }
        
        # Find optimal chunk size
        sizes = sorted(analysis["per_size_analysis"].keys())
        if sizes:
            # Balance precision, coverage, and efficiency
            best_size = None
            best_score = 0
            
            for size in sizes:
                stats = analysis["per_size_analysis"][size]
                # Combined score: precision * coverage / log(chunks)
                # Rewards good retrieval, penalizes too many chunks
                score = (stats["avg_precision"] * stats["avg_coverage"]) / (
                    1 + 0.1 * (stats["total_chunks"] / 100)
                )
                
                if score > best_score:
                    best_score = score
                    best_size = size
            
            analysis["optimal_chunk_size"] = best_size
            analysis["recommendations"].append(
                f"Optimal chunk size appears to be {best_size} characters "
                f"(score: {best_score:.3f})"
            )
        
        return analysis
    
    def save_report(self, output_path: str = "exercise8_report.json"):
        """Save results to JSON report."""
        report = {
            "exercise": "Exercise 8: Chunk Size Experiment",
            "chunk_sizes_tested": self.CHUNK_SIZES,
            "results": {
                size: [
                    {
                        "num_chunks": r.num_chunks,
                        "query": r.query,
                        "avg_score": r.avg_score,
                        "precision": r.precision,
                        "coverage": r.coverage,
                        "index_time": r.index_time,
                        "query_time": r.query_time
                    }
                    for r in results
                ]
                for size, results in self.results.items()
            },
            "analysis": self.analyze_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {output_path}")
        return report


# Import Optional at module level
from typing import Optional
