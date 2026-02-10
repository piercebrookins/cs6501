"""
Exercise 4: Effect of Top-K Retrieval Count

Vary the number of chunks retrieved and observe how it affects answer quality.
Tests k = 1, 3, 5, 10, 20.
"""

from typing import List, Dict, Any
import json
import time
from dataclasses import dataclass, field


@dataclass
class TopKResult:
    """Result for a single top-k configuration."""
    k: int
    query: str
    num_chunks: int
    avg_score: float
    max_score: float
    min_score: float
    response_time: float
    context_length: int


class Exercise4_TopK:
    """
    Exercise 4: Effect of Top-K Retrieval Count
    
    Tests: k = 1, 3, 5, 10, 20
    
    For each k:
    - Run the same 3-5 queries
    - Note answer quality, completeness, accuracy
    - Note response latency
    
    Questions to explore:
    - At what point does adding more context stop helping?
    - When does too much context hurt (irrelevant information, confusion)?
    - How does k interact with chunk size?
    """
    
    K_VALUES = [1, 3, 5, 10, 20]
    
    def __init__(self, pipeline):
        """
        Initialize Exercise 4.
        
        Args:
            pipeline: Initialized RAGPipeline
        """
        self.pipeline = pipeline
        self.results: Dict[int, List[TopKResult]] = {k: [] for k in self.K_VALUES}
    
    def run_experiment(
        self,
        queries: List[str],
        k_values: Optional[List[int]] = None
    ) -> Dict[int, List[TopKResult]]:
        """
        Run top-k experiment across different k values.
        
        Args:
            queries: List of test queries
            k_values: Override default k values
            
        Returns:
            Dict mapping k value to list of results
        """
        k_list = k_values if k_values is not None else self.K_VALUES
        
        print(f"\n{'='*60}")
        print("Exercise 4: Top-K Retrieval Experiment")
        print(f"{'='*60}")
        print(f"Testing k values: {k_list}")
        print(f"Queries: {len(queries)}")
        
        for k in k_list:
            print(f"\n{'-'*40}")
            print(f"Testing k={k}")
            print(f"{'-'*40}")
            
            for query in queries:
                result = self._test_single_query(query, k)
                self.results[k].append(result)
                
                print(f"  Q: {query[:50]}...")
                print(f"     Retrieved: {result.num_chunks}, "
                      f"Avg score: {result.avg_score:.3f}, "
                      f"Time: {result.response_time:.3f}s")
        
        return self.results
    
    def _test_single_query(self, query: str, k: int) -> TopKResult:
        """Test a single query with specific k value."""
        start_time = time.time()
        
        # Retrieve chunks
        retrieved = self.pipeline.retrieve(query, top_k=k)
        
        response_time = time.time() - start_time
        
        # Calculate statistics
        scores = [c.get('score', 0) for c in retrieved]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        # Calculate context length
        context = self.pipeline.format_context(retrieved)
        context_length = len(context)
        
        return TopKResult(
            k=k,
            query=query,
            num_chunks=len(retrieved),
            avg_score=avg_score,
            max_score=max_score,
            min_score=min_score,
            response_time=response_time,
            context_length=context_length
        )
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze top-k experiment results.
        
        Returns:
            Analysis dict with findings about optimal k,
            diminishing returns, and latency trade-offs.
        """
        if not any(self.results.values()):
            return {"error": "No results to analyze. Run experiment first."}
        
        analysis = {
            "per_k_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        for k, results in self.results.items():
            if not results:
                continue
            
            avg_time = sum(r.response_time for r in results) / len(results)
            avg_context = sum(r.context_length for r in results) / len(results)
            avg_score = sum(r.avg_score for r in results) / len(results)
            
            analysis["per_k_analysis"][k] = {
                "avg_response_time": avg_time,
                "avg_context_length": avg_context,
                "avg_retrieval_score": avg_score,
                "total_queries": len(results)
            }
        
        # Generate findings
        k_values = sorted(analysis["per_k_analysis"].keys())
        if len(k_values) >= 2:
            # Check for diminishing returns
            scores = [analysis["per_k_analysis"][k]["avg_retrieval_score"] 
                     for k in k_values]
            
            # Find where score improvement plateaus
            for i in range(1, len(scores)):
                improvement = scores[i] - scores[i-1]
                if improvement < 0.01:  # Less than 0.01 improvement
                    analysis["findings"].append(
                        f"Diminishing returns observed at k={k_values[i]} "
                        f"(improvement: {improvement:.4f})"
                    )
                    analysis["recommendations"].append(
                        f"Consider k={k_values[i-1]} for optimal efficiency"
                    )
                    break
        
        return analysis
    
    def save_report(self, output_path: str = "exercise4_report.json"):
        """Save results to JSON report."""
        report = {
            "exercise": "Exercise 4: Effect of Top-K Retrieval Count",
            "k_values_tested": self.K_VALUES,
            "results": {
                k: [
                    {
                        "query": r.query,
                        "num_chunks": r.num_chunks,
                        "avg_score": r.avg_score,
                        "max_score": r.max_score,
                        "min_score": r.min_score,
                        "response_time": r.response_time,
                        "context_length": r.context_length
                    }
                    for r in results
                ]
                for k, results in self.results.items()
            },
            "analysis": self.analyze_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {output_path}")
        return report


# Import Optional at module level
from typing import Optional
