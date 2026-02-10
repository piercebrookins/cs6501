"""
Exercise 2: Open Model + RAG vs. Large Model Comparison

Compare small open model with RAG against larger model without RAG.
Tests GPT-4o Mini against Qwen 2.5 1.5B with RAG.
"""

from typing import List, Dict, Any
import json
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result comparing two approaches."""
    query: str
    small_model_rag: str
    large_model_no_rag: str
    small_model_hallucinated: bool
    large_model_hallucinated: bool


class Exercise2_OpenVsLarge:
    """
    Exercise 2: Open Model + RAG vs. Large Model
    
    Compare Qwen 2.5 1.5B with RAG vs GPT-4o Mini without tools.
    
    Document:
    - Does GPT 4o Mini do a better job than Qwen in avoiding hallucinations?
    - Which questions does GPT 4o Mini answer correctly?
    - Compare the cut-off date of GPT 4o Mini pre-training and the age
      of the Model T Ford and Congressional Record corpora.
    """
    
    def __init__(self, rag_pipeline, large_model_generator):
        """
        Initialize Exercise 2.
        
        Args:
            rag_pipeline: RAGPipeline with small model
            large_model_generator: Function to query large model (e.g., GPT-4o Mini)
        """
        self.rag_pipeline = rag_pipeline
        self.large_model_generator = large_model_generator
        self.results: List[ComparisonResult] = []
    
    def run_comparison(self, queries: List[str]) -> List[ComparisonResult]:
        """Run comparison on list of queries."""
        print(f"\n{'='*60}")
        print("Exercise 2: Open Model + RAG vs. Large Model")
        print(f"{'='*60}")
        
        for query in queries:
            print(f"\nQ: {query}")
            
            # Small model with RAG
            rag_result = self.rag_pipeline.query(query)
            small_response = self._query_small_model(rag_result['prompt'])
            
            # Large model without RAG
            large_response = self.large_model_generator(query)
            
            # Analyze for hallucinations (simplified)
            small_hallucinated = self._detect_hallucination(small_response)
            large_hallucinated = self._detect_hallucination(large_response)
            
            result = ComparisonResult(
                query=query,
                small_model_rag=small_response,
                large_model_no_rag=large_response,
                small_model_hallucinated=small_hallucinated,
                large_model_hallucinated=large_hallucinated
            )
            self.results.append(result)
            
            print(f"  Small+RAG: {small_response[:100]}...")
            print(f"  Large: {large_response[:100]}...")
        
        return self.results
    
    def _query_small_model(self, prompt: str) -> str:
        """Query small model - implement with actual LLM."""
        return "[Implement small model query]"
    
    def _detect_hallucination(self, response: str) -> bool:
        """
        Detect potential hallucination.
        
        This is a simplified heuristic. In practice, you'd want
        more sophisticated detection or manual review.
        """
        # Look for uncertainty indicators
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "i cannot",
            "not mentioned", "not specified"
        ]
        
        response_lower = response.lower()
        has_uncertainty = any(p in response_lower for p in uncertainty_phrases)
        
        # Look for specific values (potential hallucination indicators for this domain)
        # This would need domain-specific logic
        
        return not has_uncertainty  # Simplified
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze comparison results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        total = len(self.results)
        small_hallucinations = sum(1 for r in self.results if r.small_model_hallucinated)
        large_hallucinations = sum(1 for r in self.results if r.large_model_hallucinated)
        
        return {
            "total_queries": total,
            "small_model_hallucinations": small_hallucinations,
            "large_model_hallucinations": large_hallucinations,
            "small_model_hallucination_rate": small_hallucinations / total,
            "large_model_hallucination_rate": large_hallucinations / total
        }
    
    def save_report(self, output_path: str = "exercise2_report.json"):
        """Save results to JSON report."""
        report = {
            "exercise": "Exercise 2: Open Model + RAG vs. Large Model",
            "results": [
                {
                    "query": r.query,
                    "small_model_rag": r.small_model_rag,
                    "large_model_no_rag": r.large_model_no_rag,
                    "small_model_hallucinated": r.small_model_hallucinated,
                    "large_model_hallucinated": r.large_model_hallucinated
                }
                for r in self.results
            ],
            "analysis": self.analyze_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {output_path}")
        return report
