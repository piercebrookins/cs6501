"""
Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model

Compare local RAG pipeline against frontier models (GPT-4, Claude).
"""

from typing import List, Dict, Any
import json
from dataclasses import dataclass


@dataclass
class FrontierComparisonResult:
    """Result comparing local RAG vs frontier model."""
    query: str
    local_rag_response: str
    frontier_response: str
    local_grounded: bool
    frontier_used_search: bool


class Exercise3_OpenVsFrontier:
    """
    Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model
    
    Compare Qwen 2.5 1.5B with RAG vs GPT-4 or Claude via web interface.
    
    Document:
    - Where does the frontier model's general knowledge succeed?
    - When did the frontier model appear to be using live web search?
    - Where does your RAG system provide more accurate, specific answers?
    - What does this tell you about when RAG adds value vs. when a 
      powerful model suffices?
    """
    
    def __init__(self, rag_pipeline):
        """
        Initialize Exercise 3.
        
        Args:
            rag_pipeline: RAGPipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.results: List[FrontierComparisonResult] = []
    
    def document_manual_comparison(
        self,
        query: str,
        local_response: str,
        frontier_response: str,
        frontier_used_search: bool = False
    ) -> FrontierComparisonResult:
        """
        Document a manual comparison.
        
        Since frontier models are accessed via web interface,
        this method is used to record manual comparisons.
        
        Args:
            query: The query
            local_response: Response from local RAG
            frontier_response: Response from frontier model (manual entry)
            frontier_used_search: Whether frontier model appeared to use search
            
        Returns:
            FrontierComparisonResult
        """
        # Check if local response is grounded in retrieved context
        rag_result = self.rag_pipeline.query(query)
        local_grounded = self._check_grounded(local_response, rag_result['retrieved_chunks'])
        
        result = FrontierComparisonResult(
            query=query,
            local_rag_response=local_response,
            frontier_response=frontier_response,
            local_grounded=local_grounded,
            frontier_used_search=frontier_used_search
        )
        self.results.append(result)
        return result
    
    def _check_grounded(self, response: str, retrieved_chunks: List[Dict]) -> bool:
        """Check if response is grounded in retrieved chunks."""
        response_words = set(response.lower().split())
        
        for chunk in retrieved_chunks:
            chunk_words = set(chunk['text'].lower().split())
            overlap = len(response_words & chunk_words)
            if overlap > len(response_words) * 0.3:  # 30% overlap heuristic
                return True
        
        return False
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze frontier comparison results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        grounded_count = sum(1 for r in self.results if r.local_grounded)
        search_count = sum(1 for r in self.results if r.frontier_used_search)
        
        return {
            "total_comparisons": len(self.results),
            "local_grounded_count": grounded_count,
            "frontier_used_search_count": search_count,
            "observations": [
                "Local RAG provides grounded answers with source attribution",
                "Frontier models may use web search for recent information",
                "RAG excels for specific domain knowledge not in training data"
            ]
        }
    
    def save_report(self, output_path: str = "exercise3_report.json"):
        """Save results to JSON report."""
        report = {
            "exercise": "Exercise 3: Open Model + RAG vs. State-of-the-Art",
            "note": "Frontier model responses recorded manually via web interface",
            "results": [
                {
                    "query": r.query,
                    "local_rag_response": r.local_rag_response,
                    "frontier_response": r.frontier_response,
                    "local_grounded": r.local_grounded,
                    "frontier_used_search": r.frontier_used_search
                }
                for r in self.results
            ],
            "analysis": self.analyze_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {output_path}")
        return report
