"""
Exercise 1: Open Model RAG vs. No RAG Comparison

Compare LLM's answers with and without retrieval augmentation.
Uses Qwen 2.5 1.5B with the Model T Ford repair manual 
and Congressional Record corpus.
"""

from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueryResult:
    """Result from a single query."""
    query: str
    without_rag: str
    with_rag: str
    retrieved_chunks: List[Dict]
    notes: str = ""


class Exercise1_RAGvsNoRAG:
    """
    Exercise 1: Compare RAG vs No-RAG responses.
    
    Setup: Use Qwen 2.5 1.5B (or another small open model) with:
    - Model T Ford repair manual
    - Congressional Record corpus (separately)
    """
    
    # Queries for Model T Ford corpus
    MODELT_QUERIES = [
        "How do I adjust the carburetor on a Model T?",
        "What is the correct spark plug gap for a Model T Ford?",
        "How do I fix a slipping transmission band?",
        "What oil should I use in a Model T engine?"
    ]
    
    # Queries for Congressional Record corpus
    CONGRESS_QUERIES = [
        "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
        "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
        "What is the purpose of the Main Street Parity Act?",
        "Who in Congress has spoken for and against funding of pregnancy centers?"
    ]
    
    def __init__(self, pipeline, llm_generator=None):
        """
        Initialize Exercise 1.
        
        Args:
            pipeline: Initialized RAGPipeline
            llm_generator: Function to generate LLM responses (query -> response)
        """
        self.pipeline = pipeline
        self.llm_generator = llm_generator
        self.results: List[QueryResult] = []
    
    def run_modelt_queries(self) -> List[QueryResult]:
        """Run all Model T queries and compare RAG vs No-RAG."""
        return self._run_queries(self.MODELT_QUERIES, "Model T Ford")
    
    def run_congress_queries(self) -> List[QueryResult]:
        """Run all Congressional Record queries."""
        return self._run_queries(self.CONGRESS_QUERIES, "Congressional Record")
    
    def _run_queries(
        self,
        queries: List[str],
        corpus_name: str
    ) -> List[QueryResult]:
        """
        Run a set of queries comparing RAG vs No-RAG.
        
        Args:
            queries: List of query strings
            corpus_name: Name of corpus for reporting
            
        Returns:
            List of QueryResult objects
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running {corpus_name} Queries")
        print(f"{'='*60}")
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")
            print(f"Q: {query}")
            
            # Without RAG (direct LLM query)
            without_rag = self._query_without_rag(query)
            print(f"\n[Without RAG]")
            print(without_rag[:200] + "..." if len(without_rag) > 200 else without_rag)
            
            # With RAG
            rag_result = self.pipeline.query(query, return_context=True)
            with_rag = self._query_with_rag(rag_result['prompt'])
            
            print(f"\n[With RAG]")
            print(with_rag[:200] + "..." if len(with_rag) > 200 else with_rag)
            
            print(f"\n[Retrieved {len(rag_result['retrieved_chunks'])} chunks]")
            for chunk in rag_result['retrieved_chunks']:
                print(f"  - {chunk['metadata']['source']} p.{chunk['metadata']['page']} (score: {chunk.get('score', 'N/A'):.3f})")
            
            result = QueryResult(
                query=query,
                without_rag=without_rag,
                with_rag=with_rag,
                retrieved_chunks=rag_result['retrieved_chunks']
            )
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _query_without_rag(self, query: str) -> str:
        """Query LLM without RAG context."""
        if self.llm_generator:
            return self.llm_generator(query)
        return "[LLM generation not configured - implement llm_generator]"
    
    def _query_with_rag(self, prompt: str) -> str:
        """Query LLM with RAG prompt."""
        if self.llm_generator:
            return self.llm_generator(prompt)
        return "[LLM generation not configured - implement llm_generator]"
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze comparison results.
        
        Returns:
            Analysis dict with observations about hallucinations,
            grounding, and general knowledge accuracy.
        """
        if not self.results:
            return {"error": "No results to analyze. Run queries first."}
        
        analysis = {
            "total_queries": len(self.results),
            "observations": [],
            "potential_hallucinations_without_rag": [],
            "well_grounded_with_rag": [],
            "general_knowledge_correct": []
        }
        
        for result in self.results:
            # Check for specific values (potential hallucination indicators)
            without_rag_lower = result.without_rag.lower()
            
            # Look for specific numbers/measurements
            has_specific_values = bool(
                re.search(r'\d+\.?\d*\s*(inch|mm|gap|degree)', without_rag_lower) or
                re.search(r'\d+\.?\d*\s*(oz|quart|pint|gallon)', without_rag_lower)
            )
            
            obs = {
                "query": result.query,
                "has_specific_values_no_rag": has_specific_values,
                "retrieved_sources": [c['metadata']['source'] for c in result.retrieved_chunks]
            }
            analysis["observations"].append(obs)
        
        return analysis
    
    def save_report(self, output_path: str = "exercise1_report.json"):
        """Save results to JSON report."""
        report = {
            "exercise": "Exercise 1: RAG vs No-RAG Comparison",
            "pipeline_stats": self.pipeline.get_stats(),
            "results": [
                {
                    "query": r.query,
                    "without_rag": r.without_rag,
                    "with_rag": r.with_rag,
                    "retrieved_chunks": [
                        {
                            "text": c['text'][:200] + "..." if len(c['text']) > 200 else c['text'],
                            "metadata": c['metadata'],
                            "score": c.get('score')
                        }
                        for c in r.retrieved_chunks
                    ]
                }
                for r in self.results
            ],
            "analysis": self.analyze_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {output_path}")
        return report


# Import re at module level for the analyze_results method
import re
