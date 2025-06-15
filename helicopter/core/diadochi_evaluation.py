"""
Diadochi Evaluation Framework

Comprehensive evaluation system for multi-domain expert model combinations,
implementing metrics specific to cross-domain integration and domain expertise retention.
"""

import asyncio
import json
import logging
import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import statistics
from pathlib import Path

from .diadochi import DiadochiCore, QueryContext, IntegrationPattern, ModelInterface

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    domain_specific_accuracy: Dict[str, float] = field(default_factory=dict)
    cross_domain_accuracy: float = 0.0
    domain_expertise_retention: Dict[str, float] = field(default_factory=dict)
    integration_coherence: float = 0.0
    response_quality: float = 0.0
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    coverage_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvaluationQuery:
    """Represents a query for evaluation."""
    query: str
    domains: List[str]
    expected_response: Optional[str] = None
    reference_answers: Dict[str, str] = field(default_factory=dict)  # domain -> answer
    difficulty: str = "medium"  # easy, medium, hard
    query_type: str = "general"  # general, cross_domain, domain_specific
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Results from evaluating a single query."""
    query: EvaluationQuery
    generated_response: str
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    domain_scores: Dict[str, float] = field(default_factory=dict)
    latency: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    error: Optional[str] = None

class DomainSpecificEvaluator:
    """Evaluates domain-specific expertise."""
    
    def __init__(self, domain_experts: Dict[str, ModelInterface]):
        """Initialize with domain expert models for comparison."""
        self.domain_experts = domain_experts
    
    async def evaluate_domain_accuracy(self, 
                                     query: EvaluationQuery,
                                     generated_response: str) -> Dict[str, float]:
        """Evaluate accuracy in each relevant domain."""
        scores = {}
        
        for domain in query.domains:
            if domain in self.domain_experts:
                expert_model = self.domain_experts[domain]
                
                # Get expert's response for comparison
                expert_response = await expert_model.generate(query.query)
                
                # Compare responses using similarity and quality metrics
                domain_score = await self._compare_responses(
                    generated_response, expert_response, domain
                )
                scores[domain] = domain_score
        
        return scores
    
    async def _compare_responses(self, response1: str, response2: str, domain: str) -> float:
        """Compare two responses for similarity and quality."""
        # Simple similarity-based scoring
        # In production, this could use more sophisticated evaluation models
        
        similarity_score = self._semantic_similarity(response1, response2)
        factual_score = self._assess_factual_consistency(response1, response2, domain)
        completeness_score = self._assess_completeness(response1, response2)
        
        # Weighted combination
        overall_score = (
            0.4 * similarity_score +
            0.4 * factual_score +
            0.2 * completeness_score
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple word overlap-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _assess_factual_consistency(self, response1: str, response2: str, domain: str) -> float:
        """Assess factual consistency between responses."""
        # Placeholder implementation - in production would use fact-checking models
        # For now, return moderate score based on keyword overlap
        
        domain_keywords = {
            "computer_vision": ["pixel", "image", "vision", "detection", "classification"],
            "machine_learning": ["model", "training", "algorithm", "neural", "learning"],
            "mathematics": ["equation", "formula", "proof", "theorem", "calculation"],
            "programming": ["code", "function", "variable", "programming", "software"]
        }
        
        if domain not in domain_keywords:
            return 0.7  # Default moderate score
        
        keywords = domain_keywords[domain]
        response1_lower = response1.lower()
        response2_lower = response2.lower()
        
        response1_matches = sum(1 for kw in keywords if kw in response1_lower)
        response2_matches = sum(1 for kw in keywords if kw in response2_lower)
        
        if response1_matches == 0 and response2_matches == 0:
            return 0.5
        
        # Compare keyword density
        max_matches = max(response1_matches, response2_matches)
        min_matches = min(response1_matches, response2_matches)
        
        return min_matches / max_matches if max_matches > 0 else 0.5
    
    def _assess_completeness(self, response1: str, response2: str) -> float:
        """Assess completeness of response compared to reference."""
        len1, len2 = len(response1.split()), len(response2.split())
        
        if len2 == 0:
            return 0.0
        
        # Completeness based on length ratio (with bounds)
        ratio = len1 / len2
        
        # Optimal range is 0.8 to 1.2 of reference length
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif ratio < 0.8:
            return ratio / 0.8  # Penalize for being too short
        else:
            return max(0.5, 1.2 / ratio)  # Penalize for being too long

class CrossDomainEvaluator:
    """Evaluates cross-domain integration capabilities."""
    
    def __init__(self, integration_model: Optional[ModelInterface] = None):
        self.integration_model = integration_model
    
    async def evaluate_integration_coherence(self, 
                                           query: EvaluationQuery,
                                           generated_response: str) -> float:
        """Evaluate logical coherence of cross-domain integration."""
        if len(query.domains) < 2:
            return 1.0  # Not a cross-domain query
        
        # Check for domain-specific information in response
        domain_coverage = self._assess_domain_coverage(generated_response, query.domains)
        logical_flow = self._assess_logical_flow(generated_response)
        contradiction_penalty = self._detect_contradictions(generated_response)
        
        coherence_score = (
            0.4 * domain_coverage +
            0.4 * logical_flow +
            0.2 * (1.0 - contradiction_penalty)  # Subtract contradiction penalty
        )
        
        return min(max(coherence_score, 0.0), 1.0)
    
    def _assess_domain_coverage(self, response: str, domains: List[str]) -> float:
        """Assess how well the response covers all relevant domains."""
        response_lower = response.lower()
        
        domain_indicators = {
            "computer_vision": ["image", "visual", "pixel", "detection", "vision"],
            "machine_learning": ["model", "algorithm", "training", "neural", "learning"],
            "mathematics": ["equation", "formula", "calculation", "mathematical"],
            "programming": ["code", "programming", "function", "software"],
            "data_science": ["data", "analysis", "statistics", "correlation"]
        }
        
        covered_domains = 0
        for domain in domains:
            if domain in domain_indicators:
                indicators = domain_indicators[domain]
                if any(indicator in response_lower for indicator in indicators):
                    covered_domains += 1
        
        return covered_domains / len(domains) if domains else 0.0
    
    def _assess_logical_flow(self, response: str) -> float:
        """Assess logical flow and organization of the response."""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.8  # Short responses are assumed coherent
        
        # Check for transition words and logical connectors
        connectors = ["however", "therefore", "furthermore", "additionally", "consequently", 
                     "moreover", "in contrast", "similarly", "meanwhile", "thus"]
        
        connector_count = sum(1 for sentence in sentences 
                            for connector in connectors 
                            if connector in sentence.lower())
        
        # Normalize by sentence count
        connector_density = connector_count / max(len(sentences) - 1, 1)
        
        # Good flow has some connectors but not too many
        if 0.1 <= connector_density <= 0.4:
            return 1.0
        elif connector_density < 0.1:
            return 0.7  # Lacks connective flow
        else:
            return 0.8  # Too many connectors
    
    def _detect_contradictions(self, response: str) -> float:
        """Detect potential contradictions in the response."""
        # Simple contradiction detection based on opposing terms
        opposing_pairs = [
            ("increase", "decrease"), ("high", "low"), ("positive", "negative"),
            ("better", "worse"), ("effective", "ineffective"), ("correct", "incorrect")
        ]
        
        response_lower = response.lower()
        contradiction_score = 0.0
        
        for word1, word2 in opposing_pairs:
            if word1 in response_lower and word2 in response_lower:
                # Check if they're in close proximity (potential contradiction)
                positions1 = [i for i, word in enumerate(response_lower.split()) if word1 in word]
                positions2 = [i for i, word in enumerate(response_lower.split()) if word2 in word]
                
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) < 10:  # Within 10 words
                            contradiction_score += 0.1
        
        return min(contradiction_score, 1.0)

class PerformanceEvaluator:
    """Evaluates performance metrics like latency and cost."""
    
    def __init__(self):
        self.baseline_metrics = {}
    
    def evaluate_latency(self, start_time: float, end_time: float) -> Dict[str, float]:
        """Evaluate response latency."""
        latency = end_time - start_time
        
        return {
            "total_latency": latency,
            "latency_category": self._categorize_latency(latency)
        }
    
    def _categorize_latency(self, latency: float) -> float:
        """Categorize latency performance (0-1 scale)."""
        if latency < 1.0:
            return 1.0  # Excellent
        elif latency < 3.0:
            return 0.8  # Good
        elif latency < 10.0:
            return 0.6  # Acceptable
        elif latency < 30.0:
            return 0.4  # Slow
        else:
            return 0.2  # Very slow
    
    def evaluate_efficiency(self, 
                          response_quality: float,
                          latency: float,
                          cost: float = 0.0) -> float:
        """Evaluate overall efficiency combining quality, speed, and cost."""
        latency_score = self._categorize_latency(latency)
        cost_score = self._categorize_cost(cost)
        
        # Weighted efficiency score
        efficiency = (
            0.5 * response_quality +
            0.3 * latency_score +
            0.2 * cost_score
        )
        
        return min(max(efficiency, 0.0), 1.0)
    
    def _categorize_cost(self, cost: float) -> float:
        """Categorize cost performance (0-1 scale)."""
        if cost == 0.0:
            return 1.0  # Free
        elif cost < 0.01:
            return 0.9  # Very cheap
        elif cost < 0.1:
            return 0.7  # Cheap
        elif cost < 1.0:
            return 0.5  # Moderate
        else:
            return 0.3  # Expensive

class DiadochiEvaluator:
    """Main evaluation framework for Diadochi systems."""
    
    def __init__(self, 
                 domain_experts: Optional[Dict[str, ModelInterface]] = None,
                 integration_model: Optional[ModelInterface] = None):
        self.domain_evaluator = DomainSpecificEvaluator(domain_experts or {})
        self.cross_domain_evaluator = CrossDomainEvaluator(integration_model)
        self.performance_evaluator = PerformanceEvaluator()
        
        self.evaluation_history: List[EvaluationResult] = []
    
    async def evaluate_single_query(self, 
                                  diadochi_system: DiadochiCore,
                                  query: EvaluationQuery) -> EvaluationResult:
        """Evaluate a single query against the Diadochi system."""
        start_time = time.time()
        
        try:
            # Generate response
            generated_response = await diadochi_system.generate(query.query)
            end_time = time.time()
            
            # Initialize result
            result = EvaluationResult(
                query=query,
                generated_response=generated_response,
                latency=end_time - start_time
            )
            
            # Domain-specific evaluation
            if query.domains:
                domain_scores = await self.domain_evaluator.evaluate_domain_accuracy(
                    query, generated_response
                )
                result.domain_scores = domain_scores
                result.evaluation_scores["domain_accuracy"] = statistics.mean(domain_scores.values()) if domain_scores else 0.0
            
            # Cross-domain evaluation
            integration_score = await self.cross_domain_evaluator.evaluate_integration_coherence(
                query, generated_response
            )
            result.evaluation_scores["integration_coherence"] = integration_score
            
            # Performance evaluation
            latency_metrics = self.performance_evaluator.evaluate_latency(start_time, end_time)
            result.evaluation_scores.update(latency_metrics)
            
            # Overall quality score
            quality_components = [
                result.evaluation_scores.get("domain_accuracy", 0.7),
                result.evaluation_scores.get("integration_coherence", 0.7)
            ]
            result.evaluation_scores["response_quality"] = statistics.mean(quality_components)
            
            # Efficiency score
            efficiency = self.performance_evaluator.evaluate_efficiency(
                result.evaluation_scores["response_quality"],
                result.latency
            )
            result.evaluation_scores["efficiency"] = efficiency
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            result = EvaluationResult(
                query=query,
                generated_response="",
                error=str(e),
                latency=time.time() - start_time
            )
        
        self.evaluation_history.append(result)
        return result
    
    async def evaluate_dataset(self, 
                             diadochi_system: DiadochiCore,
                             queries: List[EvaluationQuery],
                             parallel: bool = True) -> EvaluationMetrics:
        """Evaluate a dataset of queries."""
        logger.info(f"Starting evaluation of {len(queries)} queries")
        
        if parallel:
            # Evaluate queries in parallel
            tasks = [self.evaluate_single_query(diadochi_system, query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, EvaluationResult) and not r.error]
        else:
            # Evaluate queries sequentially
            valid_results = []
            for query in queries:
                result = await self.evaluate_single_query(diadochi_system, query)
                if not result.error:
                    valid_results.append(result)
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(valid_results)
        
        logger.info(f"Evaluation completed. {len(valid_results)}/{len(queries)} queries successful")
        return metrics
    
    def _compute_aggregate_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Compute aggregate metrics from evaluation results."""
        if not results:
            return EvaluationMetrics()
        
        metrics = EvaluationMetrics()
        
        # Domain-specific accuracy
        domain_scores = defaultdict(list)
        for result in results:
            for domain, score in result.domain_scores.items():
                domain_scores[domain].append(score)
        
        metrics.domain_specific_accuracy = {
            domain: statistics.mean(scores) 
            for domain, scores in domain_scores.items()
        }
        
        # Cross-domain accuracy
        integration_scores = [
            result.evaluation_scores.get("integration_coherence", 0.0)
            for result in results
        ]
        metrics.cross_domain_accuracy = statistics.mean(integration_scores)
        
        # Overall metrics
        quality_scores = [
            result.evaluation_scores.get("response_quality", 0.0)
            for result in results
        ]
        metrics.response_quality = statistics.mean(quality_scores)
        
        # Latency metrics
        latencies = [result.latency for result in results]
        metrics.latency_metrics = {
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "max_latency": max(latencies)
        }
        
        # Coverage metrics (how many domains are covered)
        all_domains = set()
        covered_domains = set()
        for result in results:
            all_domains.update(result.query.domains)
            if result.domain_scores:
                covered_domains.update(result.domain_scores.keys())
        
        metrics.coverage_metrics = {
            "domain_coverage": len(covered_domains) / len(all_domains) if all_domains else 0.0,
            "total_domains": len(all_domains),
            "covered_domains": len(covered_domains)
        }
        
        return metrics
    
    def generate_report(self, metrics: EvaluationMetrics, output_path: Optional[Path] = None) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("# Diadochi Evaluation Report\n")
        
        # Summary section
        report.append("## Summary")
        report.append(f"- Overall Response Quality: {metrics.response_quality:.3f}")
        report.append(f"- Cross-Domain Integration: {metrics.cross_domain_accuracy:.3f}")
        report.append(f"- Domain Coverage: {metrics.coverage_metrics.get('domain_coverage', 0):.3f}")
        report.append("")
        
        # Domain-specific performance
        report.append("## Domain-Specific Performance")
        for domain, accuracy in metrics.domain_specific_accuracy.items():
            report.append(f"- {domain.title()}: {accuracy:.3f}")
        report.append("")
        
        # Performance metrics
        report.append("## Performance Metrics")
        latency = metrics.latency_metrics
        report.append(f"- Mean Latency: {latency.get('mean_latency', 0):.2f}s")
        report.append(f"- Median Latency: {latency.get('median_latency', 0):.2f}s")
        report.append(f"- 95th Percentile: {latency.get('p95_latency', 0):.2f}s")
        report.append("")
        
        # Coverage analysis
        report.append("## Coverage Analysis")
        coverage = metrics.coverage_metrics
        report.append(f"- Total Domains: {coverage.get('total_domains', 0)}")
        report.append(f"- Covered Domains: {coverage.get('covered_domains', 0)}")
        report.append(f"- Coverage Rate: {coverage.get('domain_coverage', 0):.1%}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            output_path.write_text(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def compare_patterns(self, 
                        pattern_results: Dict[IntegrationPattern, EvaluationMetrics]) -> str:
        """Compare performance across different integration patterns."""
        if not pattern_results:
            return "No pattern results to compare."
        
        report = []
        report.append("# Pattern Comparison Report\n")
        
        # Create comparison table
        patterns = list(pattern_results.keys())
        metrics_names = ["response_quality", "cross_domain_accuracy"]
        
        # Header
        report.append("| Pattern | Response Quality | Cross-Domain | Mean Latency |")
        report.append("|---------|------------------|--------------|--------------|")
        
        # Data rows
        for pattern in patterns:
            metrics = pattern_results[pattern]
            quality = f"{metrics.response_quality:.3f}"
            cross_domain = f"{metrics.cross_domain_accuracy:.3f}"
            latency = f"{metrics.latency_metrics.get('mean_latency', 0):.2f}s"
            
            report.append(f"| {pattern.value.replace('_', ' ').title()} | {quality} | {cross_domain} | {latency} |")
        
        report.append("")
        
        # Best performing pattern
        best_quality = max(pattern_results.items(), key=lambda x: x[1].response_quality)
        best_speed = min(pattern_results.items(), key=lambda x: x[1].latency_metrics.get('mean_latency', float('inf')))
        
        report.append("## Best Performers")
        report.append(f"- **Best Quality**: {best_quality[0].value.replace('_', ' ').title()} ({best_quality[1].response_quality:.3f})")
        report.append(f"- **Fastest**: {best_speed[0].value.replace('_', ' ').title()} ({best_speed[1].latency_metrics.get('mean_latency', 0):.2f}s)")
        
        return "\n".join(report)

# Utility functions for creating evaluation datasets
def create_domain_specific_queries(domain: str, count: int = 10) -> List[EvaluationQuery]:
    """Create domain-specific evaluation queries."""
    query_templates = {
        "computer_vision": [
            "How does {technique} work in image processing?",
            "What are the advantages of {method} for object detection?",
            "Explain the difference between {concept1} and {concept2} in computer vision.",
        ],
        "machine_learning": [
            "How does {algorithm} handle overfitting?",
            "What are the key differences between {model1} and {model2}?",
            "When would you use {technique} in machine learning?",
        ],
        "mathematics": [
            "Prove that {theorem} holds for {condition}.",
            "How do you calculate {metric} in {context}?",
            "What is the relationship between {concept1} and {concept2}?",
        ]
    }
    
    # Placeholder implementation - would be expanded with real templates and terms
    queries = []
    templates = query_templates.get(domain, ["What is {topic} in {domain}?"])
    
    for i in range(count):
        template = templates[i % len(templates)]
        query_text = template.format(
            technique=f"technique_{i}",
            method=f"method_{i}",
            concept1=f"concept1_{i}",
            concept2=f"concept2_{i}",
            topic=f"topic_{i}",
            domain=domain
        )
        
        queries.append(EvaluationQuery(
            query=query_text,
            domains=[domain],
            query_type="domain_specific",
            difficulty="medium"
        ))
    
    return queries

def create_cross_domain_queries(domains: List[str], count: int = 10) -> List[EvaluationQuery]:
    """Create cross-domain evaluation queries."""
    cross_templates = [
        "How do {domain1} and {domain2} techniques combine to solve {problem}?",
        "What are the {domain1} implications of {domain2} advances in {application}?",
        "Compare {domain1} and {domain2} approaches to {challenge}.",
    ]
    
    queries = []
    for i in range(count):
        template = cross_templates[i % len(cross_templates)]
        domain1 = domains[i % len(domains)]
        domain2 = domains[(i + 1) % len(domains)]
        
        query_text = template.format(
            domain1=domain1,
            domain2=domain2,
            problem=f"problem_{i}",
            application=f"application_{i}",
            challenge=f"challenge_{i}"
        )
        
        queries.append(EvaluationQuery(
            query=query_text,
            domains=[domain1, domain2],
            query_type="cross_domain",
            difficulty="hard"
        ))
    
    return queries

# Export main classes
__all__ = [
    "DiadochiEvaluator",
    "EvaluationMetrics",
    "EvaluationQuery", 
    "EvaluationResult",
    "DomainSpecificEvaluator",
    "CrossDomainEvaluator",
    "PerformanceEvaluator",
    "create_domain_specific_queries",
    "create_cross_domain_queries"
]