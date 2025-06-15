"""
Diadochi: Intelligent Model Combination Framework

Named after Alexander the Great's successors who divided his empire into specialized
domains, this module intelligently combines domain-expert models to produce
superior expertise across multiple domains.

Based on the Combine Harvester framework, implementing five architectural patterns:
1. Router-Based Ensembles
2. Sequential Chaining  
3. Mixture of Experts
4. Specialized System Prompts
5. Knowledge Distillation Across Domains
"""

import abc
import asyncio
import json
import logging
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from enum import Enum
import hashlib
import pickle
from pathlib import Path

# Logging setup
logger = logging.getLogger(__name__)

class IntegrationPattern(Enum):
    """Enumeration of available integration patterns."""
    ROUTER_ENSEMBLE = "router_ensemble"
    SEQUENTIAL_CHAIN = "sequential_chain"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    SYSTEM_PROMPTS = "system_prompts"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"

@dataclass
class DomainExpertise:
    """Represents domain expertise configuration."""
    domain: str
    description: str
    keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    specialized_prompts: Dict[str, str] = field(default_factory=dict)
    weight: float = 1.0

@dataclass
class QueryContext:
    """Context for processing queries through the Diadochi system."""
    query: str
    domains: List[str] = field(default_factory=list)
    routing_scores: Dict[str, float] = field(default_factory=dict)
    expert_responses: Dict[str, str] = field(default_factory=dict)
    integration_method: Optional[IntegrationPattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class ModelInterface(ABC):
    """Abstract interface for all model implementations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the model."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @abstractmethod
    def get_confidence(self, query: str, domain: str) -> float:
        """Get confidence score for handling a query in a specific domain."""
        pass

class DomainRouter(ABC):
    """Abstract base class for all routers."""
    
    @abstractmethod
    async def route(self, query: str, available_domains: List[str]) -> Optional[str]:
        """Route a query to the most appropriate domain."""
        pass
    
    @abstractmethod
    async def route_multiple(self, query: str, available_domains: List[str], k: int) -> List[str]:
        """Route a query to the k most appropriate domains."""
        pass

class ResponseMixer(ABC):
    """Abstract base class for response mixing strategies."""
    
    @abstractmethod
    async def mix(self, query: str, responses: Dict[str, str], weights: Optional[Dict[str, float]] = None) -> str:
        """Mix multiple responses into a single coherent response."""
        pass

class EmbeddingRouter(DomainRouter):
    """Router based on embedding similarity."""
    
    def __init__(self, embedding_model: ModelInterface, threshold: float = 0.75):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.domain_embeddings: Dict[str, List[float]] = {}
        self.domain_descriptions: Dict[str, str] = {}
    
    def add_domain(self, domain: str, description: str):
        """Add a domain with its description."""
        self.domain_descriptions[domain] = description
    
    async def _compute_domain_embeddings(self):
        """Compute embeddings for all domain descriptions."""
        for domain, description in self.domain_descriptions.items():
            if domain not in self.domain_embeddings:
                self.domain_embeddings[domain] = await self.embedding_model.embed(description)
    
    async def route(self, query: str, available_domains: List[str]) -> Optional[str]:
        """Route query to the most similar domain."""
        await self._compute_domain_embeddings()
        
        query_embedding = await self.embedding_model.embed(query)
        best_domain = None
        best_score = 0.0
        
        for domain in available_domains:
            if domain in self.domain_embeddings:
                similarity = self._cosine_similarity(query_embedding, self.domain_embeddings[domain])
                if similarity > best_score and similarity >= self.threshold:
                    best_score = similarity
                    best_domain = domain
        
        return best_domain
    
    async def route_multiple(self, query: str, available_domains: List[str], k: int) -> List[str]:
        """Route query to top k most similar domains."""
        await self._compute_domain_embeddings()
        
        query_embedding = await self.embedding_model.embed(query)
        scores = []
        
        for domain in available_domains:
            if domain in self.domain_embeddings:
                similarity = self._cosine_similarity(query_embedding, self.domain_embeddings[domain])
                if similarity >= self.threshold:
                    scores.append((domain, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in scores[:k]]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        np_a, np_b = np.array(a), np.array(b)
        return np.dot(np_a, np_b) / (np.linalg.norm(np_a) * np.linalg.norm(np_b))

class KeywordRouter(DomainRouter):
    """Simple keyword-based router."""
    
    def __init__(self):
        self.domain_keywords: Dict[str, List[str]] = {}
        self.keyword_weights: Dict[str, Dict[str, float]] = {}
    
    def add_domain(self, domain: str, keywords: List[str], weights: Optional[Dict[str, float]] = None):
        """Add domain with associated keywords."""
        self.domain_keywords[domain] = keywords
        self.keyword_weights[domain] = weights or {kw: 1.0 for kw in keywords}
    
    async def route(self, query: str, available_domains: List[str]) -> Optional[str]:
        """Route based on keyword matching."""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain in available_domains:
            score = 0.0
            if domain in self.domain_keywords:
                for keyword in self.domain_keywords[domain]:
                    if keyword.lower() in query_lower:
                        weight = self.keyword_weights[domain].get(keyword, 1.0)
                        score += weight
                domain_scores[domain] = score
        
        if not domain_scores:
            return None
        
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else None
    
    async def route_multiple(self, query: str, available_domains: List[str], k: int) -> List[str]:
        """Route to top k domains by keyword score."""
        query_lower = query.lower()
        domain_scores = []
        
        for domain in available_domains:
            score = 0.0
            if domain in self.domain_keywords:
                for keyword in self.domain_keywords[domain]:
                    if keyword.lower() in query_lower:
                        weight = self.keyword_weights[domain].get(keyword, 1.0)
                        score += weight
            if score > 0:
                domain_scores.append((domain, score))
        
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in domain_scores[:k]]

class SynthesisMixer(ResponseMixer):
    """Mixer that uses an LLM to synthesize responses."""
    
    def __init__(self, synthesis_model: ModelInterface, prompt_template: Optional[str] = None):
        self.synthesis_model = synthesis_model
        self.prompt_template = prompt_template or self._default_template()
    
    def _default_template(self) -> str:
        return """
        You are tasked with synthesizing responses from multiple domain experts into a coherent, integrated response.
        
        Original query: {query}
        
        Expert responses:
        {weighted_responses}
        
        Create a unified response that integrates insights from all experts, giving appropriate weight to each domain based on their relevance. Ensure the response is coherent, non-repetitive, and directly addresses the original query.
        """
    
    async def mix(self, query: str, responses: Dict[str, str], weights: Optional[Dict[str, float]] = None) -> str:
        """Synthesize multiple expert responses."""
        if not responses:
            return "No expert responses available."
        
        if len(responses) == 1:
            return list(responses.values())[0]
        
        # Format weighted responses
        weighted_responses = []
        for domain, response in responses.items():
            weight = weights.get(domain, 1.0) if weights else 1.0
            weight_pct = int(weight * 100)
            weighted_responses.append(f"[{domain.title()} Expert ({weight_pct}%)]:\n{response}")
        
        formatted_responses = "\n\n".join(weighted_responses)
        
        prompt = self.prompt_template.format(
            query=query,
            weighted_responses=formatted_responses
        )
        
        return await self.synthesis_model.generate(prompt)

class ConcatenationMixer(ResponseMixer):
    """Simple mixer that concatenates responses with domain labels."""
    
    async def mix(self, query: str, responses: Dict[str, str], weights: Optional[Dict[str, float]] = None) -> str:
        """Concatenate responses with domain headers."""
        if not responses:
            return "No expert responses available."
        
        if len(responses) == 1:
            return list(responses.values())[0]
        
        mixed_response = f"Multi-domain analysis for: {query}\n\n"
        
        # Sort by weight if provided
        sorted_responses = responses.items()
        if weights:
            sorted_responses = sorted(responses.items(), 
                                    key=lambda x: weights.get(x[0], 0.0), 
                                    reverse=True)
        
        for domain, response in sorted_responses:
            weight_info = ""
            if weights and domain in weights:
                weight_pct = int(weights[domain] * 100)
                weight_info = f" ({weight_pct}%)"
            
            mixed_response += f"## {domain.title()} Expert{weight_info}\n\n{response}\n\n"
        
        return mixed_response.strip()

class ModelRegistry:
    """Registry for managing multiple model instances."""
    
    def __init__(self):
        self.models: Dict[str, ModelInterface] = {}
        self.domain_mappings: Dict[str, str] = {}
    
    def register_model(self, name: str, model: ModelInterface, domains: Optional[List[str]] = None):
        """Register a model with optional domain mappings."""
        self.models[name] = model
        if domains:
            for domain in domains:
                self.domain_mappings[domain] = name
    
    def get_model(self, identifier: str) -> Optional[ModelInterface]:
        """Get a model by name or domain."""
        if identifier in self.models:
            return self.models[identifier]
        elif identifier in self.domain_mappings:
            return self.models[self.domain_mappings[identifier]]
        return None
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self.domain_mappings.keys())

class RouterBasedEnsemble:
    """Router-based ensemble implementation."""
    
    def __init__(self, 
                 router: DomainRouter,
                 registry: ModelRegistry,
                 default_model: Optional[str] = None,
                 mixer: Optional[ResponseMixer] = None):
        self.router = router
        self.registry = registry
        self.default_model = default_model
        self.mixer = mixer
    
    async def generate(self, query: str, top_k: int = 1) -> str:
        """Generate response using router-based ensemble."""
        available_domains = self.registry.list_domains()
        
        if top_k == 1:
            # Single domain routing
            best_domain = await self.router.route(query, available_domains)
            if best_domain:
                model = self.registry.get_model(best_domain)
                if model:
                    return await model.generate(query)
            
            # Fallback to default model
            if self.default_model:
                default = self.registry.get_model(self.default_model)
                if default:
                    return await default.generate(query)
        
        else:
            # Multi-domain routing with mixing
            top_domains = await self.router.route_multiple(query, available_domains, top_k)
            if not top_domains:
                # Fallback to default model
                if self.default_model:
                    default = self.registry.get_model(self.default_model)
                    if default:
                        return await default.generate(query)
                return "No suitable domain experts available."
            
            # Get responses from multiple domains
            responses = {}
            for domain in top_domains:
                model = self.registry.get_model(domain)
                if model:
                    responses[domain] = await model.generate(query)
            
            # Mix responses if mixer is available
            if self.mixer and len(responses) > 1:
                return await self.mixer.mix(query, responses)
            elif len(responses) == 1:
                return list(responses.values())[0]
            else:
                # Simple concatenation fallback
                mixer = ConcatenationMixer()
                return await mixer.mix(query, responses)
        
        return "Unable to generate response."

class SequentialChain:
    """Sequential chaining implementation."""
    
    def __init__(self, 
                 models: List[str],
                 registry: ModelRegistry,
                 prompt_templates: Optional[Dict[str, str]] = None,
                 max_context_length: int = 4000):
        self.model_sequence = models
        self.registry = registry
        self.prompt_templates = prompt_templates or {}
        self.max_context_length = max_context_length
    
    async def generate(self, query: str) -> str:
        """Generate response through sequential chain."""
        context = {
            "query": query,
            "responses": [],
            "current_step": 0
        }
        
        for i, model_name in enumerate(self.model_sequence):
            model = self.registry.get_model(model_name)
            if not model:
                logger.warning(f"Model {model_name} not found in registry")
                continue
            
            # Build prompt using template if available
            prompt = self._build_prompt(model_name, context)
            
            # Generate response
            response = await model.generate(prompt)
            context["responses"].append(response)
            context["current_step"] = i + 1
            context[f"response_{i}"] = response
            context[model_name] = response
            
            # Manage context length
            if self._estimate_context_length(context) > self.max_context_length:
                context = await self._summarize_context(context)
        
        return context["responses"][-1] if context["responses"] else "No response generated."
    
    def _build_prompt(self, model_name: str, context: Dict[str, Any]) -> str:
        """Build prompt for the current model in the chain."""
        if model_name in self.prompt_templates:
            return self.prompt_templates[model_name].format(**context)
        
        # Default prompt structure
        if context["current_step"] == 0:
            return context["query"]
        else:
            prev_response = context["responses"][-1]
            return f"Previous analysis: {prev_response}\n\nOriginal query: {context['query']}\n\nBuild upon this analysis with your expertise:"
    
    def _estimate_context_length(self, context: Dict[str, Any]) -> int:
        """Estimate the total context length."""
        total_length = len(context["query"])
        for response in context["responses"]:
            total_length += len(response)
        return total_length
    
    async def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize context to manage length."""
        if len(context["responses"]) > 1:
            # Summarize the oldest response
            oldest_response = context["responses"][0]
            summary = f"[Summarized]: {oldest_response[:200]}..."
            context["responses"][0] = summary
        return context

class MixtureOfExperts:
    """Mixture of Experts implementation."""
    
    def __init__(self,
                 registry: ModelRegistry,
                 confidence_estimator: Optional[Callable] = None,
                 mixer: Optional[ResponseMixer] = None,
                 threshold: float = 0.1,
                 temperature: float = 0.5):
        self.registry = registry
        self.confidence_estimator = confidence_estimator
        self.mixer = mixer or SynthesisMixer(None)  # Will need to set synthesis model
        self.threshold = threshold
        self.temperature = temperature
    
    async def generate(self, query: str) -> str:
        """Generate response using mixture of experts."""
        available_domains = self.registry.list_domains()
        
        # Compute confidence scores for each domain
        confidence_scores = {}
        for domain in available_domains:
            model = self.registry.get_model(domain)
            if model:
                if self.confidence_estimator:
                    confidence_scores[domain] = self.confidence_estimator(query, domain)
                else:
                    confidence_scores[domain] = model.get_confidence(query, domain)
        
        # Filter by threshold and apply softmax weighting
        weights = self._compute_weights(confidence_scores)
        
        # Get responses from relevant experts
        responses = {}
        tasks = []
        relevant_domains = [d for d, w in weights.items() if w > 0]
        
        for domain in relevant_domains:
            model = self.registry.get_model(domain)
            if model:
                tasks.append(self._get_expert_response(model, query, domain))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                domain, response = result
                responses[domain] = response
        
        # Mix responses
        if len(responses) > 1 and self.mixer:
            return await self.mixer.mix(query, responses, weights)
        elif len(responses) == 1:
            return list(responses.values())[0]
        else:
            return "No suitable experts available for this query."
    
    async def _get_expert_response(self, model: ModelInterface, query: str, domain: str) -> Tuple[str, str]:
        """Get response from a single expert."""
        response = await model.generate(query)
        return domain, response
    
    def _compute_weights(self, confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute softmax weights from confidence scores."""
        # Filter by threshold
        filtered_scores = {d: s for d, s in confidence_scores.items() if s >= self.threshold}
        
        if not filtered_scores:
            return {}
        
        # Apply softmax
        exp_scores = {d: np.exp(s / self.temperature) for d, s in filtered_scores.items()}
        total = sum(exp_scores.values())
        
        return {d: exp_s / total for d, exp_s in exp_scores.items()}

class SystemPromptExpert:
    """Single model with specialized system prompts for multiple domains."""
    
    def __init__(self, 
                 base_model: ModelInterface,
                 domain_prompts: Dict[str, str],
                 integration_prompt: Optional[str] = None):
        self.base_model = base_model
        self.domain_prompts = domain_prompts
        self.integration_prompt = integration_prompt or self._default_integration_prompt()
    
    def _default_integration_prompt(self) -> str:
        return """
        You are an expert in multiple domains. For each query, determine which domains are relevant and provide an integrated response that combines insights from all relevant areas of expertise.
        
        Available domains:
        {domains}
        
        Query: {query}
        
        Provide a comprehensive response that integrates knowledge from all relevant domains.
        """
    
    async def generate(self, query: str, domains: Optional[List[str]] = None) -> str:
        """Generate response using specialized system prompts."""
        if domains and len(domains) == 1:
            # Single domain query
            domain = domains[0]
            if domain in self.domain_prompts:
                prompt = f"{self.domain_prompts[domain]}\n\nQuery: {query}"
                return await self.base_model.generate(prompt)
        
        # Multi-domain or general query
        domain_descriptions = "\n".join([
            f"- {domain}: {prompt.split('.')[0]}."
            for domain, prompt in self.domain_prompts.items()
        ])
        
        prompt = self.integration_prompt.format(
            domains=domain_descriptions,
            query=query
        )
        
        return await self.base_model.generate(prompt)

class DiadochiCore:
    """Main Diadochi framework orchestrator."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.domain_expertise: Dict[str, DomainExpertise] = {}
        self.active_pattern: Optional[IntegrationPattern] = None
        self.ensemble: Optional[RouterBasedEnsemble] = None
        self.chain: Optional[SequentialChain] = None
        self.mixture: Optional[MixtureOfExperts] = None
        self.system_expert: Optional[SystemPromptExpert] = None
        self._cache: Dict[str, Any] = {}
    
    def add_domain_expertise(self, expertise: DomainExpertise):
        """Add domain expertise configuration."""
        self.domain_expertise[expertise.domain] = expertise
        logger.info(f"Added domain expertise: {expertise.domain}")
    
    def register_model(self, name: str, model: ModelInterface, domains: Optional[List[str]] = None):
        """Register a model with the system."""
        self.registry.register_model(name, model, domains)
        logger.info(f"Registered model: {name} for domains: {domains}")
    
    def configure_router_ensemble(self, 
                                router_type: str = "embedding",
                                embedding_model: str = "default",
                                threshold: float = 0.75,
                                mixer_type: str = "synthesis"):
        """Configure router-based ensemble."""
        # Get embedding model
        embed_model = self.registry.get_model(embedding_model)
        if not embed_model:
            raise ValueError(f"Embedding model {embedding_model} not found")
        
        # Create router
        if router_type == "embedding":
            router = EmbeddingRouter(embed_model, threshold)
            for domain, expertise in self.domain_expertise.items():
                router.add_domain(domain, expertise.description)
        elif router_type == "keyword":
            router = KeywordRouter()
            for domain, expertise in self.domain_expertise.items():
                router.add_domain(domain, expertise.keywords)
        else:
            raise ValueError(f"Unknown router type: {router_type}")
        
        # Create mixer
        mixer = None
        if mixer_type == "synthesis":
            synthesis_model = self.registry.get_model(embedding_model)  # Using same model
            mixer = SynthesisMixer(synthesis_model)
        elif mixer_type == "concatenation":
            mixer = ConcatenationMixer()
        
        self.ensemble = RouterBasedEnsemble(router, self.registry, embedding_model, mixer)
        self.active_pattern = IntegrationPattern.ROUTER_ENSEMBLE
        logger.info(f"Configured router ensemble with {router_type} router and {mixer_type} mixer")
    
    def configure_sequential_chain(self, 
                                 model_sequence: List[str],
                                 prompt_templates: Optional[Dict[str, str]] = None):
        """Configure sequential chaining."""
        self.chain = SequentialChain(model_sequence, self.registry, prompt_templates)
        self.active_pattern = IntegrationPattern.SEQUENTIAL_CHAIN
        logger.info(f"Configured sequential chain with {len(model_sequence)} models")
    
    def configure_mixture_of_experts(self,
                                   mixer_model: str = "default",
                                   threshold: float = 0.1,
                                   temperature: float = 0.5):
        """Configure mixture of experts."""
        mixer_model_instance = self.registry.get_model(mixer_model)
        mixer = SynthesisMixer(mixer_model_instance) if mixer_model_instance else None
        
        self.mixture = MixtureOfExperts(
            self.registry, 
            mixer=mixer,
            threshold=threshold,
            temperature=temperature
        )
        self.active_pattern = IntegrationPattern.MIXTURE_OF_EXPERTS
        logger.info("Configured mixture of experts")
    
    def configure_system_prompts(self, 
                               base_model: str,
                               integration_prompt: Optional[str] = None):
        """Configure system prompt expert."""
        base_model_instance = self.registry.get_model(base_model)
        if not base_model_instance:
            raise ValueError(f"Base model {base_model} not found")
        
        domain_prompts = {}
        for domain, expertise in self.domain_expertise.items():
            if expertise.specialized_prompts:
                domain_prompts[domain] = expertise.specialized_prompts.get("system", expertise.description)
            else:
                domain_prompts[domain] = f"You are an expert in {domain}. {expertise.description}"
        
        self.system_expert = SystemPromptExpert(base_model_instance, domain_prompts, integration_prompt)
        self.active_pattern = IntegrationPattern.SYSTEM_PROMPTS
        logger.info(f"Configured system prompt expert with base model: {base_model}")
    
    async def generate(self, query: str, **kwargs) -> str:
        """Generate response using the configured pattern."""
        # Create query context
        context = QueryContext(query=query, integration_method=self.active_pattern)
        
        # Check cache
        cache_key = self._get_cache_key(query, kwargs)
        if cache_key in self._cache:
            logger.info("Returning cached response")
            return self._cache[cache_key]
        
        # Route to appropriate pattern
        response = ""
        try:
            if self.active_pattern == IntegrationPattern.ROUTER_ENSEMBLE and self.ensemble:
                response = await self.ensemble.generate(query, **kwargs)
            elif self.active_pattern == IntegrationPattern.SEQUENTIAL_CHAIN and self.chain:
                response = await self.chain.generate(query)
            elif self.active_pattern == IntegrationPattern.MIXTURE_OF_EXPERTS and self.mixture:
                response = await self.mixture.generate(query)
            elif self.active_pattern == IntegrationPattern.SYSTEM_PROMPTS and self.system_expert:
                domains = kwargs.get('domains')
                response = await self.system_expert.generate(query, domains)
            else:
                response = "No integration pattern configured or pattern not supported yet."
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = f"Error generating response: {str(e)}"
        
        # Cache response
        self._cache[cache_key] = response
        
        # Log metrics
        context.expert_responses = {"final": response}
        await self._log_metrics(context)
        
        return response
    
    def _get_cache_key(self, query: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for query and parameters."""
        key_data = {
            "query": query,
            "pattern": self.active_pattern.value if self.active_pattern else "none",
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _log_metrics(self, context: QueryContext):
        """Log performance metrics."""
        logger.info(f"Query processed in {time.time() - context.timestamp:.2f}s using {context.integration_method}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "active_pattern": self.active_pattern.value if self.active_pattern else None,
            "registered_models": self.registry.list_models(),
            "available_domains": list(self.domain_expertise.keys()),
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        logger.info("Cache cleared")

# Export main classes
__all__ = [
    "DiadochiCore",
    "DomainExpertise", 
    "QueryContext",
    "IntegrationPattern",
    "ModelInterface",
    "DomainRouter",
    "ResponseMixer",
    "EmbeddingRouter",
    "KeywordRouter", 
    "SynthesisMixer",
    "ConcatenationMixer",
    "ModelRegistry",
    "RouterBasedEnsemble",
    "SequentialChain",
    "MixtureOfExperts",
    "SystemPromptExpert"
]