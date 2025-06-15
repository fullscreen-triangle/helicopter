"""
Diadochi Model Implementations

Concrete implementations of the ModelInterface for different LLM providers,
supporting the Diadochi framework's multi-domain expert system.
"""

import asyncio
import json
import logging
import os
import re
import requests
import time
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .diadochi import ModelInterface

logger = logging.getLogger(__name__)

class OllamaModel(ModelInterface):
    """Model interface for Ollama local models."""
    
    def __init__(self, 
                 model_name: str,
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120,
                 **kwargs):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.kwargs = kwargs
        self._session = requests.Session()
    
    @property
    def name(self) -> str:
        return f"ollama/{self.model_name}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama."""
        try:
            merged_kwargs = {**self.kwargs, **kwargs}
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                **merged_kwargs
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._session.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=self.timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                logger.error(f"Ollama embedding error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting embeddings from Ollama: {e}")
            return []
    
    def get_confidence(self, query: str, domain: str) -> float:
        """Estimate confidence based on keyword matching and model capabilities."""
        # Simple heuristic - can be improved with domain-specific fine-tuning
        query_lower = query.lower()
        
        # Domain-specific keyword patterns
        domain_patterns = {
            "computer_vision": ["image", "vision", "visual", "pixel", "detection", "segmentation"],
            "natural_language": ["text", "language", "nlp", "words", "sentence", "grammar"],
            "machine_learning": ["model", "training", "algorithm", "neural", "learning", "prediction"],
            "data_science": ["data", "analysis", "statistics", "correlation", "regression"],
            "mathematics": ["equation", "formula", "theorem", "proof", "calculation"],
            "programming": ["code", "function", "variable", "programming", "debug", "software"]
        }
        
        if domain in domain_patterns:
            matches = sum(1 for pattern in domain_patterns[domain] if pattern in query_lower)
            confidence = min(matches / len(domain_patterns[domain]) * 2, 1.0)  # Scale to 0-1
            return max(confidence, 0.1)  # Minimum baseline confidence
        
        return 0.5  # Default moderate confidence

class OpenAIModel(ModelInterface):
    """Model interface for OpenAI models."""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 **kwargs):
        if not HAS_OPENAI:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.model_name = model_name
        self.kwargs = kwargs
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    @property
    def name(self) -> str:
        return f"openai/{self.model_name}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI."""
        try:
            merged_kwargs = {
                "max_tokens": 1000,
                "temperature": 0.7,
                **self.kwargs,
                **kwargs
            }
            
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **merged_kwargs
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        try:
            # Use appropriate embedding model
            embedding_model = "text-embedding-3-small"
            if "ada" in self.model_name:
                embedding_model = "text-embedding-ada-002"
            
            response = await self.client.embeddings.create(
                model=embedding_model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI: {e}")
            return []
    
    def get_confidence(self, query: str, domain: str) -> float:
        """Estimate confidence for OpenAI models."""
        # OpenAI models are generally strong across domains
        base_confidence = 0.8
        
        # Adjust based on domain complexity
        domain_multipliers = {
            "general": 1.0,
            "computer_vision": 0.9,  # Good but not specialized
            "natural_language": 1.1,  # Excellent for NLP
            "mathematics": 0.95,
            "programming": 1.0,
            "creative_writing": 1.1
        }
        
        multiplier = domain_multipliers.get(domain, 0.9)
        return min(base_confidence * multiplier, 1.0)

class AnthropicModel(ModelInterface):
    """Model interface for Anthropic Claude models."""
    
    def __init__(self, 
                 model_name: str = "claude-3-sonnet-20240229",
                 api_key: Optional[str] = None,
                 **kwargs):
        if not HAS_ANTHROPIC:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        
        self.model_name = model_name
        self.kwargs = kwargs
        
        # Initialize Anthropic client
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    @property
    def name(self) -> str:
        return f"anthropic/{self.model_name}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic Claude."""
        try:
            merged_kwargs = {
                "max_tokens": 1000,
                "temperature": 0.7,
                **self.kwargs,
                **kwargs
            }
            
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **merged_kwargs
            )
            
            return response.content[0].text if response.content else ""
            
        except Exception as e:
            logger.error(f"Error calling Anthropic: {e}")
            return f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings - Anthropic doesn't provide embeddings API."""
        logger.warning("Anthropic doesn't provide embeddings API, using fallback")
        # Return a simple hash-based embedding as fallback
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert hash to pseudo-embedding (not semantically meaningful)
        hash_bytes = hash_obj.digest()
        return [float(b) / 255.0 for b in hash_bytes]  # Normalize to 0-1
    
    def get_confidence(self, query: str, domain: str) -> float:
        """Estimate confidence for Anthropic models."""
        # Claude models are excellent for reasoning and analysis
        base_confidence = 0.85
        
        domain_multipliers = {
            "general": 1.0,
            "reasoning": 1.1,
            "analysis": 1.1,
            "writing": 1.1,  
            "mathematics": 1.0,
            "programming": 0.95,
            "creative": 1.05
        }
        
        multiplier = domain_multipliers.get(domain, 0.95)
        return min(base_confidence * multiplier, 1.0)

class HuggingFaceModel(ModelInterface):
    """Model interface for HuggingFace models."""
    
    def __init__(self, 
                 model_name: str,
                 device: str = "auto",
                 **kwargs):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library not installed. Run: pip install transformers torch")
        
        self.model_name = model_name
        self.kwargs = kwargs
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {model_name}: {e}")
            raise
    
    @property
    def name(self) -> str:
        return f"huggingface/{self.model_name}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using HuggingFace model."""
        try:
            merged_kwargs = {
                "max_length": 512,
                "temperature": 0.7,
                "do_sample": True,
                **self.kwargs,
                **kwargs
            }
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        inputs.input_ids,
                        **merged_kwargs
                    )
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating with HuggingFace model: {e}")
            return f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFace model."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model(**inputs)
                )
            
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error getting embeddings from HuggingFace model: {e}")
            return []
    
    def get_confidence(self, query: str, domain: str) -> float:
        """Estimate confidence based on model type and domain."""
        # Confidence varies greatly by specific model
        base_confidence = 0.7
        
        # Adjust based on model type
        if "bert" in self.model_name.lower():
            base_confidence = 0.8  # BERT models are generally robust
        elif "gpt" in self.model_name.lower():
            base_confidence = 0.85  # GPT models are strong for generation
        elif "t5" in self.model_name.lower():
            base_confidence = 0.8  # T5 models are good for various tasks
        
        return base_confidence

class MockModel(ModelInterface):
    """Mock model for testing and development."""
    
    def __init__(self, name: str = "mock", responses: Optional[Dict[str, str]] = None):
        self._name = name
        self.responses = responses or {}
        self.call_count = 0
    
    @property
    def name(self) -> str:
        return f"mock/{self._name}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response."""
        self.call_count += 1
        
        # Check for specific responses
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default mock response
        return f"Mock response from {self._name} for: {prompt[:50]}..."
    
    async def embed(self, text: str) -> List[float]:
        """Generate mock embedding."""
        # Create a simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        
        # Convert to normalized embedding vector
        embedding = []
        for i in range(384):  # Standard embedding size
            bit = (hash_int >> (i % 32)) & 1
            embedding.append(float(bit))
        
        return embedding
    
    def get_confidence(self, query: str, domain: str) -> float:
        """Return mock confidence."""
        return 0.8  # High confidence for testing

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(engine: str, model_name: str, **kwargs) -> ModelInterface:
        """Create a model instance based on engine type."""
        engine = engine.lower()
        
        if engine == "ollama":
            return OllamaModel(model_name, **kwargs)
        elif engine == "openai":
            return OpenAIModel(model_name, **kwargs)
        elif engine == "anthropic":
            return AnthropicModel(model_name, **kwargs)
        elif engine == "huggingface":
            return HuggingFaceModel(model_name, **kwargs)
        elif engine == "mock":
            return MockModel(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    @staticmethod
    def list_supported_engines() -> List[str]:
        """List all supported engine types."""
        engines = ["mock", "ollama"]
        
        if HAS_OPENAI:
            engines.append("openai")
        if HAS_ANTHROPIC:
            engines.append("anthropic")
        if HAS_TRANSFORMERS:
            engines.append("huggingface")
        
        return engines

# Confidence estimation utilities
class ConfidenceEstimator:
    """Utilities for estimating model confidence in domains."""
    
    @staticmethod
    def keyword_based_confidence(query: str, domain_keywords: Dict[str, List[str]]) -> Dict[str, float]:
        """Estimate confidence based on keyword matching."""
        query_lower = query.lower()
        confidences = {}
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            confidence = min(matches / len(keywords) * 1.5, 1.0)  # Scale and cap at 1.0
            confidences[domain] = max(confidence, 0.1)  # Minimum baseline
        
        return confidences
    
    @staticmethod
    async def embedding_based_confidence(query: str, 
                                       domain_descriptions: Dict[str, str],
                                       embedding_model: ModelInterface) -> Dict[str, float]:
        """Estimate confidence based on embedding similarity."""
        query_embedding = await embedding_model.embed(query)
        if not query_embedding:
            return {}
        
        confidences = {}
        for domain, description in domain_descriptions.items():
            domain_embedding = await embedding_model.embed(description)
            if domain_embedding:
                similarity = ConfidenceEstimator._cosine_similarity(query_embedding, domain_embedding)
                confidences[domain] = max(similarity, 0.1)
        
        return confidences
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        np_a, np_b = np.array(a), np.array(b)
        return float(np.dot(np_a, np_b) / (np.linalg.norm(np_a) * np.linalg.norm(np_b)))

# Export main classes
__all__ = [
    "ModelInterface",
    "OllamaModel",
    "OpenAIModel", 
    "AnthropicModel",
    "HuggingFaceModel",
    "MockModel",
    "ModelFactory",
    "ConfidenceEstimator"
]