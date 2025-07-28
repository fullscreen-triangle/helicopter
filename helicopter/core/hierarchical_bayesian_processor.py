"""
Hierarchical Bayesian Processing Framework

Implements three-level Bayesian hierarchy for uncertainty quantification:
1. Molecular Level: Characters, tokens, primitive visual elements
2. Neural Level: Syntactic and semantic parsing
3. Cognitive Level: Contextual integration and high-level reasoning

Features:
- Variational inference for uncertainty propagation
- Multi-scale integration
- Calibrated uncertainty estimates
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ProcessingLevel(Enum):
    """Hierarchical processing levels"""
    MOLECULAR = "molecular"    # Characters, tokens, primitive elements
    NEURAL = "neural"         # Syntactic and semantic parsing
    COGNITIVE = "cognitive"   # Contextual integration and reasoning


@dataclass
class BayesianState:
    """Bayesian state for a processing level"""
    mean: torch.Tensor                    # Posterior mean
    variance: torch.Tensor               # Posterior variance
    prior_mean: torch.Tensor            # Prior mean
    prior_variance: torch.Tensor        # Prior variance
    likelihood_precision: float         # Likelihood precision
    kl_divergence: float                # KL divergence from prior
    uncertainty: float                  # Total uncertainty estimate
    level: ProcessingLevel              # Processing level


@dataclass
class HierarchicalResult:
    """Result from hierarchical Bayesian processing"""
    molecular_state: BayesianState
    neural_state: BayesianState
    cognitive_state: BayesianState
    total_uncertainty: float
    calibration_score: float
    variational_bound: float
    processing_time: float


class BayesianProcessor(ABC):
    """Abstract base class for Bayesian processors at different levels"""
    
    @abstractmethod
    def forward(
        self, 
        observations: torch.Tensor,
        prior_state: Optional[BayesianState] = None
    ) -> BayesianState:
        """Process observations and return Bayesian state"""
        pass
    
    @abstractmethod
    def compute_kl_divergence(
        self, 
        posterior_mean: torch.Tensor,
        posterior_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor
    ) -> float:
        """Compute KL divergence between posterior and prior"""
        pass


class MolecularProcessor(BayesianProcessor):
    """Molecular-level Bayesian processor for primitive visual elements"""
    
    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 64,
        prior_variance: float = 1.0
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.prior_variance = prior_variance
        
        # Encoder for mean and variance
        self.encoder_mean = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        logger.info("Initialized Molecular Bayesian Processor")
    
    def forward(
        self, 
        observations: torch.Tensor,
        prior_state: Optional[BayesianState] = None
    ) -> BayesianState:
        """Process molecular-level observations"""
        batch_size = observations.shape[0]
        
        # Encode to posterior parameters
        posterior_mean = self.encoder_mean(observations)
        posterior_logvar = self.encoder_logvar(observations)
        posterior_var = torch.exp(posterior_logvar)
        
        # Set prior (if not provided, use standard prior)
        if prior_state is None:
            prior_mean = torch.zeros_like(posterior_mean)
            prior_var = torch.full_like(posterior_var, self.prior_variance)
        else:
            prior_mean = prior_state.mean
            prior_var = prior_state.variance
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(
            posterior_mean, posterior_var, prior_mean, prior_var
        )
        
        # Estimate uncertainty (trace of covariance)
        uncertainty = torch.mean(posterior_var).item()
        
        return BayesianState(
            mean=posterior_mean,
            variance=posterior_var,
            prior_mean=prior_mean,
            prior_variance=prior_var,
            likelihood_precision=1.0,  # Will be learned
            kl_divergence=kl_div,
            uncertainty=uncertainty,
            level=ProcessingLevel.MOLECULAR
        )
    
    def compute_kl_divergence(
        self, 
        posterior_mean: torch.Tensor,
        posterior_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor
    ) -> float:
        """Compute KL divergence for Gaussian distributions"""
        # KL(q||p) = 0.5 * (log(σ²_p/σ²_q) + σ²_q/σ²_p + (μ_q-μ_p)²/σ²_p - 1)
        kl_div = 0.5 * torch.sum(
            torch.log(prior_var / posterior_var) +
            posterior_var / prior_var +
            (posterior_mean - prior_mean)**2 / prior_var - 1.0
        )
        return kl_div.item()


class NeuralProcessor(BayesianProcessor):
    """Neural-level Bayesian processor for syntactic and semantic parsing"""
    
    def __init__(
        self,
        molecular_dim: int = 64,
        neural_dim: int = 128,
        attention_heads: int = 8
    ):
        self.molecular_dim = molecular_dim
        self.neural_dim = neural_dim
        self.attention_heads = attention_heads
        
        # Attention mechanism for molecular-to-neural processing
        self.attention = nn.MultiheadAttention(
            embed_dim=molecular_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        
        # Neural processing layers
        self.neural_mean = nn.Sequential(
            nn.Linear(molecular_dim, neural_dim),
            nn.LayerNorm(neural_dim),
            nn.ReLU(),
            nn.Linear(neural_dim, neural_dim)
        )
        
        self.neural_logvar = nn.Sequential(
            nn.Linear(molecular_dim, neural_dim),
            nn.LayerNorm(neural_dim),
            nn.ReLU(),
            nn.Linear(neural_dim, neural_dim)
        )
        
        logger.info("Initialized Neural Bayesian Processor")
    
    def forward(
        self, 
        molecular_state: BayesianState,
        prior_state: Optional[BayesianState] = None
    ) -> BayesianState:
        """Process neural-level features from molecular state"""
        # Sample from molecular posterior
        molecular_samples = self._sample_from_posterior(
            molecular_state.mean, molecular_state.variance
        )
        
        # Apply attention to molecular features
        attended_features, attention_weights = self.attention(
            molecular_samples, molecular_samples, molecular_samples
        )
        
        # Process through neural layers
        neural_mean = self.neural_mean(attended_features.mean(dim=1))
        neural_logvar = self.neural_logvar(attended_features.mean(dim=1))
        neural_var = torch.exp(neural_logvar)
        
        # Hierarchical prior: conditioned on molecular state
        if prior_state is None:
            prior_mean = torch.zeros_like(neural_mean)
            # Prior variance increases with molecular uncertainty
            prior_var = torch.full_like(neural_var, 1.0 + molecular_state.uncertainty)
        else:
            prior_mean = prior_state.mean
            prior_var = prior_state.variance
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(
            neural_mean, neural_var, prior_mean, prior_var
        )
        
        # Propagate uncertainty from molecular level
        molecular_uncertainty = molecular_state.uncertainty
        neural_uncertainty = torch.mean(neural_var).item()
        total_uncertainty = molecular_uncertainty + neural_uncertainty
        
        return BayesianState(
            mean=neural_mean,
            variance=neural_var,
            prior_mean=prior_mean,
            prior_variance=prior_var,
            likelihood_precision=1.0,
            kl_divergence=kl_div,
            uncertainty=total_uncertainty,
            level=ProcessingLevel.NEURAL
        )
    
    def _sample_from_posterior(
        self, 
        mean: torch.Tensor, 
        variance: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """Sample from posterior distribution"""
        batch_size, dim = mean.shape
        
        # Reparameterization trick
        eps = torch.randn(num_samples, batch_size, dim)
        std = torch.sqrt(variance)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        
        return samples
    
    def compute_kl_divergence(
        self, 
        posterior_mean: torch.Tensor,
        posterior_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor
    ) -> float:
        """Compute KL divergence for Gaussian distributions"""
        kl_div = 0.5 * torch.sum(
            torch.log(prior_var / posterior_var) +
            posterior_var / prior_var +
            (posterior_mean - prior_mean)**2 / prior_var - 1.0
        )
        return kl_div.item()


class CognitiveProcessor(BayesianProcessor):
    """Cognitive-level Bayesian processor for contextual integration"""
    
    def __init__(
        self,
        neural_dim: int = 128,
        cognitive_dim: int = 256,
        context_length: int = 512
    ):
        self.neural_dim = neural_dim
        self.cognitive_dim = cognitive_dim
        self.context_length = context_length
        
        # Transformer for contextual processing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=neural_dim,
                nhead=8,
                dim_feedforward=cognitive_dim,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Cognitive state estimation
        self.cognitive_mean = nn.Sequential(
            nn.Linear(neural_dim, cognitive_dim),
            nn.LayerNorm(cognitive_dim),
            nn.GELU(),
            nn.Linear(cognitive_dim, cognitive_dim)
        )
        
        self.cognitive_logvar = nn.Sequential(
            nn.Linear(neural_dim, cognitive_dim),
            nn.LayerNorm(cognitive_dim),
            nn.GELU(),
            nn.Linear(cognitive_dim, cognitive_dim)
        )
        
        logger.info("Initialized Cognitive Bayesian Processor")
    
    def forward(
        self, 
        neural_state: BayesianState,
        prior_state: Optional[BayesianState] = None
    ) -> BayesianState:
        """Process cognitive-level features from neural state"""
        # Expand neural features for contextual processing
        neural_features = neural_state.mean.unsqueeze(1).repeat(1, self.context_length, 1)
        
        # Apply transformer for contextual understanding
        context_features = self.transformer(neural_features)
        aggregated_features = context_features.mean(dim=1)
        
        # Generate cognitive state
        cognitive_mean = self.cognitive_mean(aggregated_features)
        cognitive_logvar = self.cognitive_logvar(aggregated_features)
        cognitive_var = torch.exp(cognitive_logvar)
        
        # Hierarchical prior: conditioned on neural state
        if prior_state is None:
            prior_mean = torch.zeros_like(cognitive_mean)
            # Prior variance scales with accumulated uncertainty
            prior_var = torch.full_like(cognitive_var, 1.0 + neural_state.uncertainty)
        else:
            prior_mean = prior_state.mean
            prior_var = prior_state.variance
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(
            cognitive_mean, cognitive_var, prior_mean, prior_var
        )
        
        # Total uncertainty propagation
        total_uncertainty = neural_state.uncertainty + torch.mean(cognitive_var).item()
        
        return BayesianState(
            mean=cognitive_mean,
            variance=cognitive_var,
            prior_mean=prior_mean,
            prior_variance=prior_var,
            likelihood_precision=1.0,
            kl_divergence=kl_div,
            uncertainty=total_uncertainty,
            level=ProcessingLevel.COGNITIVE
        )
    
    def compute_kl_divergence(
        self, 
        posterior_mean: torch.Tensor,
        posterior_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor
    ) -> float:
        """Compute KL divergence for Gaussian distributions"""
        kl_div = 0.5 * torch.sum(
            torch.log(prior_var / posterior_var) +
            posterior_var / prior_var +
            (posterior_mean - prior_mean)**2 / prior_var - 1.0
        )
        return kl_div.item()


class HierarchicalBayesianProcessor:
    """
    Main hierarchical Bayesian processing framework.
    
    Coordinates processing across molecular, neural, and cognitive levels
    with uncertainty propagation and calibration.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        molecular_dim: int = 64,
        neural_dim: int = 128,
        cognitive_dim: int = 256
    ):
        self.molecular_processor = MolecularProcessor(input_dim, molecular_dim)
        self.neural_processor = NeuralProcessor(molecular_dim, neural_dim)
        self.cognitive_processor = CognitiveProcessor(neural_dim, cognitive_dim)
        
        self.calibration_temperature = nn.Parameter(torch.ones(1))
        
        logger.info("Initialized Hierarchical Bayesian Processor")
    
    def process_hierarchically(
        self, 
        observations: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[HierarchicalResult, Tuple[HierarchicalResult, Dict]]:
        """
        Process observations through hierarchical Bayesian framework.
        
        Args:
            observations: Input observations
            return_intermediates: Whether to return intermediate states
            
        Returns:
            HierarchicalResult with all processing levels
        """
        start_time = time.time()
        
        # Level 1: Molecular processing
        molecular_state = self.molecular_processor.forward(observations)
        
        # Level 2: Neural processing (conditioned on molecular)
        neural_state = self.neural_processor.forward(molecular_state)
        
        # Level 3: Cognitive processing (conditioned on neural)
        cognitive_state = self.cognitive_processor.forward(neural_state)
        
        # Compute total uncertainty
        total_uncertainty = cognitive_state.uncertainty
        
        # Compute variational bound (ELBO)
        variational_bound = self._compute_variational_bound(
            molecular_state, neural_state, cognitive_state
        )
        
        # Calibrate uncertainty
        calibration_score = self._calibrate_uncertainty(cognitive_state)
        
        processing_time = time.time() - start_time
        
        result = HierarchicalResult(
            molecular_state=molecular_state,
            neural_state=neural_state,
            cognitive_state=cognitive_state,
            total_uncertainty=total_uncertainty,
            calibration_score=calibration_score,
            variational_bound=variational_bound,
            processing_time=processing_time
        )
        
        if return_intermediates:
            intermediates = {
                'molecular_features': molecular_state.mean,
                'neural_features': neural_state.mean,
                'cognitive_features': cognitive_state.mean,
                'attention_weights': getattr(self.neural_processor, 'last_attention_weights', None)
            }
            return result, intermediates
        
        return result
    
    def _compute_variational_bound(
        self,
        molecular_state: BayesianState,
        neural_state: BayesianState,
        cognitive_state: BayesianState
    ) -> float:
        """Compute Evidence Lower BOund (ELBO)"""
        # ELBO = E[log p(x|z)] - KL[q(z)||p(z)]
        # For hierarchical model, sum across levels
        total_kl = (
            molecular_state.kl_divergence +
            neural_state.kl_divergence +
            cognitive_state.kl_divergence
        )
        
        # Simplified likelihood term (would be computed properly in practice)
        likelihood_term = -0.5 * torch.sum(cognitive_state.variance).item()
        
        return likelihood_term - total_kl
    
    def _calibrate_uncertainty(self, cognitive_state: BayesianState) -> float:
        """Calibrate uncertainty estimates using temperature scaling"""
        # Temperature scaling for calibration
        scaled_variance = cognitive_state.variance / self.calibration_temperature**2
        
        # Calibration score based on variance magnitude
        calibration_score = 1.0 / (1.0 + torch.mean(scaled_variance).item())
        
        return calibration_score
    
    def compute_expected_calibration_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE) as in the paper.
        
        ECE = Σ |B_m|/n * |acc(B_m) - conf(B_m)|
        """
        # Convert predictions to confidences
        confidences = torch.softmax(predictions / self.calibration_temperature, dim=-1).max(dim=-1)[0]
        
        # Get predicted classes
        predicted_classes = predictions.argmax(dim=-1)
        correct = (predicted_classes == targets).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def get_uncertainty_report(self, result: HierarchicalResult) -> str:
        """Generate uncertainty analysis report"""
        report = f"""
Hierarchical Bayesian Processing Report:
=======================================
Molecular Level:
  - Uncertainty: {result.molecular_state.uncertainty:.4f}
  - KL Divergence: {result.molecular_state.kl_divergence:.4f}

Neural Level:
  - Uncertainty: {result.neural_state.uncertainty:.4f}
  - KL Divergence: {result.neural_state.kl_divergence:.4f}

Cognitive Level:
  - Uncertainty: {result.cognitive_state.uncertainty:.4f}
  - KL Divergence: {result.cognitive_state.kl_divergence:.4f}

Total Results:
  - Total Uncertainty: {result.total_uncertainty:.4f}
  - Calibration Score: {result.calibration_score:.4f}
  - Variational Bound: {result.variational_bound:.4f}
  - Processing Time: {result.processing_time:.3f}s
        """
        return report


# Import time for timing measurements
import time 