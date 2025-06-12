"""
Bayesian Objective Engine with Fuzzy Logic

Implements the objective function that the metacognitive orchestrator optimizes.
Recognizes that images are probabilistic collections of pixels, not deterministic data.

Key Components:
1. Bayesian Belief Network - Updates beliefs about image content probabilistically
2. Fuzzy Logic System - Handles the continuous, non-binary nature of pixel data
3. Evidence Accumulation - Combines multiple sources of uncertain evidence
4. Objective Function - Provides optimization target for the orchestrator

This addresses the fundamental issue that pixels are not binary objects - there are
infinite shades of possibility for any given visual description.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import math
from scipy.stats import beta, norm
from sklearn.mixture import GaussianMixture
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class FuzzySet:
    """Represents a fuzzy set with membership function"""
    name: str
    center: float
    width: float
    membership_type: str = 'gaussian'  # 'gaussian', 'triangular', 'trapezoidal'
    
    def membership(self, x: float) -> float:
        """Calculate membership degree for value x"""
        if self.membership_type == 'gaussian':
            return math.exp(-0.5 * ((x - self.center) / self.width) ** 2)
        elif self.membership_type == 'triangular':
            if abs(x - self.center) >= self.width:
                return 0.0
            return 1.0 - abs(x - self.center) / self.width
        elif self.membership_type == 'trapezoidal':
            # Simplified trapezoidal
            if abs(x - self.center) <= self.width / 2:
                return 1.0
            elif abs(x - self.center) >= self.width:
                return 0.0
            else:
                return 1.0 - (abs(x - self.center) - self.width/2) / (self.width/2)
        return 0.0


@dataclass
class BeliefNode:
    """Node in the Bayesian belief network"""
    name: str
    prior_belief: float
    current_belief: float
    evidence_history: List[float] = field(default_factory=list)
    uncertainty: float = 0.5  # Epistemic uncertainty
    fuzzy_sets: List[FuzzySet] = field(default_factory=list)
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    
    def update_belief(self, evidence: float, confidence: float, fuzzy_membership: float = 1.0):
        """Update belief using Bayesian inference with fuzzy logic"""
        
        # Fuzzy-weighted evidence
        fuzzy_evidence = evidence * fuzzy_membership
        
        # Bayesian update with uncertainty
        likelihood = confidence
        
        # Prior odds
        prior_odds = self.current_belief / (1 - self.current_belief + 1e-8)
        
        # Likelihood ratio with fuzzy weighting
        likelihood_ratio = (likelihood * fuzzy_evidence) / ((1 - likelihood) * (1 - fuzzy_evidence) + 1e-8)
        
        # Posterior odds
        posterior_odds = prior_odds * likelihood_ratio
        
        # Convert back to probability
        new_belief = posterior_odds / (1 + posterior_odds)
        
        # Update with uncertainty decay
        uncertainty_factor = 1.0 - self.uncertainty
        self.current_belief = (uncertainty_factor * new_belief + 
                              self.uncertainty * self.current_belief)
        
        # Store evidence
        self.evidence_history.append(fuzzy_evidence)
        
        # Update uncertainty (decreases with more evidence)
        self.uncertainty *= 0.95  # Gradual uncertainty reduction


@dataclass
class FuzzyEvidence:
    """Evidence with fuzzy logic properties"""
    source: str
    value: float
    confidence: float
    linguistic_label: str
    fuzzy_membership: Dict[str, float]  # Membership in different fuzzy sets
    pixel_support: np.ndarray  # Which pixels support this evidence
    spatial_distribution: Dict[str, float]  # Spatial properties


class FuzzyLogicProcessor:
    """Processes evidence using fuzzy logic principles"""
    
    def __init__(self):
        # Define standard fuzzy sets for visual properties
        self.visual_fuzzy_sets = {
            'brightness': [
                FuzzySet('very_dark', 0.0, 0.2),
                FuzzySet('dark', 0.25, 0.15),
                FuzzySet('medium', 0.5, 0.2),
                FuzzySet('bright', 0.75, 0.15),
                FuzzySet('very_bright', 1.0, 0.2)
            ],
            'contrast': [
                FuzzySet('low', 0.0, 0.3),
                FuzzySet('medium', 0.5, 0.3),
                FuzzySet('high', 1.0, 0.3)
            ],
            'motion': [
                FuzzySet('static', 0.0, 0.1),
                FuzzySet('slow', 0.3, 0.2),
                FuzzySet('moderate', 0.6, 0.2),
                FuzzySet('fast', 1.0, 0.3)
            ],
            'confidence': [
                FuzzySet('uncertain', 0.0, 0.3),
                FuzzySet('possible', 0.4, 0.2),
                FuzzySet('likely', 0.7, 0.2),
                FuzzySet('certain', 1.0, 0.2)
            ]
        }
    
    def fuzzify_evidence(self, evidence: Dict[str, Any]) -> FuzzyEvidence:
        """Convert crisp evidence to fuzzy evidence"""
        
        fuzzy_memberships = {}
        
        # Calculate membership in relevant fuzzy sets
        for property_name, value in evidence.items():
            if property_name in self.visual_fuzzy_sets:
                memberships = {}
                for fuzzy_set in self.visual_fuzzy_sets[property_name]:
                    memberships[fuzzy_set.name] = fuzzy_set.membership(value)
                fuzzy_memberships[property_name] = memberships
        
        # Determine linguistic label (highest membership)
        primary_label = self._get_primary_linguistic_label(fuzzy_memberships)
        
        # Extract spatial information if available
        spatial_dist = self._extract_spatial_distribution(evidence)
        
        return FuzzyEvidence(
            source=evidence.get('source', 'unknown'),
            value=evidence.get('primary_value', 0.5),
            confidence=evidence.get('confidence', 0.5),
            linguistic_label=primary_label,
            fuzzy_membership=fuzzy_memberships,
            pixel_support=evidence.get('pixel_mask', np.array([])),
            spatial_distribution=spatial_dist
        )
    
    def _get_primary_linguistic_label(self, fuzzy_memberships: Dict[str, Dict[str, float]]) -> str:
        """Get the primary linguistic label from fuzzy memberships"""
        
        max_membership = 0.0
        primary_label = "uncertain"
        
        for property_name, memberships in fuzzy_memberships.items():
            for label, membership in memberships.items():
                if membership > max_membership:
                    max_membership = membership
                    primary_label = f"{property_name}_{label}"
        
        return primary_label
    
    def _extract_spatial_distribution(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """Extract spatial distribution properties"""
        
        spatial_dist = {
            'centrality': 0.5,  # How central the evidence is
            'dispersion': 0.5,  # How spread out the evidence is
            'coherence': 0.5    # How coherent the spatial pattern is
        }
        
        if 'pixel_mask' in evidence and len(evidence['pixel_mask']) > 0:
            mask = evidence['pixel_mask']
            
            # Calculate centrality (distance from image center)
            if mask.ndim == 2:
                h, w = mask.shape
                center_y, center_x = h // 2, w // 2
                
                # Find centroid of evidence
                y_coords, x_coords = np.where(mask > 0.5)
                if len(y_coords) > 0:
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    
                    # Distance from center (normalized)
                    center_dist = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
                    max_dist = np.sqrt(center_y**2 + center_x**2)
                    spatial_dist['centrality'] = 1.0 - (center_dist / max_dist)
                    
                    # Dispersion (standard deviation of positions)
                    if len(y_coords) > 1:
                        dispersion = np.sqrt(np.var(y_coords) + np.var(x_coords))
                        spatial_dist['dispersion'] = min(1.0, dispersion / (max(h, w) / 4))
        
        return spatial_dist
    
    def combine_fuzzy_evidence(self, evidence_list: List[FuzzyEvidence]) -> FuzzyEvidence:
        """Combine multiple fuzzy evidence using fuzzy operators"""
        
        if not evidence_list:
            return FuzzyEvidence("empty", 0.0, 0.0, "uncertain", {}, np.array([]), {})
        
        if len(evidence_list) == 1:
            return evidence_list[0]
        
        # Combine using fuzzy aggregation operators
        combined_value = self._fuzzy_aggregate_values([e.value for e in evidence_list])
        combined_confidence = self._fuzzy_aggregate_confidences([e.confidence for e in evidence_list])
        
        # Combine fuzzy memberships
        combined_memberships = self._combine_fuzzy_memberships([e.fuzzy_membership for e in evidence_list])
        
        # Combine spatial distributions
        combined_spatial = self._combine_spatial_distributions([e.spatial_distribution for e in evidence_list])
        
        # Determine combined linguistic label
        combined_label = self._get_primary_linguistic_label(combined_memberships)
        
        return FuzzyEvidence(
            source="combined",
            value=combined_value,
            confidence=combined_confidence,
            linguistic_label=combined_label,
            fuzzy_membership=combined_memberships,
            pixel_support=np.concatenate([e.pixel_support for e in evidence_list if len(e.pixel_support) > 0]),
            spatial_distribution=combined_spatial
        )
    
    def _fuzzy_aggregate_values(self, values: List[float]) -> float:
        """Aggregate values using fuzzy operators"""
        
        if not values:
            return 0.0
        
        # Use ordered weighted averaging (OWA)
        sorted_values = sorted(values, reverse=True)
        weights = np.array([1.0 / (i + 1) for i in range(len(sorted_values))])
        weights = weights / np.sum(weights)
        
        return np.sum(sorted_values * weights)
    
    def _fuzzy_aggregate_confidences(self, confidences: List[float]) -> float:
        """Aggregate confidences using fuzzy operators"""
        
        if not confidences:
            return 0.0
        
        # Use fuzzy intersection (minimum) for conservative confidence
        return min(confidences)
    
    def _combine_fuzzy_memberships(self, membership_lists: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Combine fuzzy memberships from multiple evidence"""
        
        combined = defaultdict(lambda: defaultdict(float))
        
        for memberships in membership_lists:
            for property_name, property_memberships in memberships.items():
                for label, membership in property_memberships.items():
                    # Use fuzzy union (maximum) to combine memberships
                    combined[property_name][label] = max(combined[property_name][label], membership)
        
        return dict(combined)
    
    def _combine_spatial_distributions(self, spatial_lists: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine spatial distributions"""
        
        if not spatial_lists:
            return {'centrality': 0.5, 'dispersion': 0.5, 'coherence': 0.5}
        
        combined = {}
        for key in ['centrality', 'dispersion', 'coherence']:
            values = [spatial.get(key, 0.5) for spatial in spatial_lists]
            combined[key] = np.mean(values)  # Average spatial properties
        
        return combined


class BayesianBeliefNetwork:
    """Bayesian belief network with fuzzy logic integration"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.nodes: Dict[str, BeliefNode] = {}
        self.graph = nx.DiGraph()
        self.fuzzy_processor = FuzzyLogicProcessor()
        
        # Initialize domain-specific belief structure
        self._initialize_belief_structure()
    
    def _initialize_belief_structure(self):
        """Initialize belief network structure based on domain"""
        
        # Core visual analysis beliefs
        core_beliefs = [
            ('object_present', 0.5),
            ('motion_detected', 0.3),
            ('pose_identified', 0.4),
            ('physics_valid', 0.6),
            ('quality_sufficient', 0.7),
            ('semantic_meaningful', 0.5)
        ]
        
        for belief_name, prior in core_beliefs:
            self.add_belief_node(belief_name, prior)
        
        # Add domain-specific beliefs
        if self.domain == 'medical':
            medical_beliefs = [
                ('pathology_present', 0.2),
                ('normal_anatomy', 0.8),
                ('measurement_accurate', 0.6)
            ]
            for belief_name, prior in medical_beliefs:
                self.add_belief_node(belief_name, prior)
        
        elif self.domain == 'sports':
            sports_beliefs = [
                ('technique_correct', 0.5),
                ('performance_optimal', 0.4),
                ('injury_risk', 0.1)
            ]
            for belief_name, prior in sports_beliefs:
                self.add_belief_node(belief_name, prior)
        
        # Add causal relationships
        self._add_causal_relationships()
    
    def add_belief_node(self, name: str, prior_belief: float, fuzzy_sets: Optional[List[FuzzySet]] = None):
        """Add a belief node to the network"""
        
        if fuzzy_sets is None:
            fuzzy_sets = [
                FuzzySet('low', 0.2, 0.3),
                FuzzySet('medium', 0.5, 0.3),
                FuzzySet('high', 0.8, 0.3)
            ]
        
        node = BeliefNode(
            name=name,
            prior_belief=prior_belief,
            current_belief=prior_belief,
            fuzzy_sets=fuzzy_sets
        )
        
        self.nodes[name] = node
        self.graph.add_node(name)
        
        logger.debug(f"Added belief node: {name} with prior {prior_belief}")
    
    def add_causal_relationship(self, parent: str, child: str, strength: float = 1.0):
        """Add causal relationship between beliefs"""
        
        if parent in self.nodes and child in self.nodes:
            self.nodes[parent].children.add(child)
            self.nodes[child].parents.add(parent)
            self.graph.add_edge(parent, child, weight=strength)
            
            logger.debug(f"Added causal relationship: {parent} -> {child} (strength: {strength})")
    
    def _add_causal_relationships(self):
        """Add domain-appropriate causal relationships"""
        
        # Universal relationships
        self.add_causal_relationship('quality_sufficient', 'object_present', 0.8)
        self.add_causal_relationship('object_present', 'pose_identified', 0.7)
        self.add_causal_relationship('pose_identified', 'physics_valid', 0.6)
        self.add_causal_relationship('motion_detected', 'physics_valid', 0.5)
        self.add_causal_relationship('object_present', 'semantic_meaningful', 0.8)
        
        # Domain-specific relationships
        if self.domain == 'medical':
            if 'pathology_present' in self.nodes and 'normal_anatomy' in self.nodes:
                self.add_causal_relationship('pathology_present', 'normal_anatomy', -0.9)  # Negative relationship
                self.add_causal_relationship('quality_sufficient', 'measurement_accurate', 0.8)
        
        elif self.domain == 'sports':
            if 'technique_correct' in self.nodes:
                self.add_causal_relationship('pose_identified', 'technique_correct', 0.7)
                self.add_causal_relationship('technique_correct', 'performance_optimal', 0.6)
                self.add_causal_relationship('technique_correct', 'injury_risk', -0.5)  # Negative relationship
    
    def update_beliefs(self, evidence_dict: Dict[str, Any]) -> Dict[str, float]:
        """Update all beliefs based on new evidence using fuzzy logic"""
        
        # Convert evidence to fuzzy evidence
        fuzzy_evidence = self.fuzzy_processor.fuzzify_evidence(evidence_dict)
        
        # Update beliefs that have direct evidence
        updated_beliefs = {}
        
        for belief_name, node in self.nodes.items():
            if belief_name in evidence_dict or self._has_relevant_fuzzy_evidence(node, fuzzy_evidence):
                
                # Get evidence value and confidence
                if belief_name in evidence_dict:
                    evidence_value = evidence_dict[belief_name]
                    confidence = evidence_dict.get(f'{belief_name}_confidence', 0.7)
                else:
                    evidence_value = 0.5  # Neutral evidence
                    confidence = 0.5
                
                # Calculate fuzzy membership for this belief
                fuzzy_membership = self._calculate_fuzzy_membership(node, fuzzy_evidence)
                
                # Update belief with fuzzy logic
                old_belief = node.current_belief
                node.update_belief(evidence_value, confidence, fuzzy_membership)
                
                updated_beliefs[belief_name] = node.current_belief
                
                logger.debug(f"Updated {belief_name}: {old_belief:.3f} -> {node.current_belief:.3f} "
                           f"(fuzzy_membership: {fuzzy_membership:.3f})")
        
        # Propagate belief updates through causal network
        self._propagate_beliefs()
        
        return updated_beliefs
    
    def _has_relevant_fuzzy_evidence(self, node: BeliefNode, fuzzy_evidence: FuzzyEvidence) -> bool:
        """Check if fuzzy evidence is relevant to this belief node"""
        
        # Check if linguistic label matches node name
        if node.name.lower() in fuzzy_evidence.linguistic_label.lower():
            return True
        
        # Check fuzzy memberships for relevance
        for property_name, memberships in fuzzy_evidence.fuzzy_membership.items():
            for label, membership in memberships.items():
                if membership > 0.3 and (property_name in node.name or label in node.name):
                    return True
        
        return False
    
    def _calculate_fuzzy_membership(self, node: BeliefNode, fuzzy_evidence: FuzzyEvidence) -> float:
        """Calculate fuzzy membership degree for belief node given evidence"""
        
        max_membership = 0.0
        
        # Check membership in node's fuzzy sets
        for fuzzy_set in node.fuzzy_sets:
            membership = fuzzy_set.membership(fuzzy_evidence.value)
            max_membership = max(max_membership, membership)
        
        # Check relevance in evidence fuzzy memberships
        for property_name, memberships in fuzzy_evidence.fuzzy_membership.items():
            for label, membership in memberships.items():
                if property_name in node.name or label in node.name:
                    max_membership = max(max_membership, membership)
        
        return max_membership
    
    def _propagate_beliefs(self):
        """Propagate belief updates through the causal network"""
        
        # Topological sort for proper propagation order
        try:
            propagation_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Handle cycles by using approximate propagation
            propagation_order = list(self.nodes.keys())
        
        # Propagate in multiple passes for convergence
        for _ in range(3):  # Multiple passes for convergence
            for node_name in propagation_order:
                node = self.nodes[node_name]
                
                if node.parents:
                    # Calculate influence from parents
                    parent_influence = 0.0
                    total_weight = 0.0
                    
                    for parent_name in node.parents:
                        parent_node = self.nodes[parent_name]
                        edge_weight = self.graph[parent_name][node_name]['weight']
                        
                        # Fuzzy influence calculation
                        if edge_weight > 0:
                            influence = parent_node.current_belief * edge_weight
                        else:
                            influence = (1 - parent_node.current_belief) * abs(edge_weight)
                        
                        parent_influence += influence
                        total_weight += abs(edge_weight)
                    
                    if total_weight > 0:
                        # Combine with current belief using fuzzy operators
                        propagated_belief = parent_influence / total_weight
                        
                        # Fuzzy combination (weighted average with uncertainty)
                        uncertainty_weight = node.uncertainty
                        node.current_belief = (
                            (1 - uncertainty_weight) * propagated_belief +
                            uncertainty_weight * node.current_belief
                        )
    
    def get_objective_value(self) -> float:
        """Calculate objective function value for optimization"""
        
        # Weighted combination of all beliefs
        belief_weights = {
            'object_present': 0.2,
            'motion_detected': 0.15,
            'pose_identified': 0.15,
            'physics_valid': 0.15,
            'quality_sufficient': 0.2,
            'semantic_meaningful': 0.15
        }
        
        objective_value = 0.0
        total_weight = 0.0
        
        for belief_name, weight in belief_weights.items():
            if belief_name in self.nodes:
                node = self.nodes[belief_name]
                
                # Weight by inverse uncertainty (more certain beliefs contribute more)
                certainty_weight = 1.0 - node.uncertainty
                effective_weight = weight * certainty_weight
                
                objective_value += node.current_belief * effective_weight
                total_weight += effective_weight
        
        if total_weight > 0:
            objective_value /= total_weight
        
        # Add domain-specific objectives
        domain_objective = self._calculate_domain_objective()
        
        # Combine with fuzzy aggregation
        final_objective = 0.7 * objective_value + 0.3 * domain_objective
        
        return final_objective
    
    def _calculate_domain_objective(self) -> float:
        """Calculate domain-specific objective component"""
        
        if self.domain == 'medical':
            medical_beliefs = ['pathology_present', 'normal_anatomy', 'measurement_accurate']
            medical_values = [self.nodes[b].current_belief for b in medical_beliefs if b in self.nodes]
            return np.mean(medical_values) if medical_values else 0.5
        
        elif self.domain == 'sports':
            sports_beliefs = ['technique_correct', 'performance_optimal']
            sports_values = [self.nodes[b].current_belief for b in sports_beliefs if b in sports_beliefs]
            injury_penalty = self.nodes.get('injury_risk', BeliefNode('', 0, 0)).current_belief
            sports_objective = np.mean(sports_values) if sports_values else 0.5
            return sports_objective * (1.0 - injury_penalty)  # Penalize injury risk
        
        return 0.5  # Neutral for unknown domains
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of current belief state"""
        
        summary = {
            'objective_value': self.get_objective_value(),
            'beliefs': {},
            'uncertainties': {},
            'evidence_counts': {}
        }
        
        for name, node in self.nodes.items():
            summary['beliefs'][name] = node.current_belief
            summary['uncertainties'][name] = node.uncertainty
            summary['evidence_counts'][name] = len(node.evidence_history)
        
        return summary
    
    def reset_beliefs(self):
        """Reset all beliefs to their priors"""
        
        for node in self.nodes.values():
            node.current_belief = node.prior_belief
            node.evidence_history.clear()
            node.uncertainty = 0.5
        
        logger.info("Reset all beliefs to priors")


class BayesianObjectiveEngine:
    """
    Main engine that provides the objective function for metacognitive optimization
    
    Uses Bayesian belief networks with fuzzy logic to handle the probabilistic,
    non-deterministic nature of image analysis.
    """
    
    def __init__(self, domain: str):
        self.domain = domain
        self.belief_network = BayesianBeliefNetwork(domain)
        self.fuzzy_processor = FuzzyLogicProcessor()
        
        # Optimization history
        self.objective_history = []
        self.evidence_history = []
        
        logger.info(f"Initialized Bayesian Objective Engine for domain: {domain}")
    
    def update_objective(self, analysis_results: Dict[str, Any], image_data: Optional[np.ndarray] = None) -> float:
        """Update objective function based on analysis results"""
        
        # Extract evidence from analysis results
        evidence = self._extract_evidence_from_analysis(analysis_results, image_data)
        
        # Update belief network
        updated_beliefs = self.belief_network.update_beliefs(evidence)
        
        # Calculate new objective value
        objective_value = self.belief_network.get_objective_value()
        
        # Store history
        self.objective_history.append(objective_value)
        self.evidence_history.append(evidence)
        
        logger.debug(f"Updated objective value: {objective_value:.3f}")
        
        return objective_value
    
    def _extract_evidence_from_analysis(self, analysis_results: Dict[str, Any], image_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Extract evidence from analysis results for belief updating"""
        
        evidence = {}
        
        # Extract from different analysis methods
        for method_name, result in analysis_results.items():
            if isinstance(result, dict):
                
                # Object detection evidence
                if 'objects' in result or 'detections' in result:
                    evidence['object_present'] = min(1.0, result.get('confidence', 0.5))
                    evidence['object_present_confidence'] = result.get('confidence', 0.5)
                
                # Motion evidence
                if 'motion' in method_name.lower() or 'optical_flow' in method_name.lower():
                    motion_magnitude = result.get('magnitude', result.get('motion_energy', 0.0))
                    evidence['motion_detected'] = min(1.0, motion_magnitude)
                    evidence['motion_detected_confidence'] = result.get('confidence', 0.5)
                
                # Pose evidence
                if 'pose' in method_name.lower() or 'joint' in method_name.lower():
                    pose_confidence = result.get('confidence', result.get('pose_confidence', 0.0))
                    evidence['pose_identified'] = pose_confidence
                    evidence['pose_identified_confidence'] = pose_confidence
                
                # Physics validation evidence
                if 'physics' in method_name.lower() or 'validation' in method_name.lower():
                    physics_valid = result.get('valid', result.get('physics_valid', 0.5))
                    evidence['physics_valid'] = physics_valid if isinstance(physics_valid, (int, float)) else 0.5
                    evidence['physics_valid_confidence'] = result.get('confidence', 0.6)
                
                # Quality evidence
                if 'quality' in method_name.lower() or 'assessment' in method_name.lower():
                    quality_score = result.get('quality_score', result.get('score', 0.5))
                    evidence['quality_sufficient'] = quality_score
                    evidence['quality_sufficient_confidence'] = result.get('confidence', 0.7)
                
                # Semantic evidence
                if 'semantic' in method_name.lower() or 'pakati' in method_name.lower():
                    semantic_score = result.get('semantic_score', result.get('meaningfulness', 0.5))
                    evidence['semantic_meaningful'] = semantic_score
                    evidence['semantic_meaningful_confidence'] = result.get('confidence', 0.5)
        
        # Add image-based evidence if available
        if image_data is not None:
            image_evidence = self._extract_image_evidence(image_data)
            evidence.update(image_evidence)
        
        # Add meta-analysis evidence
        if '_meta' in analysis_results:
            meta = analysis_results['_meta']
            evidence['quality_sufficient'] = meta.get('overall_confidence', 0.5)
            evidence['source'] = 'meta_analysis'
        
        return evidence
    
    def _extract_image_evidence(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Extract evidence directly from image pixels"""
        
        evidence = {}
        
        # Basic image statistics
        if len(image_data.shape) == 3:
            # Color image
            gray = np.mean(image_data, axis=2)
        else:
            gray = image_data
        
        # Brightness evidence
        mean_brightness = np.mean(gray) / 255.0
        evidence['brightness'] = mean_brightness
        
        # Contrast evidence
        contrast = np.std(gray) / 255.0
        evidence['contrast'] = contrast
        
        # Edge density (proxy for detail/quality)
        edges = np.gradient(gray)
        edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
        edge_density = np.mean(edge_magnitude) / 255.0
        evidence['edge_density'] = edge_density
        
        # Spatial coherence (how organized the image is)
        # Simple measure: variance of local means
        kernel_size = min(32, gray.shape[0]//8, gray.shape[1]//8)
        if kernel_size > 1:
            local_means = []
            for i in range(0, gray.shape[0]-kernel_size, kernel_size):
                for j in range(0, gray.shape[1]-kernel_size, kernel_size):
                    local_mean = np.mean(gray[i:i+kernel_size, j:j+kernel_size])
                    local_means.append(local_mean)
            
            if local_means:
                coherence = 1.0 - (np.std(local_means) / 255.0)  # Higher std = less coherent
                evidence['spatial_coherence'] = max(0.0, coherence)
        
        # Create pixel mask for spatial analysis
        # Use edge detection as a simple proxy for "interesting" pixels
        threshold = np.percentile(edge_magnitude, 75)  # Top 25% of edges
        evidence['pixel_mask'] = (edge_magnitude > threshold).astype(float)
        
        return evidence
    
    def get_optimization_target(self) -> float:
        """Get current optimization target value"""
        return self.belief_network.get_objective_value()
    
    def get_optimization_gradient(self) -> Dict[str, float]:
        """Get gradient information for optimization"""
        
        gradient = {}
        current_objective = self.belief_network.get_objective_value()
        
        # Calculate approximate gradient for each belief
        for belief_name, node in self.belief_network.nodes.items():
            # Small perturbation
            original_belief = node.current_belief
            
            # Positive perturbation
            node.current_belief = min(1.0, original_belief + 0.01)
            pos_objective = self.belief_network.get_objective_value()
            
            # Negative perturbation
            node.current_belief = max(0.0, original_belief - 0.01)
            neg_objective = self.belief_network.get_objective_value()
            
            # Restore original
            node.current_belief = original_belief
            
            # Calculate gradient
            gradient[belief_name] = (pos_objective - neg_objective) / 0.02
        
        return gradient
    
    def get_uncertainty_map(self) -> Dict[str, float]:
        """Get uncertainty map for all beliefs"""
        
        return {name: node.uncertainty for name, node in self.belief_network.nodes.items()}
    
    def get_fuzzy_state(self) -> Dict[str, Any]:
        """Get current fuzzy logic state"""
        
        fuzzy_state = {
            'linguistic_labels': {},
            'membership_degrees': {},
            'fuzzy_rules_active': []
        }
        
        # Get linguistic labels for each belief
        for belief_name, node in self.belief_network.nodes.items():
            max_membership = 0.0
            best_label = "uncertain"
            
            for fuzzy_set in node.fuzzy_sets:
                membership = fuzzy_set.membership(node.current_belief)
                if membership > max_membership:
                    max_membership = membership
                    best_label = fuzzy_set.name
            
            fuzzy_state['linguistic_labels'][belief_name] = best_label
            fuzzy_state['membership_degrees'][belief_name] = max_membership
        
        return fuzzy_state
    
    def save_state(self, save_path: str):
        """Save the current state of the objective engine"""
        
        import pickle
        from pathlib import Path
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'domain': self.domain,
            'belief_network': self.belief_network,
            'objective_history': self.objective_history,
            'evidence_history': self.evidence_history
        }
        
        with open(save_dir / 'bayesian_objective_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved Bayesian objective state to {save_path}")
    
    def load_state(self, save_path: str):
        """Load the state of the objective engine"""
        
        import pickle
        from pathlib import Path
        
        state_file = Path(save_path) / 'bayesian_objective_state.pkl'
        
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            self.domain = state['domain']
            self.belief_network = state['belief_network']
            self.objective_history = state['objective_history']
            self.evidence_history = state['evidence_history']
            
            logger.info(f"Loaded Bayesian objective state from {save_path}")
        else:
            logger.warning(f"No saved state found at {save_path}") 