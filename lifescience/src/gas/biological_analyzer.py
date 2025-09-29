"""
Biological Gas Analyzer - High-level analyzer for life science applications
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
import logging

from . import InformationGasMolecule, GasMolecularSystem, MoleculeType, BiologicalProperties

logger = logging.getLogger(__name__)


class BiologicalGasAnalyzer:
    """
    High-level analyzer for biological applications of gas molecular dynamics.
    
    Provides convenient interfaces for common life science applications like
    protein analysis, cellular dynamics, and metabolic modeling.
    """
    
    def __init__(self):
        self.system = GasMolecularSystem()
        self.analysis_history = []
    
    def analyze_protein_structure(self, protein_image: np.ndarray, 
                                structure_type: str = "folded") -> Dict[str, Any]:
        """Analyze protein structure using gas molecular dynamics"""
        logger.info(f"Analyzing {structure_type} protein structure")
        
        # Convert image regions to information molecules
        molecules = self._image_to_protein_molecules(protein_image, structure_type)
        
        # Initialize system
        self.system = GasMolecularSystem(molecules)
        
        # Evolve to equilibrium
        evolution_results = self.system.evolve(steps=1000)
        
        # Extract biological meaning
        biological_analysis = self._extract_biological_meaning()
        
        results = {
            'protein_type': structure_type,
            'num_residues': len(molecules),
            'evolution_results': evolution_results,
            'biological_analysis': biological_analysis,
            'folding_quality': self._assess_folding_quality(biological_analysis),
            'binding_sites': self._identify_binding_sites(biological_analysis)
        }
        
        self.analysis_history.append(results)
        return results
    
    def _image_to_protein_molecules(self, image: np.ndarray, 
                                  structure_type: str) -> List[InformationGasMolecule]:
        """Convert protein image to information gas molecules"""
        molecules = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect protein regions
        # Use adaptive thresholding to find protein structures
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours (protein secondary structures)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:  # Filter small noise
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Create molecule at centroid
                    position = np.array([cx / 100.0, cy / 100.0, 0.0])  # Normalize
                    
                    # Determine biological properties based on structure
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                    
                    if structure_type == "folded":
                        bio_props = BiologicalProperties(
                            molecule_type=MoleculeType.PROTEIN,
                            biological_function="structural" if compactness > 0.5 else "catalytic",
                            cellular_location="cytoplasm",
                            stability=compactness,
                            activity_level=1.0 - compactness  # More elongated = more active
                        )
                    else:  # unfolded
                        bio_props = BiologicalProperties(
                            molecule_type=MoleculeType.PROTEIN,
                            biological_function="unfolded",
                            cellular_location="cytoplasm",
                            stability=0.3,
                            activity_level=0.1
                        )
                    
                    molecule = InformationGasMolecule(
                        position=position,
                        velocity=np.random.random(3) - 0.5,
                        mass=area / 1000.0,  # Mass proportional to area
                        biological_props=bio_props,
                        sigma=np.sqrt(area) / 100.0,
                        epsilon=1.0 + compactness,
                        information_content=area / 10000.0
                    )
                    
                    molecules.append(molecule)
        
        logger.info(f"Created {len(molecules)} protein molecules from image")
        return molecules
    
    def _extract_biological_meaning(self) -> Dict[str, Any]:
        """Extract biological meaning from equilibrium configuration"""
        props = self.system.calculate_system_properties()
        
        # Cluster molecules by position
        positions = np.array([mol.position for mol in self.system.molecules])
        
        # Simple clustering based on distance
        clusters = []
        visited = set()
        
        for i, mol in enumerate(self.system.molecules):
            if i in visited:
                continue
                
            cluster = [i]
            visited.add(i)
            
            for j, other_mol in enumerate(self.system.molecules):
                if j in visited:
                    continue
                    
                distance = np.linalg.norm(mol.position - other_mol.position)
                if distance < 2.0:  # Clustering threshold
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        # Analyze clusters biologically
        biological_clusters = []
        for cluster_indices in clusters:
            cluster_molecules = [self.system.molecules[i] for i in cluster_indices]
            cluster_types = [mol.biological_props.molecule_type for mol in cluster_molecules]
            
            biological_clusters.append({
                'size': len(cluster_indices),
                'molecule_types': cluster_types,
                'average_position': np.mean([self.system.molecules[i].position for i in cluster_indices], axis=0),
                'biological_function': self._infer_cluster_function(cluster_molecules)
            })
        
        return {
            'system_properties': props,
            'molecular_clusters': biological_clusters,
            'equilibrium_achieved': self._check_equilibrium(),
            'biological_interpretation': self._interpret_system_biologically(biological_clusters)
        }
    
    def _infer_cluster_function(self, cluster_molecules: List[InformationGasMolecule]) -> str:
        """Infer biological function of a molecular cluster"""
        functions = [mol.biological_props.biological_function for mol in cluster_molecules]
        
        # Simple majority voting
        function_counts = Counter(functions)
        most_common = function_counts.most_common(1)
        
        if most_common:
            return most_common[0][0]
        else:
            return "unknown"
    
    def _check_equilibrium(self) -> bool:
        """Check if system has reached equilibrium"""
        if len(self.system.total_energy_history) < 100:
            return False
        
        # Check energy stability over last 100 steps
        recent_energies = self.system.total_energy_history[-100:]
        energy_variance = np.var(recent_energies)
        
        return energy_variance < 0.01  # Equilibrium threshold
    
    def _interpret_system_biologically(self, clusters: List[Dict]) -> str:
        """Provide biological interpretation of the system state"""
        if not clusters:
            return "Dispersed system with minimal molecular interactions"
        
        num_clusters = len(clusters)
        largest_cluster = max(clusters, key=lambda c: c['size'])
        
        interpretation = f"System formed {num_clusters} distinct molecular complexes. "
        interpretation += f"Largest complex contains {largest_cluster['size']} molecules "
        interpretation += f"with primary function: {largest_cluster['biological_function']}. "
        
        if num_clusters == 1:
            interpretation += "Single complex formation suggests strong cooperative binding."
        elif num_clusters > 5:
            interpretation += "Multiple small complexes suggest competitive interactions."
        else:
            interpretation += "Moderate clustering indicates balanced interaction network."
        
        return interpretation
    
    def _assess_folding_quality(self, biological_analysis: Dict) -> Dict[str, Any]:
        """Assess protein folding quality from molecular analysis"""
        clusters = biological_analysis.get('molecular_clusters', [])
        system_props = biological_analysis.get('system_properties', {})
        
        # Well-folded proteins should form compact clusters
        if clusters:
            largest_cluster = max(clusters, key=lambda c: c['size'])
            compactness = largest_cluster['size'] / len(self.system.molecules)
            
            if compactness > 0.7:
                quality = "well_folded"
                score = 0.9
            elif compactness > 0.4:
                quality = "partially_folded"
                score = 0.6
            else:
                quality = "unfolded"
                score = 0.3
        else:
            quality = "unfolded"
            score = 0.1
            compactness = 0.0
        
        return {
            'quality': quality,
            'score': score,
            'compactness': compactness,
            'energy_stability': 1.0 / (1.0 + abs(system_props.get('total_energy', 0)))
        }
    
    def _identify_binding_sites(self, biological_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify potential binding sites in protein structure"""
        clusters = biological_analysis.get('molecular_clusters', [])
        binding_sites = []
        
        for i, cluster in enumerate(clusters):
            # Binding sites are typically smaller clusters with high activity
            if cluster['size'] < len(self.system.molecules) * 0.3:  # Less than 30% of total
                binding_sites.append({
                    'site_id': i,
                    'size': cluster['size'],
                    'position': cluster['average_position'],
                    'activity_level': 0.7,  # Simplified
                    'accessibility': 0.8,  # Simplified
                    'binding_potential': cluster['size'] / 10.0
                })
        
        return binding_sites
    
    def visualize_system(self, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the gas molecular system"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gas Molecular System Analysis', fontsize=16)
        
        # 1. Molecular positions
        if self.system.molecules:
            positions = np.array([mol.position for mol in self.system.molecules])
            
            scatter = axes[0, 0].scatter(positions[:, 0], positions[:, 1], 
                                       c=range(len(positions)), cmap='viridis', 
                                       s=50, alpha=0.7)
            axes[0, 0].set_title('Molecular Positions')
            axes[0, 0].set_xlabel('X Position')
            axes[0, 0].set_ylabel('Y Position')
            plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Energy evolution
        if self.system.total_energy_history:
            axes[0, 1].plot(self.system.total_energy_history)
            axes[0, 1].set_title('Total Energy Evolution')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Total Energy')
            axes[0, 1].grid(True)
        
        # 3. Temperature evolution
        if self.system.temperature_history:
            axes[0, 2].plot(self.system.temperature_history)
            axes[0, 2].set_title('Temperature Evolution')
            axes[0, 2].set_xlabel('Time Steps')
            axes[0, 2].set_ylabel('Temperature')
            axes[0, 2].grid(True)
        
        # 4. Entropy evolution
        if self.system.entropy_history:
            axes[1, 0].plot(self.system.entropy_history)
            axes[1, 0].set_title('Entropy Evolution')
            axes[1, 0].set_xlabel('Time Steps')
            axes[1, 0].set_ylabel('Total Entropy')
            axes[1, 0].grid(True)
        
        # 5. Velocity distribution
        if self.system.molecules:
            velocities = np.array([np.linalg.norm(mol.velocity) for mol in self.system.molecules])
            axes[1, 1].hist(velocities, bins=20, alpha=0.7, density=True)
            axes[1, 1].set_title('Velocity Distribution')
            axes[1, 1].set_xlabel('Speed')
            axes[1, 1].set_ylabel('Probability Density')
            axes[1, 1].grid(True)
        
        # 6. System properties summary
        props = self.system.calculate_system_properties()
        prop_names = list(props.keys())
        prop_values = list(props.values())
        
        axes[1, 2].barh(prop_names, prop_values)
        axes[1, 2].set_title('System Properties')
        axes[1, 2].set_xlabel('Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"System visualization saved to {save_path}")
        
        return fig
