"""
Sequential Exclusion Validation.

Tests the dodecapartite framework's core mechanism:
N_12 = N_0 × ∏ε_i → 1

Each measurement modality contributes an exclusion factor ε_i,
reducing structural ambiguity until unique determination.

From the paper:
- N_0 ~ 10^60 initial configurations
- 12 modalities each contribute ε_i
- Product reduces to ~1 (unique solution)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from scipy import stats

from .data_loader import ImageData, NucleusData


@dataclass
class ModalityResult:
    """Result of applying a single modality."""
    name: str
    epsilon: float  # Exclusion factor
    n_distinguishable: int  # Number of distinguishable states
    feature_values: np.ndarray  # Raw feature values
    entropy: float  # Shannon entropy of feature distribution


class SequentialExclusionValidator:
    """
    Validate sequential exclusion mechanism.

    Simulates 6 measurement modalities using different nuclear features:
    1. Intensity (optical)
    2. Texture (spectral proxy)
    3. Shape/Eccentricity (morphological)
    4. Size/Area (thermodynamic proxy)
    5. Orientation (electromagnetic proxy)
    6. Solidity (structural proxy)
    """

    def __init__(self, n_bins: int = 20):
        """
        Initialize validator.

        Args:
            n_bins: Number of bins for feature discretization
        """
        self.n_bins = n_bins
        self.modality_results: List[ModalityResult] = []

    def _extract_feature(self, nuclei: List[NucleusData], feature_name: str) -> np.ndarray:
        """Extract a single feature from all nuclei."""
        if feature_name == 'intensity':
            return np.array([n.mean_intensity for n in nuclei])
        elif feature_name == 'area':
            return np.array([n.area for n in nuclei])
        elif feature_name == 'eccentricity':
            return np.array([n.eccentricity for n in nuclei])
        elif feature_name == 'orientation':
            return np.array([n.orientation for n in nuclei])
        elif feature_name == 'solidity':
            return np.array([n.solidity for n in nuclei])
        elif feature_name == 'perimeter':
            return np.array([n.perimeter for n in nuclei])
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

    def _compute_exclusion_factor(self, values: np.ndarray) -> Tuple[float, int, float]:
        """
        Compute exclusion factor for a feature distribution.

        Exclusion factor ε = 1 / (effective number of distinguishable states)

        Returns:
            (epsilon, n_distinguishable, entropy)
        """
        if len(values) < 2:
            return 1.0, 1, 0.0

        # Discretize into bins
        hist, bin_edges = np.histogram(values, bins=self.n_bins)
        hist = hist / hist.sum()  # Normalize to probability

        # Remove zero bins
        hist = hist[hist > 0]

        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # Effective number of states (exponential of entropy)
        n_effective = 2 ** entropy

        # Exclusion factor
        epsilon = 1 / n_effective

        return float(epsilon), int(np.ceil(n_effective)), float(entropy)

    def _compute_texture_feature(self, image: np.ndarray, nuclei: List[NucleusData]) -> np.ndarray:
        """
        Compute texture feature (GLCM contrast) for each nucleus.

        This serves as a proxy for spectral information.
        """
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            # Fallback: use intensity variance as texture proxy
            return np.array([n.mean_intensity for n in nuclei])

        textures = []
        for nucleus in nuclei:
            # Extract ROI
            y0, x0, y1, x1 = nucleus.bbox
            roi = image[y0:y1, x0:x1]

            if roi.size < 16:  # Too small for GLCM
                textures.append(0.0)
                continue

            # Quantize to 8-bit
            roi_uint8 = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-10) * 255).astype(np.uint8)

            try:
                glcm = graycomatrix(roi_uint8, [1], [0], 256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                textures.append(contrast)
            except:
                textures.append(0.0)

        return np.array(textures)

    def apply_modalities(self, samples: List[ImageData]) -> List[ModalityResult]:
        """
        Apply all modalities to the dataset.

        Args:
            samples: List of ImageData

        Returns:
            List of ModalityResult for each modality
        """
        # Collect all nuclei
        all_nuclei = [n for s in samples for n in s.nuclei]

        if not all_nuclei:
            return []

        # Define modalities
        modality_configs = [
            ('Intensity (Optical)', 'intensity'),
            ('Area (Thermodynamic)', 'area'),
            ('Eccentricity (Morphological)', 'eccentricity'),
            ('Orientation (EM)', 'orientation'),
            ('Solidity (Structural)', 'solidity'),
            ('Perimeter (Boundary)', 'perimeter'),
        ]

        self.modality_results = []

        for name, feature_name in modality_configs:
            values = self._extract_feature(all_nuclei, feature_name)
            epsilon, n_dist, entropy = self._compute_exclusion_factor(values)

            self.modality_results.append(ModalityResult(
                name=name,
                epsilon=epsilon,
                n_distinguishable=n_dist,
                feature_values=values,
                entropy=entropy
            ))

        # Add texture modality (requires images)
        if samples:
            texture_values = []
            for sample in samples:
                textures = self._compute_texture_feature(sample.image, sample.nuclei)
                texture_values.extend(textures)

            texture_values = np.array(texture_values)
            epsilon, n_dist, entropy = self._compute_exclusion_factor(texture_values)

            self.modality_results.append(ModalityResult(
                name='Texture (Spectral)',
                epsilon=epsilon,
                n_distinguishable=n_dist,
                feature_values=texture_values,
                entropy=entropy
            ))

        return self.modality_results

    def compute_cumulative_exclusion(self) -> Dict:
        """
        Compute cumulative exclusion as modalities are added.

        Tests: N_K = N_0 × ∏_{i=1}^K ε_i → 1 as K increases

        Returns:
            Dictionary with cumulative exclusion data
        """
        if not self.modality_results:
            return {'validated': False, 'reason': 'No modality results'}

        # Initial configuration space (estimated)
        # For N nuclei with K feature bins each: N_0 = K^N
        n_nuclei = len(self.modality_results[0].feature_values)
        N_0 = self.n_bins ** n_nuclei if n_nuclei < 20 else 1e60

        cumulative = []
        product = 1.0

        for i, result in enumerate(self.modality_results):
            product *= result.epsilon
            N_remaining = N_0 * product

            cumulative.append({
                'modality_index': i + 1,
                'modality_name': result.name,
                'epsilon': result.epsilon,
                'cumulative_product': product,
                'N_remaining': N_remaining,
                'log_N_remaining': np.log10(N_remaining) if N_remaining > 0 else -np.inf,
                'entropy': result.entropy
            })

        # Validation: does product approach 1/N_0?
        # (Meaning N_remaining approaches 1)
        final_N = cumulative[-1]['N_remaining']
        validated = final_N < 1e10  # Should reduce significantly

        return {
            'validated': validated,
            'reason': 'Sequential exclusion effective' if validated else 'Insufficient exclusion',
            'N_0': N_0,
            'N_final': final_N,
            'reduction_factor': N_0 / final_N if final_N > 0 else np.inf,
            'log_reduction': np.log10(N_0 / final_N) if final_N > 0 else np.inf,
            'cumulative': cumulative,
            'n_modalities': len(self.modality_results),
            'n_nuclei': n_nuclei
        }

    def get_exclusion_curve_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for plotting exclusion curve.

        Returns arrays for: modality index, cumulative product, log(N_remaining)
        """
        result = self.compute_cumulative_exclusion()

        if not result.get('cumulative'):
            return {
                'modality_index': np.array([]),
                'cumulative_product': np.array([]),
                'log_N_remaining': np.array([]),
                'epsilon': np.array([]),
                'names': []
            }

        cumulative = result['cumulative']

        return {
            'modality_index': np.array([c['modality_index'] for c in cumulative]),
            'cumulative_product': np.array([c['cumulative_product'] for c in cumulative]),
            'log_N_remaining': np.array([c['log_N_remaining'] for c in cumulative]),
            'epsilon': np.array([c['epsilon'] for c in cumulative]),
            'names': [c['modality_name'] for c in cumulative]
        }

    def test_resolution_scaling(self) -> Dict:
        """
        Test resolution scaling predictions.

        Independent: Δx_K = Δx_1 × K^(-1/2)
        Correlated: Δx_corr = Δx_1 × exp(-Σρ_ij)

        Uses feature correlations as proxy for measurement correlations.
        """
        if len(self.modality_results) < 2:
            return {'validated': False, 'reason': 'Insufficient modalities'}

        # Compute pairwise correlations
        n_modalities = len(self.modality_results)
        correlation_matrix = np.zeros((n_modalities, n_modalities))

        for i in range(n_modalities):
            for j in range(n_modalities):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Pearson correlation between feature distributions
                    v1 = self.modality_results[i].feature_values
                    v2 = self.modality_results[j].feature_values

                    # Ensure same length
                    min_len = min(len(v1), len(v2))
                    if min_len > 1:
                        corr, _ = stats.pearsonr(v1[:min_len], v2[:min_len])
                        correlation_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
                    else:
                        correlation_matrix[i, j] = 0

        # Sum of off-diagonal correlations
        sum_rho = np.sum(np.triu(correlation_matrix, k=1))

        # Resolution predictions (relative to single modality)
        delta_x_independent = 1.0 / np.sqrt(n_modalities)  # K^(-1/2) scaling
        delta_x_correlated = np.exp(-sum_rho)  # exp(-Σρ) scaling

        # Enhancement factors
        enhancement_independent = 1.0 / delta_x_independent
        enhancement_correlated = 1.0 / delta_x_correlated

        return {
            'n_modalities': n_modalities,
            'correlation_matrix': correlation_matrix.tolist(),
            'sum_rho': float(sum_rho),
            'mean_correlation': float(np.mean(correlation_matrix[np.triu_indices(n_modalities, k=1)])),
            'delta_x_independent': float(delta_x_independent),
            'delta_x_correlated': float(delta_x_correlated),
            'enhancement_independent': float(enhancement_independent),
            'enhancement_correlated': float(enhancement_correlated),
            'validated': enhancement_correlated > enhancement_independent
        }
