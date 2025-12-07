"""
Hardware-Constrained Categorical Completion (HCCC) Algorithm.

Main algorithm for BMD-based image understanding through:
- Dual-objective region selection (ambiguity + stream coherence)
- Hierarchical network BMD integration
- Continuous hardware stream updates
- Local termination with perpetual network evolution
"""

import numpy as np
from typing import Dict, Any, List, Optional
from ..vision.bmd import HardwareBMDStream, NetworkBMD, BMDState
from ..categorical import AmbiguityCalculator, CategoricalCompletion
from ..regions import Region, ImageSegmenter
from .selection import RegionSelector
from .integration import HierarchicalIntegration
from .convergence import ConvergenceMonitor


class HCCCAlgorithm:
    """
    Hardware-Constrained Categorical Completion algorithm.

    Implements the complete algorithm:
    1. Initialize with hardware BMD stream
    2. Segment image into regions
    3. Iteratively select and process regions
    4. Build hierarchical network BMD
    5. Terminate on network coherence

    This is the S-Entropy framework made concrete for vision.
    """

    def __init__(
        self,
        hardware_stream: HardwareBMDStream,
        ambiguity_calculator: Optional[AmbiguityCalculator] = None,
        completion_engine: Optional[CategoricalCompletion] = None,
        lambda_stream: float = 0.5,
        coherence_threshold: float = 1.0,
        max_iterations: int = 1000,
        allow_revisitation: bool = True
    ):
        """
        Initialize HCCC algorithm.

        Args:
            hardware_stream: Hardware BMD stream measurer
            ambiguity_calculator: Ambiguity computation engine
            completion_engine: Categorical completion engine
            lambda_stream: Balance between ambiguity and stream coherence
            coherence_threshold: A_coherence for termination
            max_iterations: Maximum processing iterations
            allow_revisitation: Allow region revisitation
        """
        self.hardware_stream = hardware_stream
        self.lambda_stream = lambda_stream
        self.A_coherence = coherence_threshold
        self.max_iterations = max_iterations
        self.allow_revisitation = allow_revisitation

        # Initialize components
        self.ambiguity_calc = ambiguity_calculator or AmbiguityCalculator()
        self.completion = completion_engine or CategoricalCompletion()
        self.selector = RegionSelector(self.ambiguity_calc)
        self.integrator = HierarchicalIntegration()
        self.convergence = ConvergenceMonitor(coherence_threshold)

        # State
        self.network_bmd: Optional[NetworkBMD] = None
        self.current_hardware_stream: Optional[BMDState] = None

    def process_image(
        self,
        image: np.ndarray,
        segmentation_method: str = 'slic',
        segmentation_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process image through HCCC algorithm.

        Algorithm:
        1. Measure hardware BMD stream β^(stream)
        2. Initialize network BMD: β^(network)_0 = β^(stream)
        3. Segment image into regions {R_i}
        4. While regions available and not converged:
            a. Update hardware stream
            b. Select region maximizing: A(β^(network), R) - λ·D_stream
            c. Generate new BMD: β_{i+1} = Generate(β_i, R)
            d. Integrate into network: β^(network)_{i+1}
            e. Check revisitation criterion
            f. Check termination (network coherence)
        5. Return final network BMD and results

        Args:
            image: Input image (H x W x C)
            segmentation_method: Segmentation algorithm
            segmentation_params: Parameters for segmentation

        Returns:
            Dict containing:
            - network_bmd_final: Final network BMD state
            - processing_sequence: Order of region processing
            - ambiguity_history: Ambiguity at each step
            - stream_divergence_history: Stream divergence at each step
            - convergence_step: Step where coherence achieved
            - interpretation: High-level interpretation
            - metrics: Performance metrics
        """
        # 1. Initialize hardware stream
        self.current_hardware_stream = self.hardware_stream.measure_stream()

        # 2. Initialize network BMD
        self.network_bmd = NetworkBMD(
            initial_hardware_stream=self.current_hardware_stream,
            max_compound_order=5
        )

        # 3. Segment image
        segmenter = ImageSegmenter()
        params = segmentation_params or {}
        regions = segmenter.segment(image, method=segmentation_method, **params)

        print(f"Segmented into {len(regions)} regions")

        # Extract features for all regions
        for region in regions:
            region.extract_features()

        # 4. Main processing loop
        available_regions = set(r.id for r in regions)
        region_dict = {r.id: r for r in regions}

        iteration = 0
        history = {
            'ambiguity': [],
            'stream_divergence': [],
            'network_richness': [],
            'selected_regions': []
        }

        while (available_regions and
               iteration < self.max_iterations and
               not self.convergence.has_converged()):

            iteration += 1

            # a. Update hardware stream
            self.current_hardware_stream = self.hardware_stream.update_stream()

            # b. Select next region
            candidate_regions = [region_dict[rid] for rid in available_regions]

            selected_region = self.selector.select_next_region(
                network_bmd=self.network_bmd,
                available_regions=candidate_regions,
                hardware_stream=self.current_hardware_stream,
                lambda_stream=self.lambda_stream
            )

            if selected_region is None:
                print(f"No suitable region at iteration {iteration}")
                break

            print(f"Iteration {iteration}: Selected {selected_region.id}")

            # c. Generate new BMD through completion
            current_global_bmd = self.network_bmd.get_global_bmd()
            new_region_bmd = self.completion.generate_bmd(
                current_bmd=current_global_bmd,
                region=selected_region
            )

            # d. Integrate into network
            self.integrator.integrate(
                network_bmd=self.network_bmd,
                new_region_bmd=new_region_bmd,
                region_id=selected_region.id,
                processing_sequence=self.network_bmd.processing_sequence
            )

            # Mark region as processed
            selected_region.mark_processed(iteration, new_region_bmd)
            available_regions.remove(selected_region.id)

            # Record metrics
            ambiguity = self.ambiguity_calc.compute_network_ambiguity(
                self.network_bmd,
                selected_region
            )
            divergence = self.ambiguity_calc.stream_divergence(
                self.network_bmd,
                selected_region,
                self.current_hardware_stream
            )

            history['ambiguity'].append(ambiguity)
            history['stream_divergence'].append(divergence)
            history['network_richness'].append(
                self.network_bmd.network_categorical_richness()
            )
            history['selected_regions'].append(selected_region.id)

            # e. Check revisitation
            if self.allow_revisitation:
                revisit_ids = self._check_revisitation(region_dict)
                available_regions.update(revisit_ids)

            # f. Check convergence
            converged = self._check_termination(candidate_regions)
            self.convergence.update(
                network_bmd=self.network_bmd,
                iteration=iteration,
                ambiguity=ambiguity,
                divergence=divergence
            )

            if converged:
                print(f"Converged at iteration {iteration}")
                break

        # 5. Extract interpretation
        interpretation = self._extract_interpretation()

        # Compile results
        results = {
            'network_bmd_final': self.network_bmd,
            'processing_sequence': self.network_bmd.processing_sequence,
            'ambiguity_history': history['ambiguity'],
            'stream_divergence_history': history['stream_divergence'],
            'network_richness_history': history['network_richness'],
            'convergence_step': iteration,
            'interpretation': interpretation,
            'metrics': self._compute_metrics(history),
            'regions_processed': len(self.network_bmd.processing_sequence),
            'regions_total': len(regions)
        }

        return results

    def _check_revisitation(
        self,
        region_dict: Dict[str, Region]
    ) -> set:
        """
        Check if any processed regions should be revisited.

        Revisit R' if: A(β^(network)_{i+1}, R') > A(β^(network)_j, R')
        where R' was processed at step j.

        Returns:
            Set of region IDs to revisit
        """
        revisit = set()

        # Check recent processed regions (last 10)
        recent_ids = self.network_bmd.get_recent_regions(n=10)

        for region_id in recent_ids:
            if region_id not in region_dict:
                continue

            region = region_dict[region_id]

            # Get current ambiguity
            current_ambiguity = self.ambiguity_calc.compute_network_ambiguity(
                self.network_bmd,
                region
            )

            # Compare to ambiguity when first processed
            # (Would need to store historical ambiguities for exact comparison)
            # For now, use heuristic: revisit if very high ambiguity

            if current_ambiguity > self.A_coherence * 2.0:
                revisit.add(region_id)

        return revisit

    def _check_termination(
        self,
        available_regions: List[Region]
    ) -> bool:
        """
        Check if network coherence achieved.

        Terminate if: A(β^(network), R) < A_coherence for all R

        Returns:
            True if should terminate
        """
        if not available_regions:
            return True

        # Check ambiguity for all available regions
        for region in available_regions:
            ambiguity = self.ambiguity_calc.compute_network_ambiguity(
                self.network_bmd,
                region
            )

            if ambiguity >= self.A_coherence:
                return False  # Still high ambiguity, continue

        # All regions have low ambiguity → coherence achieved
        return True

    def _extract_interpretation(self) -> Dict[str, Any]:
        """
        Extract high-level interpretation from final network BMD.

        Returns:
            Dict with semantic interpretation
        """
        if self.network_bmd is None:
            return {}

        interpretation = {
            'n_regions_processed': len(self.network_bmd.processing_sequence),
            'network_richness': self.network_bmd.network_categorical_richness(),
            'phase_quality': self.network_bmd.global_bmd.phase_lock_quality(),
            'n_compounds': len(self.network_bmd.compound_bmds),
            'processing_order': self.network_bmd.processing_sequence,
            'hierarchical_structure': {
                'order_2': len(self.network_bmd.get_compound_bmds_by_order(2)),
                'order_3': len(self.network_bmd.get_compound_bmds_by_order(3)),
                'order_4': len(self.network_bmd.get_compound_bmds_by_order(4)),
                'order_5': len(self.network_bmd.get_compound_bmds_by_order(5)),
            }
        }

        return interpretation

    def _compute_metrics(self, history: Dict) -> Dict[str, Any]:
        """
        Compute performance metrics.

        Returns:
            Dict of metrics
        """
        metrics = {}

        if history['ambiguity']:
            metrics['final_ambiguity'] = history['ambiguity'][-1]
            metrics['mean_ambiguity'] = np.mean(history['ambiguity'])
            metrics['ambiguity_decrease'] = (
                history['ambiguity'][0] - history['ambiguity'][-1]
                if len(history['ambiguity']) > 1 else 0.0
            )

        if history['stream_divergence']:
            metrics['final_divergence'] = history['stream_divergence'][-1]
            metrics['mean_divergence'] = np.mean(history['stream_divergence'])

        if history['network_richness']:
            metrics['final_richness'] = history['network_richness'][-1]
            metrics['richness_growth'] = (
                history['network_richness'][-1] / (history['network_richness'][0] + 1e-10)
                if len(history['network_richness']) > 1 else 1.0
            )

        return metrics

    def get_network_state(self) -> Optional[NetworkBMD]:
        """Get current network BMD state."""
        return self.network_bmd

    def reset(self):
        """Reset algorithm state for new image."""
        self.network_bmd = None
        self.current_hardware_stream = None
        self.convergence.reset()
