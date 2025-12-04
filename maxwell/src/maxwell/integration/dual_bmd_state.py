"""
Dual-Membrane BMD State: Bridge between Pixel Demons and HCCC
============================================================

Converts pixel Maxwell demon dual states to HCCC BMD states,
maintaining conjugate front/back face structure.

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Import from parent maxwell module
sys.path.insert(0, str(Path(__file__).parent.parent))
from dual_membrane_pixel_demon import (
    DualMembranePixelDemon,
    SEntropyCoordinates,
    MembraneFace,
    ConjugateTransform
)

# Import from HCCC framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from vision.bmd import BMDState, OscillatoryHole, PhaseStructure


@dataclass
class DualMembraneBMDState:
    """
    Dual-membrane BMD state combining pixel demon and HCCC structures.
    
    This bridges:
    - Pixel demon dual states (S_front, S_back)
    - HCCC BMD states (categorical_state, oscillatory_holes, phase_structure)
    
    Maintains conjugate relationship: β_back = T(β_front)
    """
    # Front face BMD (observable)
    front_bmd: BMDState
    
    # Back face BMD (hidden, conjugate)
    back_bmd: BMDState
    
    # Observable face indicator
    observable_face: MembraneFace
    
    # Conjugate transform
    transform: ConjugateTransform
    
    # Source pixel demon (optional, for traceability)
    source_pixel_demon: Optional[DualMembranePixelDemon] = None
    
    def get_observable_bmd(self) -> BMDState:
        """Get currently observable BMD state."""
        if self.observable_face == MembraneFace.FRONT:
            return self.front_bmd
        else:
            return self.back_bmd
    
    def get_hidden_bmd(self) -> BMDState:
        """Get currently hidden BMD state."""
        if self.observable_face == MembraneFace.FRONT:
            return self.back_bmd
        else:
            return self.front_bmd
    
    def switch_observable_face(self):
        """Switch which face is observable."""
        if self.observable_face == MembraneFace.FRONT:
            self.observable_face = MembraneFace.BACK
        else:
            self.observable_face = MembraneFace.FRONT
    
    def membrane_thickness(self) -> float:
        """
        Calculate categorical membrane thickness (depth).
        
        Returns:
            Categorical distance between front and back faces
        """
        # Extract S_k values from BMD metadata
        S_k_front = self.front_bmd.metadata.get('S_k', 0.0)
        S_k_back = self.back_bmd.metadata.get('S_k', 0.0)
        
        # For phase conjugate: thickness = 2|S_k|
        thickness = abs(S_k_front - S_k_back)
        
        return thickness
    
    def categorical_richness_dual(self) -> float:
        """
        Total categorical richness across both faces.
        
        R_dual = R_front + R_back
        """
        R_front = self.front_bmd.categorical_richness()
        R_back = self.back_bmd.categorical_richness()
        
        return R_front + R_back
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'front_bmd': self.front_bmd.to_dict(),
            'back_bmd': self.back_bmd.to_dict(),
            'observable_face': self.observable_face.value,
            'transform_type': self.transform.transform_type,
            'membrane_thickness': self.membrane_thickness(),
            'dual_richness': self.categorical_richness_dual()
        }


def pixel_demon_to_bmd_state(
    pixel_demon: DualMembranePixelDemon,
    use_cascade: bool = True,
    cascade_depth: int = 10
) -> DualMembraneBMDState:
    """
    Convert dual-membrane pixel demon to dual BMD state.
    
    Args:
        pixel_demon: Dual-membrane pixel Maxwell demon
        use_cascade: Whether to use reflectance cascade for richness enhancement
        cascade_depth: Number of cascade levels (if use_cascade=True)
    
    Returns:
        DualMembraneBMDState compatible with HCCC framework
    """
    # Extract front and back S-entropy coordinates
    s_front = pixel_demon.dual_state.front_s
    s_back = pixel_demon.dual_state.back_s
    
    # Create front face BMD state
    front_bmd = _create_bmd_from_s_coordinates(
        s_front,
        pixel_demon.molecular_demons,
        face='front',
        use_cascade=use_cascade,
        cascade_depth=cascade_depth
    )
    
    # Create back face BMD state (conjugate)
    back_bmd = _create_bmd_from_s_coordinates(
        s_back,
        pixel_demon.molecular_demons,
        face='back',
        use_cascade=use_cascade,
        cascade_depth=cascade_depth
    )
    
    # Create dual-membrane BMD state
    dual_bmd = DualMembraneBMDState(
        front_bmd=front_bmd,
        back_bmd=back_bmd,
        observable_face=pixel_demon.observable_face,
        transform=pixel_demon.transform,
        source_pixel_demon=pixel_demon
    )
    
    return dual_bmd


def _create_bmd_from_s_coordinates(
    s_coords: SEntropyCoordinates,
    molecular_demons: Dict,
    face: str = 'front',
    use_cascade: bool = True,
    cascade_depth: int = 10
) -> BMDState:
    """
    Create HCCC BMDState from pixel demon S-coordinates.
    
    Args:
        s_coords: S-entropy coordinates (S_k, S_t, S_e)
        molecular_demons: Molecular demon lattice
        face: 'front' or 'back'
        use_cascade: Use reflectance cascade
        cascade_depth: Cascade depth
    
    Returns:
        BMDState compatible with HCCC
    """
    # Create oscillatory holes from molecular demon lattice
    oscillatory_holes = []
    
    for mol_type, demon in molecular_demons.items():
        # Each vibrational mode creates an oscillatory hole
        for freq in demon.vibrational_modes:
            # Number of configurations from molecular density and categorical states
            # O₂ has 25,110 categorical states; approximate for other molecules
            n_configs = _estimate_configurations(mol_type, demon.number_density)
            
            hole = OscillatoryHole(
                required_frequency=freq,
                required_phase=demon.s_state.S_t * 2 * np.pi,  # S_t maps to phase
                required_amplitude=1.0,  # Normalized
                n_configurations=n_configs
            )
            oscillatory_holes.append(hole)
    
    # Create phase structure from molecular demon phases
    phases = []
    frequencies = []
    
    for demon in molecular_demons.values():
        phases.append(demon.s_state.S_t * 2 * np.pi)
        if demon.vibrational_modes:
            frequencies.append(demon.vibrational_modes[0])  # Fundamental frequency
    
    phases = np.array(phases)
    frequencies = np.array(frequencies)
    
    # Compute coherence matrix from phase differences
    n_demons = len(phases)
    coherence = np.zeros((n_demons, n_demons))
    
    for i in range(n_demons):
        for j in range(n_demons):
            # Coherence from phase difference
            phase_diff = abs(phases[i] - phases[j])
            coherence[i, j] = np.cos(phase_diff)
    
    phase_structure = PhaseStructure(
        phases=phases,
        frequencies=frequencies,
        coherence=coherence
    )
    
    # Categorical state as S-coordinates
    categorical_state = {
        'S_k': s_coords.S_k,
        'S_t': s_coords.S_t,
        'S_e': s_coords.S_e,
        'face': face
    }
    
    # Create BMD state
    bmd = BMDState(
        categorical_state=categorical_state,
        oscillatory_holes=oscillatory_holes,
        phase_structure=phase_structure,
        metadata={
            'source': f'pixel_demon_{face}',
            'S_k': s_coords.S_k,
            'S_t': s_coords.S_t,
            'S_e': s_coords.S_e,
            'cascade_enabled': use_cascade,
            'cascade_depth': cascade_depth if use_cascade else 0
        }
    )
    
    # Apply reflectance cascade if enabled
    if use_cascade and cascade_depth > 0:
        bmd = _apply_cascade_enhancement(bmd, cascade_depth)
    
    return bmd


def _estimate_configurations(molecule_type: str, number_density: float) -> int:
    """
    Estimate number of categorical configurations for molecule type.
    
    Based on electronic structure and collision network complexity.
    
    Args:
        molecule_type: 'O2', 'N2', 'H2O', etc.
        number_density: Molecules per m³
    
    Returns:
        Estimated configuration count
    """
    # Base categorical states from electronic structure
    base_configs = {
        'O2': 25110,    # Paramagnetic, 4 unpaired electrons
        'N2': 1,        # Diamagnetic, closed shell
        'H2O': 3,       # Bent molecule, 2 vibrational modes
        'CO2': 4,       # Linear, 4 vibrational modes
        'Ar': 1,        # Noble gas, no structure
        'CO': 1326,     # Weak paramagnetic
    }
    
    base = base_configs.get(molecule_type, 1)
    
    # Collision network adds configurations proportional to √(density)
    # (number of collision partners scales as √N)
    collision_factor = max(1, int(np.sqrt(number_density / 1e24)))
    
    total_configs = base * collision_factor
    
    return int(total_configs)


def _apply_cascade_enhancement(bmd: BMDState, cascade_depth: int) -> BMDState:
    """
    Apply reflectance cascade to enhance categorical frequency resolution.
    
    This implements trans-Planckian precision through frequency-domain enhancement,
    not time-domain measurement. The cascade provides:
    
    - Zero-time measurement (t_meas = 0) via categorical simultaneity
    - Effective frequency resolution: f_effective = f_base × F_cascade
    - Dimensional conversion: δt = 1/(2π f_effective)
    
    Enhancement scaling: F_cascade = N_ref^β with β ≈ 2.1 (super-quadratic)
    Information accumulation: I_N = N(N+1)(2N+1)/6 ≈ N³/3
    
    Args:
        bmd: Base BMD state
        cascade_depth: Number of cascade reflections (N_ref)
    
    Returns:
        Enhanced BMD state with cascade-amplified frequency resolution
    """
    # Calculate cascade enhancement factor
    # Quadratic to super-quadratic scaling: F_cascade ≈ N_ref^2.1
    N = cascade_depth
    
    # Measured β = 2.10 ± 0.05 from cascade reflection scaling study
    beta = 2.10
    cascade_enhancement_factor = int(N ** beta)
    
    # Total information factor (for richness calculation)
    # I_N = sum_{k=1}^{N} (k+1)^2 = N(N+1)(2N+1)/6
    total_info_factor = N * (N + 1) * (2 * N + 1) // 6
    
    # Enhance each oscillatory hole's configuration count
    enhanced_holes = []
    for hole in bmd.holes:
        # Effective frequency from cascade
        # f_effective = f_base × cascade_enhancement_factor
        effective_frequency = hole.required_frequency * cascade_enhancement_factor
        
        enhanced_hole = OscillatoryHole(
            required_frequency=effective_frequency,
            required_phase=hole.required_phase,
            required_amplitude=hole.required_amplitude,
            n_configurations=hole.n_configurations * total_info_factor
        )
        enhanced_holes.append(enhanced_hole)
    
    # Calculate equivalent temporal precision (dimensional conversion)
    # δt = 1/(2π f_effective)
    # This is frequency resolution, NOT chronological time measurement
    if enhanced_holes:
        max_frequency = max(h.required_frequency for h in enhanced_holes)
        equivalent_temporal_precision = 1.0 / (2.0 * np.pi * max_frequency)
    else:
        equivalent_temporal_precision = 0.0
    
    # Create enhanced BMD
    enhanced_bmd = BMDState(
        categorical_state=bmd.c_current,
        oscillatory_holes=enhanced_holes,
        phase_structure=bmd.phase,
        metadata={
            **bmd.metadata,
            'cascade_enhancement_factor': cascade_enhancement_factor,
            'cascade_depth': N,
            'cascade_beta': beta,
            'total_info_factor': total_info_factor,
            'base_richness': bmd.categorical_richness(),
            'enhanced_richness': sum(h.n_configurations for h in enhanced_holes),
            'effective_frequency_Hz': max_frequency if enhanced_holes else 0.0,
            'equivalent_temporal_precision_s': equivalent_temporal_precision,
            'measurement_time_s': 0.0,  # Zero-time categorical measurement
            'trans_planckian': equivalent_temporal_precision < 5.39e-44  # Below Planck time
        }
    )
    
    return enhanced_bmd

