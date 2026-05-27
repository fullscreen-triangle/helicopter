"""
Phase 5: EMIT

Assemble final measurement record with world-space grounding.
Output: Result with distance, position, uncertainty, and S-entropy.
Entropy: No change (final assembly).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

from ..types.partition_state import PartitionState, SEntropy
from .execute_phase import MeasurementResult


logger = logging.getLogger(__name__)


@dataclass
class EmitPhaseInput:
    """
    Input to EMIT phase.

    Attributes:
        final_partition_state: Refined partition state from EXECUTE
        measurement: Measurement result (if any)
        structure_id: Structure identifier
        s_entropy_before: Entropy from EXECUTE phase
        s_entropy_after: Entropy from EXECUTE phase
    """
    final_partition_state: PartitionState
    measurement: Optional[MeasurementResult]
    structure_id: Optional[str]
    s_entropy_before: SEntropy
    s_entropy_after: SEntropy


@dataclass
class SCOPEResult:
    """
    Final SCOPE program result: world-space-grounded measurement.

    Attributes:
        structure: Structure identifier
        position: Position in world-space (x, y, z) in meters
        distance: Measured distance in meters (if applicable)
        uncertainty: Formal uncertainty bound in meters
        s_entropy: (S_k, S_t, S_e) triplet
        partition_state: Final partition state
    """
    structure: Optional[str]
    position: tuple  # (x, y, z) in meters
    distance: Optional[float]
    uncertainty: float
    s_entropy: SEntropy
    partition_state: PartitionState

    def __str__(self) -> str:
        pos_str = f"({self.position[0]:.3e}, {self.position[1]:.3e}, {self.position[2]:.3e})"
        if self.distance is not None:
            return (
                f"Result(structure={self.structure}, "
                f"position={pos_str}, "
                f"distance={self.distance:.3e}±{self.uncertainty:.3e}m, "
                f"S_entropy={self.s_entropy})"
            )
        else:
            return (
                f"Result(structure={self.structure}, "
                f"position={pos_str}, "
                f"uncertainty={self.uncertainty:.3e}m, "
                f"S_entropy={self.s_entropy})"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage/transmission"""
        return {
            'structure': self.structure,
            'position': {
                'x': self.position[0],
                'y': self.position[1],
                'z': self.position[2],
            },
            'distance': self.distance,
            'uncertainty': self.uncertainty,
            's_entropy': self.s_entropy.to_dict(),
            'partition_state': self.partition_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SCOPEResult':
        """Reconstruct from dictionary"""
        pos = d['position']
        position = (pos['x'], pos['y'], pos['z'])

        s_entropy_dict = d['s_entropy']
        s_entropy = SEntropy(
            S_k=s_entropy_dict['S_k'],
            S_t=s_entropy_dict['S_t'],
            S_e=s_entropy_dict['S_e'],
        )

        ps_dict = d['partition_state']
        from ..types.partition_state import Spin
        s_spin = Spin.POS_HALF if ps_dict['s'] > 0 else Spin.NEG_HALF
        partition_state = PartitionState(
            n=ps_dict['n'],
            ℓ=ps_dict['ℓ'],
            m=ps_dict['m'],
            s=s_spin,
        )

        return cls(
            structure=d['structure'],
            position=position,
            distance=d['distance'],
            uncertainty=d['uncertainty'],
            s_entropy=s_entropy,
            partition_state=partition_state,
        )


def emit_phase(inputs: EmitPhaseInput) -> SCOPEResult:
    """
    Execute EMIT phase: assemble final measurement record.

    Algorithm:
    1. Extract position from partition state
    2. Assemble distance and uncertainty (if measurement present)
    3. Include S-entropy tracking
    4. Create final result record

    Args:
        inputs: EmitPhaseInput with execution results

    Returns:
        SCOPEResult with world-space-grounded measurements
    """
    logger.info("EMIT phase starting")

    state = inputs.final_partition_state

    # Estimate position from partition state
    # In a real system, this would be more sophisticated
    # For now: use partition indices as fractional coordinates
    x = (state.ℓ / 256.0)  # Normalize to [0, 1]
    y = (state.m / 256.0)
    z = (state.s.value / 0.5)  # Spin contributes to z: -1 to +1

    position = (x, y, z)

    # Extract measurement results
    distance = inputs.measurement.distance if inputs.measurement else None
    uncertainty = inputs.measurement.uncertainty if inputs.measurement else 1e-3

    # Create result
    result = SCOPEResult(
        structure=inputs.structure_id,
        position=position,
        distance=distance,
        uncertainty=uncertainty,
        s_entropy=inputs.s_entropy_after,
        partition_state=state,
    )

    # Verify entropy conservation: S_k + S_t + S_e should be 1.0
    total = result.s_entropy.S_k + result.s_entropy.S_t + result.s_entropy.S_e
    if abs(total - 1.0) > 1e-6:
        logger.warning(
            f"EMIT: S-entropy not conserved: {total:.15f} (expected 1.0)"
        )

    logger.info(f"EMIT phase complete: {result}")

    return result


def emit_batch(
    inputs_list: list
) -> list:
    """
    Execute EMIT phase on a batch of execution results.

    Args:
        inputs_list: List of EmitPhaseInput objects

    Returns:
        List of SCOPEResult objects
    """
    results = []
    for inputs in inputs_list:
        result = emit_phase(inputs)
        results.append(result)
    return results
