"""
Phase 4: EXECUTE

Run partition morphism chain with coordinate field grounding.
Entropy: S_k increases (morphism narrows partition), S_e increases (measurement backaction).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import logging

from ..types.partition_state import PartitionState, Spin, SEntropy
from ..types.coord_field import CoordField


logger = logging.getLogger(__name__)


@dataclass
class Catalyst:
    """
    Morphism catalyst: constraint that reduces categorical distance.

    Attributes:
        name: Catalyst name (e.g., "conservation(mass)", "phase_lock(structure)")
        epsilon: Cost factor (0 to 1), added to categorical distance
    """
    name: str
    epsilon: float = 0.01

    def __str__(self) -> str:
        return f"Catalyst({self.name}, ε={self.epsilon:.3f})"


@dataclass
class MorphismStep:
    """
    Single step in a morphism chain.

    Attributes:
        step_type: "observe" | "catalyze" | "fuse" | "measure_distance" | "access"
        params: Step-specific parameters
    """
    step_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Step({self.step_type}, {self.params})"


@dataclass
class MorphismChain:
    """
    Sequence of morphism operations forming a computational pipeline.

    Attributes:
        chain_id: Unique identifier
        steps: List of morphism steps
        catalysts: Applied catalysts with accumulated epsilon
    """
    chain_id: str
    steps: List[MorphismStep] = field(default_factory=list)
    catalysts: List[Catalyst] = field(default_factory=list)

    def add_step(self, step: MorphismStep) -> None:
        self.steps.append(step)

    def add_catalyst(self, catalyst: Catalyst) -> None:
        self.catalysts.append(catalyst)

    def total_epsilon(self) -> float:
        """Total cost from all catalysts"""
        return sum(c.epsilon for c in self.catalysts)

    def __str__(self) -> str:
        return f"MorphismChain({self.chain_id}, {len(self.steps)} steps, ε_total={self.total_epsilon():.3f})"


@dataclass
class ExecutePhaseInput:
    """
    Input to EXECUTE phase.

    Attributes:
        morphism_chain: The morphism chain to execute
        initial_partition_state: Starting partition state (from COMPILE)
        frame: Current image frame
        coord_field: Coordinate field from MEASURE phase
    """
    morphism_chain: MorphismChain
    initial_partition_state: PartitionState
    frame: Any  # np.ndarray or placeholder
    coord_field: CoordField


@dataclass
class MeasurementResult:
    """
    Result of a measurement operation.

    Attributes:
        distance: Measured distance in meters
        uncertainty: Formal uncertainty bound
        targets: Locations of measured targets
    """
    distance: float
    uncertainty: float
    targets: List[Tuple[float, float]]  # [(u1, v1), (u2, v2), ...]


@dataclass
class ExecutePhaseOutput:
    """
    Output from EXECUTE phase.

    Attributes:
        final_partition_state: Refined partition state
        measurement: Measurement result (if any measure_distance was called)
        structure_id: Identified structure (from access operation)
        categorical_distance_traveled: Total distance in morphism space
        s_entropy_before: Entropy before execution
        s_entropy_after: Entropy after execution
    """
    final_partition_state: PartitionState
    measurement: Optional[MeasurementResult] = None
    structure_id: Optional[str] = None
    categorical_distance_traveled: float = 0.0
    s_entropy_before: SEntropy = field(default_factory=lambda: SEntropy(S_k=1e-6, S_t=1e-6, S_e=0.998))
    s_entropy_after: SEntropy = field(default_factory=lambda: SEntropy(S_k=0.5, S_t=1e-6, S_e=0.5))

    def to_dict(self) -> dict:
        return {
            'final_partition_state': self.final_partition_state.to_dict(),
            'measurement': {
                'distance': self.measurement.distance,
                'uncertainty': self.measurement.uncertainty,
                'targets': self.measurement.targets,
            } if self.measurement else None,
            'structure_id': self.structure_id,
            'categorical_distance_traveled': self.categorical_distance_traveled,
            's_entropy_before': self.s_entropy_before.to_dict(),
            's_entropy_after': self.s_entropy_after.to_dict(),
        }


def _execute_observe_step(
    state: PartitionState,
    frame: Any,
    depth: int
) -> PartitionState:
    """
    Execute 'observe' step: create initial partition state at given depth.

    Args:
        state: Current partition state
        frame: Image frame (unused but available)
        depth: Target depth n

    Returns:
        Updated partition state with depth set to n
    """
    # observe(frame, n=depth) sets the partition to (n, 0, 0, 0)
    new_state = PartitionState(n=depth, ℓ=0, m=0, s=Spin.POS_HALF)
    logger.debug(f"OBSERVE: {state} → {new_state}")
    return new_state


def _execute_catalyze_step(
    state: PartitionState,
    catalyst: Catalyst,
    total_epsilon: float
) -> Tuple[PartitionState, float]:
    """
    Execute 'catalyze' step: apply constraint to reduce categorical distance.

    Catalysts narrow the solution space but increase cost.

    Args:
        state: Current partition state
        catalyst: Catalyst to apply
        total_epsilon: Accumulated epsilon from all catalysts

    Returns:
        (updated_state, new_total_epsilon)
    """
    # Catalyze operation: increment mode ℓ to reflect constraint application
    new_ℓ = min(state.ℓ + 1, 255)  # Cap at 255
    new_state = PartitionState(n=state.n, ℓ=new_ℓ, m=state.m, s=state.s)
    new_epsilon = total_epsilon + catalyst.epsilon

    logger.debug(
        f"CATALYZE({catalyst.name}): {state} → {new_state}, "
        f"ε_total: {total_epsilon:.3f} → {new_epsilon:.3f}"
    )

    return new_state, new_epsilon


def _execute_access_step(
    state: PartitionState,
    target: str
) -> PartitionState:
    """
    Execute 'access' step: traverse partition to specific sub-structure.

    Increments ℓ and m to target sub-structure, narrowing partition.

    Args:
        state: Current partition state
        target: Target structure name

    Returns:
        Updated partition state with narrowed ℓ, m
    """
    # Access operation: increment both ℓ and m to reach target
    new_ℓ = min(state.ℓ + 1, 255)
    new_m = abs(hash(target)) % 256  # Hash target to index

    new_state = PartitionState(n=state.n, ℓ=new_ℓ, m=new_m, s=state.s)

    logger.debug(f"ACCESS({target}): {state} → {new_state}")

    return new_state


def _execute_measure_distance_step(
    state: PartitionState,
    target1_name: str,
    target2_name: str,
    coord_field: CoordField,
    frame: Any
) -> MeasurementResult:
    """
    Execute 'measure_distance' step: compute world-space distance.

    Uses coordinate field Φ to map pixel locations to world-space.

    Args:
        state: Current partition state
        target1_name: Name of first target
        target2_name: Name of second target
        coord_field: Coordinate field from MEASURE phase
        frame: Current frame (for locating targets)

    Returns:
        MeasurementResult with distance and uncertainty
    """
    # Mock localization: hash target names to pixel coordinates
    # In practice, this would use segmentation or other methods
    h, w = coord_field.scale_field.shape

    u1 = (abs(hash(target1_name)) % w)
    v1 = (abs(hash(target1_name)) // w) % h

    u2 = (abs(hash(target2_name)) % w)
    v2 = (abs(hash(target2_name)) // w) % h

    # Compute world-space distance
    distance = coord_field.distance(u1, v1, u2, v2)
    uncertainty = coord_field.uncertainty_at(u1, v1) + coord_field.uncertainty_at(u2, v2)

    logger.info(
        f"MEASURE_DISTANCE({target1_name}, {target2_name}): "
        f"distance={distance:.3e}m ± {uncertainty:.3e}m"
    )

    return MeasurementResult(
        distance=distance,
        uncertainty=uncertainty,
        targets=[(u1, v1), (u2, v2)]
    )


def execute_phase(inputs: ExecutePhaseInput) -> ExecutePhaseOutput:
    """
    Execute EXECUTE phase: run morphism chain with coordinate grounding.

    Algorithm:
    1. Start with initial partition state
    2. For each step in morphism chain:
        - observe: set depth
        - catalyze: apply constraint, increment ε
        - measure_distance: compute world-space distance via Φ
        - access: navigate partition to target
    3. Track categorical distance traveled and entropy changes

    Args:
        inputs: ExecutePhaseInput with morphism chain and coordinate field

    Returns:
        ExecutePhaseOutput with refined state, measurements, entropy
    """
    logger.info(f"EXECUTE phase starting: {inputs.morphism_chain}")

    state = inputs.initial_partition_state
    total_epsilon = 0.0
    measurement = None
    structure_id = None

    for step in inputs.morphism_chain.steps:
        logger.debug(f"  {step}")

        if step.step_type == "observe":
            depth = step.params.get("depth", state.n)
            state = _execute_observe_step(state, inputs.frame, depth)

        elif step.step_type == "catalyze":
            catalyst_name = step.params.get("name", "unknown")
            epsilon = step.params.get("epsilon", 0.01)
            catalyst = Catalyst(name=catalyst_name, epsilon=epsilon)
            state, total_epsilon = _execute_catalyze_step(state, catalyst, total_epsilon)

        elif step.step_type == "access":
            target = step.params.get("target", "structure")
            state = _execute_access_step(state, target)
            structure_id = target

        elif step.step_type == "measure_distance":
            target1 = step.params.get("target1", "point_a")
            target2 = step.params.get("target2", "point_b")
            measurement = _execute_measure_distance_step(
                state, target1, target2, inputs.coord_field, inputs.frame
            )

    # Compute entropy changes
    # From Theorem 3: S_k increases (narrowed partition), S_e increases (backaction)
    s_entropy_before = SEntropy(S_k=1e-6, S_t=1e-6, S_e=0.998)

    # After execution: more knowledge (S_k up), but measurement backaction (S_e still high)
    s_k_after = min(0.5, 1.0 - 0.2 * len(inputs.morphism_chain.steps))
    s_e_after = max(0.0, 1.0 - s_k_after - 1e-6)

    s_entropy_after = SEntropy(S_k=s_k_after, S_t=1e-6, S_e=s_e_after)

    logger.info(
        f"EXECUTE phase complete: {inputs.morphism_chain.chain_id}, "
        f"final_state={state}, entropy: {s_entropy_before} → {s_entropy_after}"
    )

    return ExecutePhaseOutput(
        final_partition_state=state,
        measurement=measurement,
        structure_id=structure_id,
        categorical_distance_traveled=total_epsilon,
        s_entropy_before=s_entropy_before,
        s_entropy_after=s_entropy_after,
    )
