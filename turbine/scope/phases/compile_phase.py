"""
Phase 1: COMPILE

Accumulate timing events from acquisition into trajectories.
Classify trajectories into timing cells.
Transition: Timing entropy S_t decreases, Knowledge entropy S_k increases.
"""

from dataclasses import dataclass
from typing import Optional, List
import logging

from ..types.timing_cell import Trajectory, TimingDeviation, DispatchTable
from ..types.partition_state import PartitionState, Spin, SEntropy


logger = logging.getLogger(__name__)


@dataclass
class CompilePhaseInput:
    """
    Input to COMPILE phase.

    Attributes:
        timing_events: Stream of timing deviations from acquisition
        dispatch_table: Dispatch rules for cell classification
        depth: Expected depth (number of events to accumulate)
    """
    timing_events: List[TimingDeviation]
    dispatch_table: DispatchTable
    depth: int


@dataclass
class CompilePhaseOutput:
    """
    Output from COMPILE phase.

    Attributes:
        trajectory: Completed trajectory
        cell_id: Assigned timing cell ID
        partition_state: Initial partition state (n, ℓ, m, s)
        s_entropy_before: Entropy before compilation
        s_entropy_after: Entropy after compilation
    """
    trajectory: Trajectory
    cell_id: Optional[str]
    partition_state: PartitionState
    s_entropy_before: SEntropy
    s_entropy_after: SEntropy

    def to_dict(self) -> dict:
        return {
            'trajectory': self.trajectory.to_dict(),
            'cell_id': self.cell_id,
            'partition_state': self.partition_state.to_dict(),
            's_entropy_before': self.s_entropy_before.to_dict(),
            's_entropy_after': self.s_entropy_after.to_dict(),
        }


def compile_phase(inputs: CompilePhaseInput) -> CompilePhaseOutput:
    """
    Execute COMPILE phase: accumulate and classify timing events.

    Algorithm:
    1. Accumulate timing_events into a trajectory of length depth
    2. Find matching timing cell via dispatch_table.classify()
    3. Assign partition state (n, ℓ, m, s) based on cell
    4. Track entropy transition S_t → S_k

    Args:
        inputs: CompilePhaseInput with timing events, dispatch table, depth

    Returns:
        CompilePhaseOutput with classified trajectory and partition state
    """
    # Input validation
    if len(inputs.timing_events) < inputs.depth:
        logger.warning(
            f"Only {len(inputs.timing_events)} events available, "
            f"expected {inputs.depth}. Using what we have."
        )
        depth = len(inputs.timing_events)
    else:
        depth = inputs.depth

    # Step 1: Accumulate trajectory
    trajectory = Trajectory()
    for i in range(depth):
        trajectory.add_event(inputs.timing_events[i])

    logger.info(f"COMPILE: Accumulated trajectory of length {len(trajectory)}")

    # Step 2: Classify into timing cell
    dispatch_rule = inputs.dispatch_table.dispatch(trajectory)
    cell_id = dispatch_rule.cell_id if dispatch_rule else None

    # Step 3: Assign partition state
    # From Temporal Programming interpretation:
    #   n = depth (log2 of accumulated events)
    #   ℓ = dominant channel
    #   m = trajectory ID or index
    #   s = sign of mean timing deviation
    n = int(round(__import__('math').log2(max(depth, 1))))
    ℓ = trajectory.dominant_channel()
    m = cell_id.hash() % 256 if cell_id else 0  # Hash cell ID to integer

    mean_delta_p = trajectory.mean_delta_p()
    s = Spin.POS_HALF if mean_delta_p >= 0 else Spin.NEG_HALF

    partition_state = PartitionState(n=n, ℓ=ℓ, m=m, s=s)

    # Step 4: Track entropy transition
    # Initial: all uncertainty in timing (S_t = 1.0)
    s_entropy_before = SEntropy.initial()

    # After classification: partition state is fixed, so S_t decreases to 0
    # Knowledge increases proportionally
    # By conservation: S_k + S_t + S_e = 1
    # Assume no backaction yet (E still at 0), so S_k = 1 - S_t_after
    s_entropy_after = SEntropy(
        S_k=1.0 - 1e-6,  # Nearly complete knowledge
        S_t=1e-6,  # Nearly zero timing uncertainty
        S_e=0.0  # No backaction yet
    )

    logger.info(
        f"COMPILE: Classified as cell_id={cell_id}, "
        f"partition_state={partition_state}, "
        f"entropy: {s_entropy_before} → {s_entropy_after}"
    )

    return CompilePhaseOutput(
        trajectory=trajectory,
        cell_id=cell_id,
        partition_state=partition_state,
        s_entropy_before=s_entropy_before,
        s_entropy_after=s_entropy_after,
    )


def compile_phase_batch(
    timing_events_list: List[List[TimingDeviation]],
    dispatch_table: DispatchTable,
    depth: int
) -> List[CompilePhaseOutput]:
    """
    Execute COMPILE phase on a batch of event streams.

    Args:
        timing_events_list: List of event streams
        dispatch_table: Shared dispatch table
        depth: Accumulation depth for each trajectory

    Returns:
        List of CompilePhaseOutput results
    """
    results = []
    for events in timing_events_list:
        inputs = CompilePhaseInput(
            timing_events=events,
            dispatch_table=dispatch_table,
            depth=depth
        )
        result = compile_phase(inputs)
        results.append(result)
    return results
