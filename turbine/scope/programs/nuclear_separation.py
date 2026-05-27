"""
Example SCOPE Program: Nuclear Separation Dynamics

Measures the distance between two nuclei (DAPI-stained) across a time-lapse
sequence. Dispatches different morphism chains based on detected cell cycle phase
(PROPHASE, METAPHASE, ANAPHASE) inferred from timing deviations.

Example program from SCOPE paper demonstrating the full integration of:
- Temporal Programming (timing cell dispatch)
- Partition Calculus (morphism chains with catalysts)
- Context-Dependent Coordinates (world-space distance measurement)
"""

import numpy as np
from typing import List

from ..runtime.scope_runtime import SCOPEProgram, SCOPERuntime
from ..types.timing_cell import TimingCell, TimingDeviation, CellPartition, DispatchTable, DispatchRule
from ..types.partition_state import Spin
from ..phases.execute_phase import MorphismChain, MorphismStep, Catalyst


def create_nuclear_separation_program() -> SCOPEProgram:
    """
    Create the nuclear_separation_dynamics SCOPE program.

    Returns:
        SCOPEProgram configured for nuclear separation measurement
    """
    # Program specification
    program = SCOPEProgram(
        name="nuclear_separation_dynamics",
        depth=1000,  # 2^10 = 1024 ~= field_size / resolution
        field_size_x=100.0,  # micrometers
        field_size_y=100.0,
        resolution=0.1,  # micrometers/pixel
        lambda_s=0.10,  # Spatial coherence
        lambda_t=0.05,  # Temporal coherence
    )

    # === TIMING CELLS ===
    # Classify timing deviations into cell cycle phases
    cell_prophase = TimingCell(
        cell_id="PROPHASE",
        bounds_delta_p=(-2.0e-6, -0.8e-6),
        depth=1000,
    )

    cell_metaphase = TimingCell(
        cell_id="METAPHASE",
        bounds_delta_p=(-0.8e-6, 0.8e-6),
        depth=1000,
    )

    cell_anaphase = TimingCell(
        cell_id="ANAPHASE",
        bounds_delta_p=(0.8e-6, 2.0e-6),
        depth=1000,
    )

    # Build dispatch table
    program.dispatch_table.add_cell(cell_prophase)
    program.dispatch_table.add_cell(cell_metaphase)
    program.dispatch_table.add_cell(cell_anaphase)

    # === MORPHISM CHAINS ===

    # Morphism 1: nucleus_pair_measurement
    # Measures the distance between two nuclei
    nucleus_pair = MorphismChain(chain_id="nucleus_pair_measurement")
    nucleus_pair.add_step(MorphismStep(
        step_type="observe",
        params={"depth": 1000}
    ))
    nucleus_pair.add_step(MorphismStep(
        step_type="catalyze",
        params={"name": "conservation(dna_mass)", "epsilon": 0.008}
    ))
    nucleus_pair.add_step(MorphismStep(
        step_type="catalyze",
        params={"name": "phase_lock(chromatin)", "epsilon": 0.005}
    ))
    nucleus_pair.add_step(MorphismStep(
        step_type="measure_distance",
        params={"target1": "nucleus_a", "target2": "nucleus_b"}
    ))
    nucleus_pair.add_step(MorphismStep(
        step_type="access",
        params={"target": "separation_vector"}
    ))

    # Morphism 2: membrane_boundary
    # Extracts the cell membrane boundary
    membrane_boundary = MorphismChain(chain_id="membrane_boundary")
    membrane_boundary.add_step(MorphismStep(
        step_type="observe",
        params={"depth": 1000}
    ))
    membrane_boundary.add_step(MorphismStep(
        step_type="catalyze",
        params={"name": "phase_lock(plasma_membrane)", "epsilon": 0.010}
    ))
    membrane_boundary.add_step(MorphismStep(
        step_type="access",
        params={"target": "partition_boundary"}
    ))

    # Add morphisms to program
    program.add_morphism(nucleus_pair)
    program.add_morphism(membrane_boundary)

    # === DISPATCH RULES ===
    # Map timing cells to morphism actions
    program.dispatch_table.add_rule(DispatchRule(
        cell_id="PROPHASE",
        action="nucleus_pair_measurement",
        action_type="morphism"
    ))
    program.dispatch_table.add_rule(DispatchRule(
        cell_id="METAPHASE",
        action="membrane_boundary",
        action_type="morphism"
    ))
    program.dispatch_table.add_rule(DispatchRule(
        cell_id="ANAPHASE",
        action="nucleus_pair_measurement",
        action_type="morphism"
    ))

    return program


def generate_synthetic_timing_events(
    phase: str,
    num_events: int = 1000,
    num_channels: int = 2
) -> List[TimingDeviation]:
    """
    Generate synthetic timing events for a given cell cycle phase.

    Args:
        phase: "PROPHASE", "METAPHASE", or "ANAPHASE"
        num_events: Number of events to generate
        num_channels: Number of acquisition channels

    Returns:
        List of TimingDeviation objects
    """
    events = []

    # Mean timing deviation for each phase
    phase_means = {
        "PROPHASE": -1.4e-6,   # Early timing
        "METAPHASE": 0.0e-6,   # Aligned
        "ANAPHASE": 1.4e-6,    # Late timing
    }

    mean_delta_p = phase_means.get(phase, 0.0)
    sigma = 0.3e-6  # Standard deviation

    # Generate events with noise
    for i in range(num_events):
        # Add Gaussian noise
        delta_p = np.random.normal(mean_delta_p, sigma)

        # Alternate between channels
        channel = i % num_channels

        # Optionally add intensity
        intensity = np.random.exponential(scale=100)

        event = TimingDeviation(
            delta_p=delta_p,
            channel_id=channel,
            intensity=intensity
        )
        events.append(event)

    return events


def generate_synthetic_frame(
    shape: tuple = (1024, 1024),
    num_nuclei: int = 2,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a synthetic fluorescence microscopy image (DAPI channel).

    Args:
        shape: Image shape (height, width)
        num_nuclei: Number of nuclear regions
        seed: Random seed for reproducibility

    Returns:
        Synthetic image array
    """
    np.random.seed(seed)

    h, w = shape
    frame = np.zeros((h, w), dtype=np.float32)

    # Add nuclear regions (Gaussian blobs)
    for _ in range(num_nuclei):
        # Random center
        cy = np.random.randint(100, h - 100)
        cx = np.random.randint(100, w - 100)

        # Create Gaussian blob
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Gaussian blob with random size
        sigma = np.random.uniform(20, 50)
        blob = 1000 * np.exp(-(dist**2) / (2 * sigma**2))
        frame += blob

    # Add background noise
    frame += np.random.normal(50, 10, frame.shape)
    frame = np.clip(frame, 0, 2000)

    return frame


def run_example():
    """
    Run the nuclear separation dynamics example.

    Demonstrates the full SCOPE pipeline:
    1. Create program specification
    2. Initialize runtime
    3. Generate synthetic data
    4. Execute for multiple phases
    """
    print("=" * 70)
    print("SCOPE Example: Nuclear Separation Dynamics")
    print("=" * 70)

    # Step 1: Create program
    print("\n1. Creating SCOPE program...")
    program = create_nuclear_separation_program()
    print(f"   {program}")
    print(f"   Dispatch table: {len(program.dispatch_table.cell_partition.cells)} cells")
    print(f"   Morphisms: {len(program.morphisms)}")

    # Step 2: Create runtime
    print("\n2. Initializing runtime...")
    runtime = SCOPERuntime(program)

    # Step 3: Generate synthetic data
    print("\n3. Generating synthetic data...")
    frame = generate_synthetic_frame(shape=(1024, 1024), num_nuclei=2, seed=42)
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame range: [{frame.min():.1f}, {frame.max():.1f}]")

    # Step 4: Execute for each phase
    phases = [("PROPHASE", -1.4e-6, 0.8e-6),
              ("METAPHASE", 0.0e-6, 0.5e-6),
              ("ANAPHASE", 1.4e-6, 0.8e-6)]

    for phase_name, phase_mean, phase_std in phases:
        print(f"\n4. Executing for {phase_name}...")
        print(f"   Mean ΔP: {phase_mean:.2e}s, Std: {phase_std:.2e}s")

        # Generate timing events for this phase
        timing_events = generate_synthetic_timing_events(
            phase=phase_name,
            num_events=1000,
            num_channels=2
        )

        # Run SCOPE
        try:
            result = runtime.run(timing_events, frame)
            print(f"   ✓ Success!")
            print(f"     Structure: {result.structure}")
            if result.distance is not None:
                print(f"     Distance: {result.distance:.3e}m ± {result.uncertainty:.3e}m")
            print(f"     Position: ({result.position[0]:.3e}, {result.position[1]:.3e}, {result.position[2]:.3e})")
            print(f"     S-entropy: S_k={result.s_entropy.S_k:.3f}, S_t={result.s_entropy.S_t:.3f}, S_e={result.s_entropy.S_e:.3f}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
