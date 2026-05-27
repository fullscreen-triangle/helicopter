"""
SCOPE: Spectral Coordinate Observation with Partition Execution

A unified metalanguage for microscopy image analysis integrating:
- Partition Calculus (image structure)
- Context-Dependent Coordinates (pixel-to-world mapping)
- Temporal Programming (acquisition-driven execution)
"""

from .runtime.scope_runtime import SCOPERuntime, SCOPEProgram, create_runtime
from .types.partition_state import PartitionState, SEntropy, Spin
from .types.coord_field import CoordField, ScaleFieldEstimate
from .types.timing_cell import TimingCell, CellPartition, DispatchTable, Trajectory, TimingDeviation
from .phases.compile_phase import compile_phase, CompilePhaseInput, CompilePhaseOutput
from .phases.measure_phase import measure_phase, MeasurePhaseInput, MeasurePhaseOutput
from .phases.execute_phase import execute_phase, ExecutePhaseInput, ExecutePhaseOutput, MorphismChain, MorphismStep
from .phases.emit_phase import emit_phase, EmitPhaseInput, SCOPEResult

__all__ = [
    # Runtime
    "SCOPERuntime",
    "SCOPEProgram",
    "create_runtime",
    # Types
    "PartitionState",
    "SEntropy",
    "Spin",
    "CoordField",
    "ScaleFieldEstimate",
    "TimingCell",
    "CellPartition",
    "DispatchTable",
    "Trajectory",
    "TimingDeviation",
    # Phases
    "compile_phase",
    "measure_phase",
    "execute_phase",
    "emit_phase",
    "CompilePhaseInput",
    "CompilePhaseOutput",
    "MeasurePhaseInput",
    "MeasurePhaseOutput",
    "ExecutePhaseInput",
    "ExecutePhaseOutput",
    "EmitPhaseInput",
    "SCOPEResult",
    "MorphismChain",
    "MorphismStep",
]

__version__ = "0.1.0"
__author__ = "SCOPE Development Team"
