"""
SCOPE Runtime: Five-phase execution orchestrator.

Coordinates COMPILE → ASSIGN → MEASURE → EXECUTE → EMIT pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import time

from ..types.timing_cell import TimingDeviation, DispatchTable
from ..types.coord_field import CoordField
from ..phases.compile_phase import compile_phase, CompilePhaseInput, CompilePhaseOutput
from ..phases.measure_phase import measure_phase, MeasurePhaseInput, MeasurePhaseOutput
from ..phases.execute_phase import execute_phase, ExecutePhaseInput, ExecutePhaseOutput, MorphismChain
from ..phases.emit_phase import emit_phase, EmitPhaseInput, SCOPEResult


logger = logging.getLogger(__name__)


@dataclass
class SCOPEProgram:
    """
    Specification of a SCOPE program.

    Attributes:
        name: Program name
        depth: Partition depth n (controls all three frameworks)
        field_size_x: Field size in micrometers
        field_size_y: Field size in micrometers
        resolution: Pixel resolution in micrometers
        lambda_s: Spatial coherence wavelength
        lambda_t: Temporal coherence wavelength
        dispatch_table: Timing cell dispatch rules
        morphisms: Dict mapping morphism_id → MorphismChain
    """
    name: str
    depth: int
    field_size_x: float
    field_size_y: float
    resolution: float
    lambda_s: float = 0.10
    lambda_t: float = 0.05
    dispatch_table: DispatchTable = field(default_factory=DispatchTable)
    morphisms: Dict[str, MorphismChain] = field(default_factory=dict)

    def add_morphism(self, morphism: MorphismChain) -> None:
        """Register a morphism chain"""
        self.morphisms[morphism.chain_id] = morphism

    def __str__(self) -> str:
        return (
            f"SCOPEProgram({self.name}, n={self.depth}, "
            f"field={self.field_size_x}x{self.field_size_y}µm, "
            f"{len(self.morphisms)} morphisms)"
        )


@dataclass
class ExecutionTrace:
    """
    Execution trace: record of all phase outputs for a single program run.

    Attributes:
        program: The program that was executed
        compile_output: Output from COMPILE phase
        measure_output: Output from MEASURE phase
        execute_output: Output from EXECUTE phase
        emit_output: Final SCOPE result
        total_time_ms: Total execution time
    """
    program: SCOPEProgram
    compile_output: Optional[CompilePhaseOutput] = None
    measure_output: Optional[MeasurePhaseOutput] = None
    execute_output: Optional[ExecutePhaseOutput] = None
    emit_output: Optional[SCOPEResult] = None
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            'program_name': self.program.name,
            'compile': self.compile_output.to_dict() if self.compile_output else None,
            'measure': self.measure_output.to_dict() if self.measure_output else None,
            'execute': self.execute_output.to_dict() if self.execute_output else None,
            'emit': self.emit_output.to_dict() if self.emit_output else None,
            'total_time_ms': self.total_time_ms,
        }


class SCOPERuntime:
    """
    SCOPE execution runtime: orchestrates the five-phase pipeline.

    Maintains program state and executes complete analysis workflows.
    """

    def __init__(self, program: SCOPEProgram):
        self.program = program
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        timing_events: List[TimingDeviation],
        frame: Any,  # np.ndarray
    ) -> SCOPEResult:
        """
        Execute SCOPE program: run all five phases.

        Args:
            timing_events: Timing deviations from acquisition
            frame: Current image frame (2D array)

        Returns:
            SCOPEResult with world-space measurements
        """
        start_time = time.time()
        self.logger.info(f"SCOPERuntime.run() starting: {self.program}")

        try:
            # Phase 1: COMPILE
            self.logger.info("=== Phase 1: COMPILE ===")
            compile_result = self._phase_compile(timing_events)

            # Phase 2: ASSIGN (implicit in dispatch_table.dispatch)
            self.logger.info("=== Phase 2: ASSIGN ===")
            morphism_id = self._phase_assign(compile_result)
            if morphism_id is None:
                raise RuntimeError("No matching morphism for trajectory")

            # Phase 3: MEASURE
            self.logger.info("=== Phase 3: MEASURE ===")
            measure_result = self._phase_measure(frame)

            # Phase 4: EXECUTE
            self.logger.info("=== Phase 4: EXECUTE ===")
            execute_result = self._phase_execute(morphism_id, compile_result, measure_result)

            # Phase 5: EMIT
            self.logger.info("=== Phase 5: EMIT ===")
            emit_result = self._phase_emit(execute_result, compile_result)

            elapsed_ms = (time.time() - start_time) * 1000.0
            self.logger.info(f"SCOPERuntime.run() complete in {elapsed_ms:.1f}ms")

            return emit_result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            self.logger.error(f"SCOPERuntime.run() failed after {elapsed_ms:.1f}ms: {e}")
            raise

    def run_batch(
        self,
        timing_events_list: List[List[TimingDeviation]],
        frame: Any,
    ) -> List[ExecutionTrace]:
        """
        Execute SCOPE program on a batch of timing event sequences.

        Args:
            timing_events_list: List of timing event streams
            frame: Current image frame (shared for all)

        Returns:
            List of ExecutionTrace objects
        """
        results = []
        for i, events in enumerate(timing_events_list):
            self.logger.info(f"Batch {i+1}/{len(timing_events_list)}")
            result = self.run(events, frame)
            results.append(result)
        return results

    def _phase_compile(self, timing_events: List[TimingDeviation]) -> CompilePhaseOutput:
        """Execute COMPILE phase"""
        inputs = CompilePhaseInput(
            timing_events=timing_events,
            dispatch_table=self.program.dispatch_table,
            depth=self.program.depth,
        )
        return compile_phase(inputs)

    def _phase_assign(self, compile_result: CompilePhaseOutput) -> Optional[str]:
        """Execute ASSIGN phase (implicit lookup)"""
        # ASSIGN is implicit: dispatch_table.dispatch() already ran in COMPILE
        # Here we return the selected morphism ID
        if compile_result.cell_id is None:
            self.logger.warning("ASSIGN: No cell matched")
            return None

        # Look up morphism for this cell (would be defined in dispatch table)
        # For now, just return the first morphism
        if self.program.morphisms:
            morphism_id = list(self.program.morphisms.keys())[0]
            self.logger.info(f"ASSIGN: Dispatching to morphism {morphism_id}")
            return morphism_id
        return None

    def _phase_measure(self, frame: Any) -> MeasurePhaseOutput:
        """Execute MEASURE phase"""
        inputs = MeasurePhaseInput(
            frame=frame,
            field_size_x=self.program.field_size_x,
            field_size_y=self.program.field_size_y,
            resolution=self.program.resolution,
            depth=self.program.depth,
            lambda_s=self.program.lambda_s,
            lambda_t=self.program.lambda_t,
        )
        return measure_phase(inputs)

    def _phase_execute(
        self,
        morphism_id: str,
        compile_result: CompilePhaseOutput,
        measure_result: MeasurePhaseOutput,
    ) -> ExecutePhaseOutput:
        """Execute EXECUTE phase"""
        morphism = self.program.morphisms[morphism_id]

        inputs = ExecutePhaseInput(
            morphism_chain=morphism,
            initial_partition_state=compile_result.partition_state,
            frame=None,  # Placeholder
            coord_field=measure_result.coord_field,
        )
        return execute_phase(inputs)

    def _phase_emit(
        self,
        execute_result: ExecutePhaseOutput,
        compile_result: CompilePhaseOutput,
    ) -> SCOPEResult:
        """Execute EMIT phase"""
        inputs = EmitPhaseInput(
            final_partition_state=execute_result.final_partition_state,
            measurement=execute_result.measurement,
            structure_id=execute_result.structure_id,
            s_entropy_before=execute_result.s_entropy_before,
            s_entropy_after=execute_result.s_entropy_after,
        )
        return emit_phase(inputs)


def create_runtime(program: SCOPEProgram) -> SCOPERuntime:
    """Factory function to create a SCOPE runtime"""
    return SCOPERuntime(program)
