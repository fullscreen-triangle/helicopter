"""
Temporal programming types: timing cells, trajectory classification, dispatch rules.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


@dataclass
class TimingDeviation:
    """
    Single timing event: ΔP = T_ref - t_rec

    Represents the deviation of recorded time from reference time.
    Attributes:
        delta_p: Time deviation in seconds
        channel_id: Which acquisition channel (0, 1, 2, ... for different lasers/PMTs)
        intensity: Optional intensity or count value
    """
    delta_p: float  # Seconds
    channel_id: int  # Acquisition channel index
    intensity: Optional[float] = None  # Optional amplitude

    def __str__(self) -> str:
        return f"ΔP({self.delta_p:.2e}s, ch={self.channel_id})"


@dataclass
class Trajectory:
    """
    Accumulated sequence of timing deviations: τ = (ΔP₁, ΔP₂, ..., ΔPₙ)

    Represents a multi-event sequence from acquisition.
    """
    events: List[TimingDeviation] = field(default_factory=list)
    trajectory_id: Optional[int] = None  # Unique ID for this trajectory

    def __len__(self) -> int:
        return len(self.events)

    def add_event(self, event: TimingDeviation) -> None:
        """Append timing event to trajectory"""
        self.events.append(event)

    def is_complete(self, depth: int) -> bool:
        """Check if trajectory has accumulated enough events"""
        return len(self.events) >= depth

    def mean_delta_p(self) -> float:
        """Mean timing deviation"""
        if not self.events:
            return 0.0
        return sum(e.delta_p for e in self.events) / len(self.events)

    def dominant_channel(self) -> int:
        """Most frequent channel in trajectory"""
        if not self.events:
            return 0
        channels = [e.channel_id for e in self.events]
        return max(set(channels), key=channels.count)

    def to_dict(self) -> dict:
        return {
            'trajectory_id': self.trajectory_id,
            'length': len(self.events),
            'mean_delta_p': self.mean_delta_p(),
            'events': [
                {'delta_p': e.delta_p, 'channel_id': e.channel_id, 'intensity': e.intensity}
                for e in self.events
            ]
        }


@dataclass
class TimingCell:
    """
    Cell in timing space: a Borel set partition of ΔP × channel × trajectory space.

    Each cell represents a classification: if a trajectory falls within these bounds,
    it gets assigned this cell ID and triggers the corresponding dispatch action.

    Attributes:
        cell_id: Human-readable label (e.g., "PROPHASE", "METAPHASE")
        bounds_delta_p: (min, max) for ΔP in seconds
        channel_ids: List of allowed channels, or None for all
        depth: Required accumulation depth for classification
    """
    cell_id: str
    bounds_delta_p: Tuple[float, float]  # (min, max) in seconds
    channel_ids: Optional[List[int]] = None  # None means accept all channels
    depth: int = 1000  # Default depth from coordinate_space

    def contains_trajectory(self, trajectory: Trajectory) -> bool:
        """Check if trajectory is classified in this cell"""
        if not trajectory.is_complete(self.depth):
            return False

        mean_delta_p = trajectory.mean_delta_p()
        if not (self.bounds_delta_p[0] <= mean_delta_p <= self.bounds_delta_p[1]):
            return False

        if self.channel_ids is not None:
            if trajectory.dominant_channel() not in self.channel_ids:
                return False

        return True

    def __str__(self) -> str:
        return f"Cell({self.cell_id}: ΔP ∈ [{self.bounds_delta_p[0]:.2e}, {self.bounds_delta_p[1]:.2e}])"

    def to_dict(self) -> dict:
        return {
            'cell_id': self.cell_id,
            'bounds_delta_p': list(self.bounds_delta_p),
            'channel_ids': self.channel_ids,
            'depth': self.depth,
        }


@dataclass
class CellPartition:
    """
    Collection of timing cells partitioning the ΔP space.

    Maintains a sorted list of cells for efficient lookup.
    """
    cells: List[TimingCell] = field(default_factory=list)

    def add_cell(self, cell: TimingCell) -> None:
        """Add a timing cell to the partition"""
        self.cells.append(cell)
        # Sort by lower bound for efficient lookup
        self.cells.sort(key=lambda c: c.bounds_delta_p[0])

    def classify(self, trajectory: Trajectory) -> Optional[TimingCell]:
        """
        Classify a trajectory: find the cell it belongs to.

        Uses binary search on depth, then linear search on bounds.
        Returns the first (lowest ΔP) matching cell, or None if no match.
        """
        for cell in self.cells:
            if cell.contains_trajectory(trajectory):
                return cell
        return None

    def overlap_fraction(self) -> float:
        """
        Estimate coverage of ΔP space by cells.

        Returns fraction of ΔP range covered (0 to 1).
        If cells don't overlap, returns fraction of union coverage.
        """
        if not self.cells:
            return 0.0

        # Find overall bounds
        all_mins = [c.bounds_delta_p[0] for c in self.cells]
        all_maxs = [c.bounds_delta_p[1] for c in self.cells]
        global_min, global_max = min(all_mins), max(all_maxs)

        if global_max <= global_min:
            return 0.0

        # Compute union of intervals (simple merge)
        intervals = sorted([c.bounds_delta_p for c in self.cells])
        merged = [intervals[0]]
        for current_min, current_max in intervals[1:]:
            last_min, last_max = merged[-1]
            if current_min <= last_max:
                merged[-1] = (last_min, max(last_max, current_max))
            else:
                merged.append((current_min, current_max))

        total_coverage = sum(m - n for n, m in merged)
        total_range = global_max - global_min
        return total_coverage / total_range if total_range > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'cells': [c.to_dict() for c in self.cells],
            'overlap_fraction': self.overlap_fraction(),
        }


@dataclass
class DispatchRule:
    """
    Rule mapping a cell to an action.

    Attributes:
        cell_id: The timing cell to match
        action: Action to execute (morphism chain ID or special action)
        action_type: "morphism", "emit", or "sequence"
    """
    cell_id: str
    action: str  # morphism chain ID or action name
    action_type: str = "morphism"  # "morphism" | "emit" | "sequence"

    def to_dict(self) -> dict:
        return {
            'cell_id': self.cell_id,
            'action': self.action,
            'action_type': self.action_type,
        }


@dataclass
class DispatchTable:
    """
    Table of dispatch rules: maps cell_id → action.

    Maintains both cell partition and action rules.
    """
    cell_partition: CellPartition = field(default_factory=CellPartition)
    rules: Dict[str, DispatchRule] = field(default_factory=dict)

    def add_cell(self, cell: TimingCell) -> None:
        """Register a timing cell"""
        self.cell_partition.add_cell(cell)

    def add_rule(self, rule: DispatchRule) -> None:
        """Register a dispatch rule"""
        self.rules[rule.cell_id] = rule

    def dispatch(self, trajectory: Trajectory) -> Optional[DispatchRule]:
        """
        Given a trajectory, find the applicable dispatch rule.

        Returns:
            DispatchRule if a matching cell and rule exist, None otherwise.
        """
        cell = self.cell_partition.classify(trajectory)
        if cell is not None:
            return self.rules.get(cell.cell_id)
        return None

    def to_dict(self) -> dict:
        return {
            'cell_partition': self.cell_partition.to_dict(),
            'rules': {cid: r.to_dict() for cid, r in self.rules.items()},
        }
