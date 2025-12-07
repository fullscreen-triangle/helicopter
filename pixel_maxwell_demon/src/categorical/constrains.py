"""
Constraint network management for phase-locked systems.

Manages the constraint graph G = (V, E) where:
- V: Oscillatory modes
- E: Phase-lock constraints between modes
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Set, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd.bmd_state import BMDState


class ConstraintNetwork:
    """
    Manage constraint graph for phase-locked oscillatory systems.

    The constraint graph G = (V, E) encodes:
    - Vertices V: Oscillatory modes
    - Edges E: Phase-lock constraints

    Entropy increases as |E| increases (more constraints = higher S).
    """

    def __init__(self):
        """Initialize empty constraint network."""
        self.graph = nx.Graph()
        self.mode_properties = {}  # mode_id → properties
        self.constraint_properties = {}  # (mode_i, mode_j) → constraint

    def add_mode(
        self,
        mode_id: str,
        frequency: float,
        phase: float,
        properties: Dict[str, Any] = None
    ):
        """
        Add oscillatory mode to network.

        Args:
            mode_id: Unique mode identifier
            frequency: Oscillatory frequency (Hz)
            phase: Current phase (radians)
            properties: Additional mode properties
        """
        self.graph.add_node(
            mode_id,
            frequency=frequency,
            phase=phase
        )

        if properties:
            self.mode_properties[mode_id] = properties

    def add_constraint(
        self,
        mode_i: str,
        mode_j: str,
        phase_relationship: str = 'coupled',
        coupling_strength: float = 1.0,
        phase_offset: float = 0.0
    ):
        """
        Add phase-lock constraint between modes.

        Args:
            mode_i: First mode ID
            mode_j: Second mode ID
            phase_relationship: Type of coupling ('coupled', 'synchronized', 'anti-phase')
            coupling_strength: Strength of constraint [0, 1]
            phase_offset: Required phase offset (radians)
        """
        self.graph.add_edge(
            mode_i,
            mode_j,
            relationship=phase_relationship,
            strength=coupling_strength,
            offset=phase_offset
        )

        self.constraint_properties[(mode_i, mode_j)] = {
            'type': phase_relationship,
            'strength': coupling_strength,
            'offset': phase_offset
        }

    def remove_constraint(self, mode_i: str, mode_j: str):
        """Remove phase-lock constraint."""
        if self.graph.has_edge(mode_i, mode_j):
            self.graph.remove_edge(mode_i, mode_j)
            self.constraint_properties.pop((mode_i, mode_j), None)
            self.constraint_properties.pop((mode_j, mode_i), None)

    def get_constraint_count(self) -> int:
        """
        Get total number of constraints |E|.

        Returns:
            Number of edges in constraint graph
        """
        return self.graph.number_of_edges()

    def get_mode_count(self) -> int:
        """
        Get total number of modes |V|.

        Returns:
            Number of vertices in constraint graph
        """
        return self.graph.number_of_nodes()

    def constraint_density(self) -> float:
        """
        Calculate constraint density.

        Density = |E| / (|V| * (|V| - 1) / 2)

        Returns:
            Density in [0, 1], 1 = fully connected
        """
        n_modes = self.get_mode_count()
        if n_modes < 2:
            return 0.0

        max_edges = n_modes * (n_modes - 1) / 2
        actual_edges = self.get_constraint_count()

        return actual_edges / max_edges

    def entropy_from_constraints(self, kB: float = 1.0) -> float:
        """
        Calculate entropy contribution from constraints.

        S ∝ |E(G)| (more constraints = higher entropy)

        This is the key insight: phase-lock constraints INCREASE entropy
        by reducing accessible microstates.

        Args:
            kB: Boltzmann constant scale factor

        Returns:
            Entropy value
        """
        n_constraints = self.get_constraint_count()

        # Entropy grows with constraint count
        S = kB * n_constraints

        return S

    def add_bmd_constraints(self, bmd_state: 'BMDState', prefix: str = ''):
        """
        Add constraints from BMD phase structure to network.

        Args:
            bmd_state: BMD state with phase structure
            prefix: Prefix for mode IDs
        """
        phases = bmd_state.phase.phases
        frequencies = bmd_state.phase.frequencies
        coherence = bmd_state.phase.coherence

        n_modes = len(phases)

        # Add modes
        for i in range(n_modes):
            mode_id = f"{prefix}mode_{i}"
            self.add_mode(
                mode_id,
                frequency=frequencies[i],
                phase=phases[i]
            )

        # Add constraints based on coherence matrix
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                # Add constraint if coherence is significant
                if coherence[i, j] > 0.5:
                    mode_i = f"{prefix}mode_{i}"
                    mode_j = f"{prefix}mode_{j}"

                    self.add_constraint(
                        mode_i,
                        mode_j,
                        coupling_strength=coherence[i, j],
                        phase_offset=phases[j] - phases[i]
                    )

    def merge_networks(self, other: 'ConstraintNetwork', coupling: float = 0.8):
        """
        Merge another constraint network into this one.

        Creates inter-network constraints based on coupling strength.

        Args:
            other: Another constraint network
            coupling: Inter-network coupling strength
        """
        # Add all nodes from other
        for node, data in other.graph.nodes(data=True):
            if node not in self.graph:
                self.graph.add_node(node, **data)

        # Add all edges from other
        for u, v, data in other.graph.edges(data=True):
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, **data)

        # Add inter-network constraints (sample a few)
        my_nodes = list(self.graph.nodes())
        other_nodes = list(other.graph.nodes())

        n_inter = min(5, len(my_nodes), len(other_nodes))

        for _ in range(n_inter):
            i = np.random.choice(my_nodes)
            j = np.random.choice(other_nodes)

            if i != j and not self.graph.has_edge(i, j):
                self.add_constraint(i, j, coupling_strength=coupling)

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """
        Find strongly connected components in constraint network.

        Returns:
            List of connected component sets
        """
        return list(nx.connected_components(self.graph))

    def get_mode_degree(self, mode_id: str) -> int:
        """
        Get degree (number of constraints) for a mode.

        Args:
            mode_id: Mode identifier

        Returns:
            Number of constraints involving this mode
        """
        if mode_id not in self.graph:
            return 0
        return self.graph.degree(mode_id)

    def constraint_satisfaction_check(self) -> dict:
        """
        Check if all phase-lock constraints are satisfied.

        Returns:
            Dict with:
            - satisfied: Number of satisfied constraints
            - violated: Number of violated constraints
            - satisfaction_rate: Fraction satisfied
        """
        satisfied = 0
        violated = 0

        for u, v, data in self.graph.edges(data=True):
            phase_u = self.graph.nodes[u]['phase']
            phase_v = self.graph.nodes[v]['phase']
            required_offset = data.get('offset', 0.0)
            tolerance = 2 * np.pi / 16  # ~22.5 degrees

            actual_offset = (phase_v - phase_u) % (2 * np.pi)

            if np.abs(actual_offset - required_offset) < tolerance:
                satisfied += 1
            else:
                violated += 1

        total = satisfied + violated
        rate = satisfied / total if total > 0 else 0.0

        return {
            'satisfied': satisfied,
            'violated': violated,
            'total': total,
            'satisfaction_rate': rate
        }

    def to_dict(self) -> dict:
        """Serialize constraint network to dictionary."""
        return {
            'nodes': [
                {
                    'id': node,
                    **data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            'n_modes': self.get_mode_count(),
            'n_constraints': self.get_constraint_count(),
            'density': self.constraint_density()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConstraintNetwork':
        """Deserialize constraint network from dictionary."""
        network = cls()

        # Add nodes
        for node_data in data['nodes']:
            node_id = node_data.pop('id')
            network.graph.add_node(node_id, **node_data)

        # Add edges
        for edge_data in data['edges']:
            u = edge_data.pop('source')
            v = edge_data.pop('target')
            network.graph.add_edge(u, v, **edge_data)

        return network

    def __repr__(self) -> str:
        return (
            f"ConstraintNetwork(modes={self.get_mode_count()}, "
            f"constraints={self.get_constraint_count()}, "
            f"density={self.constraint_density():.3f})"
        )
