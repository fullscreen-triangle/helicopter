"""
Convergence monitoring for HCCC algorithm.

Monitors:
- Network coherence achievement
- Ambiguity reduction
- Stream divergence stabilization
- Richness growth saturation
"""

import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..vision.bmd import NetworkBMD


class ConvergenceMonitor:
    """
    Monitor algorithm convergence.

    Convergence criteria:
    1. Network coherence: A(β^(network), R) < A_coherence for all R
    2. Ambiguity stable: No significant decrease in recent steps
    3. Richness saturated: O(2^n) growth begins to plateau
    4. Stream coherence: Divergence remains bounded
    """

    def __init__(
        self,
        coherence_threshold: float = 1.0,
        stability_window: int = 10,
        richness_saturation_factor: float = 0.1
    ):
        """
        Initialize convergence monitor.

        Args:
            coherence_threshold: A_coherence for network coherence
            stability_window: Window size for stability checks
            richness_saturation_factor: Threshold for richness growth slowdown
        """
        self.A_coherence = coherence_threshold
        self.stability_window = stability_window
        self.saturation_factor = richness_saturation_factor

        # History tracking
        self.ambiguity_history: List[float] = []
        self.divergence_history: List[float] = []
        self.richness_history: List[float] = []
        self.iteration_history: List[int] = []

        # Convergence state
        self._converged = False
        self._convergence_reason = None

    def update(
        self,
        network_bmd: 'NetworkBMD',
        iteration: int,
        ambiguity: float,
        divergence: float
    ):
        """
        Update convergence monitor with new iteration data.

        Args:
            network_bmd: Current network BMD
            iteration: Current iteration number
            ambiguity: Current ambiguity value
            divergence: Current stream divergence
        """
        # Record history
        self.iteration_history.append(iteration)
        self.ambiguity_history.append(ambiguity)
        self.divergence_history.append(divergence)
        self.richness_history.append(network_bmd.network_categorical_richness())

        # Check convergence criteria
        if len(self.ambiguity_history) >= self.stability_window:
            self._check_convergence()

    def _check_convergence(self):
        """Check all convergence criteria."""
        # 1. Ambiguity coherence achieved
        if self._check_ambiguity_coherence():
            self._converged = True
            self._convergence_reason = "ambiguity_coherence"
            return

        # 2. Ambiguity stabilized
        if self._check_ambiguity_stability():
            self._converged = True
            self._convergence_reason = "ambiguity_stable"
            return

        # 3. Richness saturated
        if self._check_richness_saturation():
            self._converged = True
            self._convergence_reason = "richness_saturated"
            return

        # 4. Stream divergence bounded
        if self._check_divergence_bounded():
            # Not sufficient alone, but good sign
            pass

    def _check_ambiguity_coherence(self) -> bool:
        """Check if ambiguity below coherence threshold."""
        if not self.ambiguity_history:
            return False

        recent_ambiguity = np.mean(self.ambiguity_history[-self.stability_window:])

        return recent_ambiguity < self.A_coherence

    def _check_ambiguity_stability(self) -> bool:
        """Check if ambiguity has stabilized (minimal change)."""
        if len(self.ambiguity_history) < self.stability_window:
            return False

        recent = self.ambiguity_history[-self.stability_window:]

        # Compute coefficient of variation
        if np.mean(recent) > 0:
            cv = np.std(recent) / np.mean(recent)
        else:
            cv = 0.0

        # Stable if CV < 10%
        return cv < 0.1

    def _check_richness_saturation(self) -> bool:
        """Check if richness growth has saturated."""
        if len(self.richness_history) < self.stability_window:
            return False

        recent = np.array(self.richness_history[-self.stability_window:])

        # Fit exponential growth rate
        log_richness = np.log(recent + 1e-10)
        steps = np.arange(len(log_richness))

        if len(steps) < 2:
            return False

        # Growth rate (slope in log space)
        growth_rate = np.polyfit(steps, log_richness, deg=1)[0]

        # Saturated if growth rate < saturation_factor
        return growth_rate < self.saturation_factor

    def _check_divergence_bounded(self) -> bool:
        """Check if stream divergence remains bounded."""
        if len(self.divergence_history) < self.stability_window:
            return False

        recent = self.divergence_history[-self.stability_window:]

        # Check if divergence not increasing
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        # Bounded if not increasing significantly
        return second_half <= first_half * 1.2

    def has_converged(self) -> bool:
        """Check if algorithm has converged."""
        return self._converged

    def get_convergence_reason(self) -> Optional[str]:
        """Get reason for convergence."""
        return self._convergence_reason

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """
        Get convergence metrics.

        Returns:
            Dict with convergence status and metrics
        """
        if not self.ambiguity_history:
            return {
                'converged': False,
                'reason': None,
                'iterations': 0
            }

        metrics = {
            'converged': self._converged,
            'reason': self._convergence_reason,
            'iterations': len(self.iteration_history),
            'final_ambiguity': self.ambiguity_history[-1],
            'final_divergence': self.divergence_history[-1],
            'final_richness': self.richness_history[-1],
            'ambiguity_reduction': (
                self.ambiguity_history[0] - self.ambiguity_history[-1]
                if len(self.ambiguity_history) > 1 else 0.0
            ),
            'richness_growth_rate': self._compute_richness_growth_rate()
        }

        # Stability metrics
        if len(self.ambiguity_history) >= self.stability_window:
            recent_ambiguity = self.ambiguity_history[-self.stability_window:]
            metrics['ambiguity_cv'] = (
                np.std(recent_ambiguity) / (np.mean(recent_ambiguity) + 1e-10)
            )

        return metrics

    def _compute_richness_growth_rate(self) -> float:
        """Compute current richness growth rate."""
        if len(self.richness_history) < 3:
            return 0.0

        log_richness = np.log(np.array(self.richness_history) + 1e-10)
        steps = np.arange(len(log_richness))

        growth_rate = np.polyfit(steps, log_richness, deg=1)[0]

        return growth_rate

    def reset(self):
        """Reset convergence monitor."""
        self.ambiguity_history = []
        self.divergence_history = []
        self.richness_history = []
        self.iteration_history = []
        self._converged = False
        self._convergence_reason = None

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot convergence metrics over iterations.

        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Ambiguity
            axes[0, 0].plot(self.iteration_history, self.ambiguity_history)
            axes[0, 0].axhline(y=self.A_coherence, color='r', linestyle='--', label='Threshold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Ambiguity')
            axes[0, 0].set_title('Ambiguity vs Iteration')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Stream divergence
            axes[0, 1].plot(self.iteration_history, self.divergence_history)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Stream Divergence')
            axes[0, 1].set_title('Stream Divergence vs Iteration')
            axes[0, 1].grid(True)

            # Richness (log scale)
            axes[1, 0].semilogy(self.iteration_history, self.richness_history)
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Network Richness (log scale)')
            axes[1, 0].set_title('Network Richness Growth')
            axes[1, 0].grid(True)

            # Convergence status
            metrics = self.get_convergence_metrics()
            status_text = '\n'.join([
                f"Converged: {metrics['converged']}",
                f"Reason: {metrics.get('reason', 'N/A')}",
                f"Iterations: {metrics['iterations']}",
                f"Final Ambiguity: {metrics.get('final_ambiguity', 0):.3f}",
                f"Ambiguity Reduction: {metrics.get('ambiguity_reduction', 0):.3f}",
                f"Growth Rate: {metrics.get('richness_growth_rate', 0):.3f}"
            ])

            axes[1, 1].text(0.1, 0.5, status_text, fontsize=12, verticalalignment='center')
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Convergence Status')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved convergence plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("matplotlib not available for plotting")
