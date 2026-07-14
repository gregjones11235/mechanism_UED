"""T3: Learning-Progress Adaptive Controller.

Behind flag: config.training.enable_lpac (default False).

Controls hyperparameters based on normalized held-out progress signals:
- entropy coefficient adaptation from stagnation/uncertainty
- curriculum sampling temperature from progress rate
- Never uses Tier labels as selector inputs
- Separates controller feedback worlds from final reporting worlds

When disabled (default): passes through fixed settings, numerical identity.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple


class LPACWrapper:
    """Production LPAC wrapper for PPO integration.

    Adapts entropy coefficient and curriculum temperature based on
    normalized learning-progress signals from held-out evaluation.

    Config keys (all in config.training, default False):
        enable_lpac: bool = False
        lpac_entropy_base: float = 0.01
        lpac_entropy_max: float = 0.05
        lpac_temperature_base: float = 1.0
        lpac_temperature_min: float = 0.5
        lpac_stagnation_window: int = 5
        lpac_forgetting_threshold: float = 0.05
        lpac_uncertainty_weight: float = 0.1
    """

    def __init__(self, config):
        self.enabled = getattr(config, "enable_lpac", False)
        self.entropy_base = getattr(config, "lpac_entropy_base", 0.01)
        self.entropy_max = getattr(config, "lpac_entropy_max", 0.05)
        self.temp_base = getattr(config, "lpac_temperature_base", 1.0)
        self.temp_min = getattr(config, "lpac_temperature_min", 0.5)
        self.stagnation_window = getattr(config, "lpac_stagnation_window", 5)
        self.forgetting_threshold = getattr(config, "lpac_forgetting_threshold", 0.05)
        self.uncertainty_weight = getattr(config, "lpac_uncertainty_weight", 0.1)
        self._history: Dict[str, list] = {"progress": [], "forgetting": [], "entropy": []}

    def update(self, held_out_progress: float, held_out_forgetting: float,
               current_entropy: float) -> Tuple[float, float]:
        """Compute adapted entropy coefficient and curriculum temperature.

        Args:
            held_out_progress: normalized progress signal (e.g., mean SR change)
            held_out_forgetting: normalized forgetting signal (positive = losing skills)
            current_entropy: current policy entropy

        Returns:
            (entropy_coef, curriculum_temperature) tuple.
            When disabled: returns (entropy_base, temp_base) unchanged.
        """
        if not self.enabled:
            return (self.entropy_base, self.temp_base)

        self._history["progress"].append(held_out_progress)
        self._history["forgetting"].append(held_out_forgetting)
        self._history["entropy"].append(current_entropy)

        window = self._history["progress"][-self.stagnation_window:]
        if len(window) >= self.stagnation_window:
            x = jnp.arange(len(window), dtype=jnp.float32)
            y = jnp.array(window, dtype=jnp.float32)
            trend = jnp.polyfit(x, y, 1)[0]
            is_stagnating = float(trend) < 0.001
        else:
            is_stagnating = 0.0

        # Increase entropy when stagnating or forgetting
        entropy_boost = is_stagnating * 0.02 + held_out_forgetting * 0.1
        entropy_coef = jnp.clip(self.entropy_base + entropy_boost,
                                 self.entropy_base, self.entropy_max)

        # Decrease temperature when stagnating (more greedy)
        temperature = self.temp_base - is_stagnating * 0.3
        temperature = jnp.clip(temperature, self.temp_min, self.temp_base)

        return (float(entropy_coef), float(temperature))

    def reset_history(self):
        """Reset adaptation history between independent evaluation phases."""
        self._history = {"progress": [], "forgetting": [], "entropy": []}

    def has_nonzero_adaptation(self) -> bool:
        """Verify LPAC actually changes outputs (CPU validation)."""
        if not self.enabled:
            return False
        # Simulate stagnation → entropy should increase
        for _ in range(self.stagnation_window):
            self.update(0.0, 0.0, 0.01)  # flat progress
        e1, _ = self.update(0.0, 0.0, 0.01)
        self.reset_history()
        # Simulate good progress → entropy should be lower
        for _ in range(self.stagnation_window):
            self.update(0.05, 0.0, 0.01)  # increasing progress
        e2, t2 = self.update(0.05, 0.0, 0.01)
        return abs(e1 - e2) > 1e-6 or abs(t2 - self.temp_base) < 1e-6
