"""Deterministic pseudo-random generator used by every solver in this repo."""

from __future__ import annotations

from dataclasses import dataclass


UINT32_MAX = 0xFFFFFFFF
UINT32_SCALE = float(UINT32_MAX) + 1.0


@dataclass
class SeededRandom:
    """Simple xorshift32 generator with a Math.random-like interface."""

    seed: int

    def __post_init__(self) -> None:
        state = self.seed & UINT32_MAX
        # JS Math.random can never stay at zero; pick a non-zero default.
        if state == 0:
            state = 0xA5366B4D
        self._state = state

    def random(self) -> float:
        """Return a float in [0, 1) using the xorshift32 recurrence."""
        x = self._state
        x ^= (x << 13) & UINT32_MAX
        x ^= (x >> 17) & UINT32_MAX
        x ^= (x << 5) & UINT32_MAX
        self._state = x & UINT32_MAX
        return self._state / UINT32_SCALE

    def randint(self, n: int) -> int:
        """Return a random integer in [0, n)."""
        if n <= 0:
            raise ValueError("Upper bound must be positive")
        return int(self.random() * n)
