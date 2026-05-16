from __future__ import annotations


class StatsWindow:
    """Running sum/sum-of-squares/count stats for one sliding window.

    Used in pairs (_l = accumulating, _r = reference) to mirror the
    HalfSpaceTrees l_mass / r_mass pivot mechanism so z-score baselines
    always reflect the same window the model scores against.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._sum_sq: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, key: str, value: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + value
        self._sum_sq[key] = self._sum_sq.get(key, 0.0) + value * value
        self._counts[key] = self._counts.get(key, 0) + 1

    @property
    def fields(self) -> list[str]:
        return sorted(self._counts.keys())

    @property
    def total_samples(self) -> int:
        return sum(self._counts.values())

    def is_empty(self) -> bool:
        return not self._counts

    def mean(self, key: str) -> float:
        n = self._counts.get(key, 0)
        if n == 0:
            return 0.0
        return self._sums[key] / n

    def std(self, key: str) -> float:
        n = self._counts.get(key, 0)
        if n == 0:
            return 1.0
        mean = self._sums[key] / n
        variance = max(self._sum_sq[key] / n - mean ** 2, 0.0)
        return variance ** 0.5 if variance > 0 else 1.0
