from __future__ import annotations

from app.features.preprocessors import FieldInfo
from app.models.half_space_trees.stats_window import StatsWindow


class ModelSlice:
    """One staggered unit in a HalfSpaceTreesDetector: an HST instance coupled with its StatsWindow pair.

    The HST and z-score windows share the same phase offset and pivot trigger so
    they always describe the same reference period. Preprocessors live in the
    parent Detector and are shared across all slices; learn_one() and score_one()
    receive the already-preprocessed feature dict.
    """

    def __init__(self, n_trees: int, height: int, window_size: int, seed: int) -> None:
        from river import anomaly
        self._hst = anomaly.HalfSpaceTrees(
            n_trees=n_trees, height=height, window_size=window_size, seed=seed
        )
        self._window_size = window_size
        self._l_window = StatsWindow()
        self._r_window = StatsWindow()
        self._window_counter: int = 0
        self._first_window: bool = True
        self._limits_seen: dict[str, tuple[float, float]] = {}

    @property
    def is_ready(self) -> bool:
        """True once the first pivot has completed and a reference window exists."""
        return not self._first_window

    @property
    def recency(self) -> int:
        """Events since the last pivot. Lower means a more recently refreshed reference."""
        return self._window_counter

    @property
    def r_window(self) -> StatsWindow:
        return self._r_window

    def learn_one(self, final: dict[FieldInfo, float]) -> None:
        str_final = self._register_limits_and_flatten(final)
        for key, value in str_final.items():
            self._l_window.update(key, value)
        self._hst.learn_one(str_final)
        self._window_counter += 1
        if self._window_counter == self._window_size:
            self._r_window = self._l_window
            self._l_window = StatsWindow()
            self._window_counter = 0
            self._first_window = False

    def score_one(self, final: dict[FieldInfo, float]) -> float:
        str_final = self._register_limits_and_flatten(final)
        return self._hst.score_one(str_final)

    def _register_limits_and_flatten(self, final: dict[FieldInfo, float]) -> dict[str, float]:
        """Convert FieldInfo keys to str, update HST limits for new keys, validate ranges."""
        result: dict[str, float] = {}
        for fi, v in final.items():
            key = fi.unique_key
            if key not in self._limits_seen:
                effective = fi.limits if fi.limits is not None else (-1.0, 1.0)
                self._limits_seen[key] = effective
                if fi.limits is not None:
                    self._hst.limits[key] = fi.limits
            lo, hi = self._limits_seen[key]
            if not (lo <= v <= hi):
                raise ValueError(
                    f"feature '{key}' = {v} is outside expected range [{lo}, {hi}]"
                )
            result[key] = v
        return result
