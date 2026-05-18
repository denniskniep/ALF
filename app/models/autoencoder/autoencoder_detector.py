from __future__ import annotations

import bisect
import threading
from collections import deque
from typing import Any

import torch
import torch.nn as nn

from app.utils import get_device

from app.features.preprocessors.base import FieldInfo
from app.models.autoencoder.network import EntityEmbeddingAutoencoder

from app.models.base import (
    BaseModel,
    CohortExplanation,
    DetectorResult,
    FieldContribution,
)

_EMBEDDING_PREPROCESSORS: frozenset[str] = frozenset({"LabelIndex", "HashIndex"})

class AutoencoderDetector(BaseModel):
    """Autoencoder anomaly detector with periodic warm-start batch retraining.

    Learns to reconstruct normal events via a small neural autoencoder; high
    reconstruction MSE signals an anomaly. Anomaly scores are percentile ranks of
    the current MSE within the baseline computed at the last retrain.

    Uses entity embeddings for categorical features to capture co-occurrence
    structure (e.g. "Chrome + Linux" is anomalous even when each value is
    individually common).

    Training: events accumulate in a rolling buffer. Every retrain_every ingest
    events the model is retrained from its current weights for exactly
    epochs_per_retrain passes over the entire buffer — warm-start batch training.
    Weights are NOT reset between retrains. They carry weight-level memory of
    long-term normal behaviour across all previous retrains, far beyond the hard
    data boundary of the rolling buffer.

    Epochs are intentionally fixed (not convergence-based). Fixed epochs
    guarantee every event receives equal gradient exposure over the model's lifetime.

    Missing features are imputed at both train and score time:
    - Categorical: see max observed/expected LabelIndex and vocab_headroom
    - Numeric: [0.0, 0.0] pair — distinguishes absent from a field present with a
      near-mean value, which contributes [v, 1.0].

    Feature order is fixed once at model-build time from the union of all FieldInfos
    seen across the entire buffer. A field absent from early events but present
    later is still included and zero-imputed rather than silently dropped.

    Parameters
    ----------
    buffer_size
        Capacity of the rolling event buffer. Oldest events are evicted when full.
    retrain_every
        Trigger a retrain after every N new ingest events.
    epochs_per_retrain
        Fixed number of full buffer passes per retrain cycle. Keep this low (1–5)
        rather than chasing convergence — warm-start weights already encode
        historical normal behaviour; each cycle should be a gentle nudge.
    min_buffer_size
        Model is not built until this many events have been buffered.
    embed_dim
        Global cap on embedding dimensions per categorical field. Actual dim per field
        is min(embed_dim, vocab_size // 2 + 1) at model-build time — small-vocab fields
        get fewer dims, large-vocab fields hit the cap. The cap must be the same for all
        fields: a higher cap on one field gives it more width in the concatenated input
        vector and implicitly more influence on the anomaly signal. Any vocab_size above
        2 * embed_dim - 2 hits the cap, so for the default embed_dim=8 any field with
        more than 14 expected categories gets exactly 8 dims.
    bottleneck
        Latent dimension. None auto-derives as max(total_input_dim // 2, 2).
    batch_size
        Number of events per mini-batch during training. The buffer is shuffled
        before each epoch and split into chunks of this size, giving multiple
        gradient steps per epoch and better optimization for large buffers.
        If batch_size >= len(buffer) the epoch degrades to a single full-batch
        step.
    vocab_headroom
        Extra rows pre-allocated in each embedding table beyond the vocabulary
        observed at model-build time. Growing a table resets Adam momentum.
    lr
        Adam learning rate.
    seed
        Seed for weight initialisation at model-build time. None = non-deterministic.
    """

    PREPROCESSOR_TYPE_DEFAULTS: dict[str, str] = {
        "numeric":         "StandardScaler",
        "boolean":         "PassThrough",
        "str_categorical": "LabelIndex",
        "str_identifier":  "HashIndex",
        "str_text":        "SentenceTransformerEncoder"
    }

    def __init__(
        self,
        buffer_size: int = 500,
        retrain_every: int = 50,
        epochs_per_retrain: int = 2,
        min_buffer_size: int = 100,
        batch_size: int = 50,
        embed_dim: int = 8,
        bottleneck: int | None = None,
        vocab_headroom: int = 50,
        lr: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self._buffer_size = buffer_size
        self._retrain_every = retrain_every
        self._epochs_per_retrain = epochs_per_retrain
        self._min_buffer_size = min_buffer_size
        self._batch_size = batch_size
        self._embed_dim = embed_dim
        self._bottleneck = bottleneck
        self._vocab_headroom = vocab_headroom
        self._lr = lr
        self._seed = seed

        # Rolling buffer: deque evicts the oldest event automatically when full.
        # This is the sole training set and scoring baseline — the model learns
        # exactly what is in the buffer, nothing more.
        self._buffer: deque = deque(maxlen=buffer_size)

        # Sorted per-event MSEs from the last retrain, used for percentile scoring.
        # None until the first retrain has completed.
        self._sorted_baseline_mses: list[float] | None = None
        self._events_since_retrain: int = 0

        # Feature order fixed once at model-build time from the union of all FieldInfos
        # across the entire buffer (not just the first event).
        self._cat_fi_order: list[FieldInfo] = []
        self._vec_fi_order: list[FieldInfo] = []  # all non-categorical fields

        # Max observed LabelIndex per categorical field, for embedding table sizing.
        self._cat_vocab_sizes: dict[str, int] = {}

        # Vector length per fi.unique_key
        # Fixed at model-build time;
        # Needed for zero-imputation (missing data handling) of absent vector fields
        self._vec_lengths: dict[str, int] = {}

        # Contiguous slice in the AE input vector per fi.unique_key.
        # Each feature occupies its own dedicated chunk of positions
        # that is fed as input vector (1D array of numerical values) into the autoencoder
        # Categorical: embedding-dim–wide.
        # All others: (vec_len + 1)-wide [v0, ..., vN, is_present].
        self._feature_slices: dict[str, tuple[int, int]] = {}

        # All unique_keys in the fixed feature order — populated at model-build time.
        # Used for O(1) unknown-feature detection at train/score time.
        self._known_feature_keys: frozenset[str] = frozenset()

        self._device = torch.device(get_device())
        self._model: EntityEmbeddingAutoencoder | None = None
        self._optimizer: torch.optim.Adam | None = None
        self._n_trained: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # BaseModel                                                            #
    # ------------------------------------------------------------------ #

    def train(self, features: dict[FieldInfo, list[float]], n_learned: int) -> None:
        with self._lock:
            if not features:
                return

            # Feature order is fixed at model-build time and cannot be extended.
            # New fields after build would be silently dropped — fail fast instead.
            if self._model is not None:
                self._assert_no_unknown_features(features, context="train")

            self._buffer.append(features)
            self._update_cat_vocab_sizes(features)
            self._n_trained += 1

            if len(self._buffer) < self._min_buffer_size:
                return

            self._events_since_retrain += 1

            if self._model is None:
                self._build_model()
                self._retrain()
            elif self._events_since_retrain >= self._retrain_every:
                self._retrain()

    def score(
        self,
        features: dict[FieldInfo, list[float]],
        flat: dict[str, Any],
        explain: bool,
    ) -> DetectorResult:
        with self._lock:
            if self._sorted_baseline_mses is None:
                return DetectorResult(score=None)

            self._assert_no_unknown_features(features, context="score")
            cat_indices, num_floats = self._encode_features(features)
            self._assert_cat_indices_in_bounds(cat_indices)

            cat_tensor = torch.tensor([cat_indices], dtype=torch.long, device=self._device)
            num_tensor = torch.tensor([num_floats], dtype=torch.float32, device=self._device)
            self._model.eval()
            with torch.no_grad():
                reconstructed, original_x = self._model(cat_tensor, num_tensor)  # calls forward() via nn.Module.__call__

            mse = float(((reconstructed - original_x) ** 2).mean().item())
            score = self._percentile_to_linear(self._mse_to_percentile_score(mse))

            explanation = None
            if explain:
                explanation = self._build_explanation(
                    features, flat, reconstructed, original_x, score
                )

            return DetectorResult(
                score=score,
                properties={
                    "buffer_size": len(self._buffer),
                    "events_since_retrain": self._events_since_retrain,
                },
                explanation=explanation,
            )

    def get_state(self) -> Any:
        return {
            "buffer": list(self._buffer),
            "sorted_baseline_mses": self._sorted_baseline_mses,
            "events_since_retrain": self._events_since_retrain,
            "cat_fi_order": self._cat_fi_order,
            "vec_fi_order": self._vec_fi_order,
            "cat_vocab_sizes": dict(self._cat_vocab_sizes),
            "vec_lengths": dict(self._vec_lengths),
            "feature_slices": dict(self._feature_slices),
            "n_trained": self._n_trained,
            "model": self._model,
            "optimizer": self._optimizer,
        }

    def set_state(self, state: Any) -> None:
        self._buffer = deque(state["buffer"], maxlen=self._buffer_size)
        self._sorted_baseline_mses = state["sorted_baseline_mses"]
        self._events_since_retrain = state["events_since_retrain"]
        self._cat_fi_order = state["cat_fi_order"]
        self._vec_fi_order = state["vec_fi_order"]
        self._known_feature_keys = frozenset(
            fi.unique_key for fi in self._cat_fi_order
        ) | frozenset(
            fi.unique_key for fi in self._vec_fi_order
        )
        self._cat_vocab_sizes = state["cat_vocab_sizes"]
        self._vec_lengths = state["vec_lengths"]
        self._feature_slices = state["feature_slices"]
        self._n_trained = state["n_trained"]
        self._model = state["model"]
        if self._model is not None:
            self._model.to(self._device)
        self._optimizer = state["optimizer"]

    # ------------------------------------------------------------------ #
    # Model lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def _build_model(self) -> None:
        """Construct the AE network once min_buffer_size events have been buffered.

        Feature order is fixed here — from the union of all FieldInfos in the
        current buffer — and never changes for the lifetime of this model instance.
        """
        self._fix_feature_order_from_buffer()

        if self._seed is not None:
            torch.manual_seed(self._seed)

        vocab_sizes = [
            max(1, self._cat_vocab_sizes.get(fi.original, 1))
            for fi in self._cat_fi_order
        ]
        numeric_dim = sum(self._vec_lengths[fi.unique_key] + 1 for fi in self._vec_fi_order)
        bottleneck = self._resolve_bottleneck(vocab_sizes, numeric_dim)

        self._model = EntityEmbeddingAutoencoder(
            vocab_sizes=vocab_sizes,
            numeric_dim=numeric_dim,
            embed_dim=self._embed_dim,
            bottleneck=bottleneck,
            vocab_headroom=self._vocab_headroom,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._assign_feature_slices()

    def _retrain(self) -> None:
        """Warm-start batch retrain: train on the buffer, then rebuild the scoring baseline."""
        self._assert_vocab_within_embedding_capacity()
        cat_tensor, num_tensor = self._buffer_to_tensors()
        self._run_warm_start_training_epochs(cat_tensor, num_tensor)
        per_event_mses = self._evaluate_buffer_mses(cat_tensor, num_tensor)
        self._update_scoring_baseline_and_fit_metrics(per_event_mses)
        self._events_since_retrain = 0

    # ------------------------------------------------------------------ #
    # Training helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_warm_start_training_epochs(
        self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor
    ) -> None:
        """Run exactly epochs_per_retrain passes over the buffer using mini-batches.

        Each epoch shuffles the buffer independently so batch boundaries vary,
        giving the optimizer a different gradient sequence every pass. When
        batch_size >= len(buffer) the inner loop produces a single full-batch
        step.

        Warm-start: weights are NOT reset between retrains.
        Fixed epochs ensure every event receives equal gradient exposure.
        """
        n = cat_tensor.shape[0]
        self._model.train()
        for _ in range(self._epochs_per_retrain):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, self._batch_size):
                idx = perm[start : start + self._batch_size]
                self._optimizer.zero_grad()
                reconstructed, original = self._model(cat_tensor[idx], num_tensor[idx]) # calls forward() via nn.Module.__call__
                nn.functional.mse_loss(reconstructed, original).backward()
                self._optimizer.step()
        self._model.eval()  # restore eval mode — train() affects Dropout/BatchNorm behaviour

    def _evaluate_buffer_mses(
        self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor
    ) -> list[float]:
        """Compute per-event MSE for all buffered events."""
        with torch.no_grad():
            reconstructed, original = self._model(cat_tensor, num_tensor)  # calls forward() via nn.Module.__call__
            return ((reconstructed - original) ** 2).mean(dim=1).tolist()

    def _update_scoring_baseline_and_fit_metrics(
        self, per_event_mses: list[float]
    ) -> None:
        """Sort MSEs into the percentile-scoring baseline and record fit quality."""
        self._sorted_baseline_mses = sorted(per_event_mses)

    def _assert_vocab_within_embedding_capacity(self) -> None:
        """Raise if any categorical field's observed vocabulary exceeds its embedding table.

        vocab_headroom pre-allocates extra rows at model-build time to absorb new
        vocabulary items that arrive between retrains. If this raises, vocab_headroom
        was set too low for the rate at which new categorical values appear in
        training data. Increase vocab_headroom and reset the model to fix this.
        """
        for i, fi in enumerate(self._cat_fi_order):
            max_idx = self._cat_vocab_sizes.get(fi.original, 0)
            emb = self._model.embeddings[i]
            if max_idx >= emb.num_embeddings:
                raise ValueError(
                    f"Vocabulary for field '{fi.original}' has grown to index {max_idx}, "
                    f"exceeding the embedding table capacity of {emb.num_embeddings} rows "
                    f"(vocab_headroom={self._vocab_headroom} was exhausted). "
                    f"Increase vocab_headroom to pre-allocate more rows at model-build time."
                )

    # ------------------------------------------------------------------ #
    # Feature handling                                                     #
    # ------------------------------------------------------------------ #

    def _fix_feature_order_from_buffer(self) -> None:
        """Fix AE input order from the union of all FieldInfos across the buffer.

        Called once at model-build time. Fields absent from early events but present
        in later buffered ones are included and zero-imputed rather than silently
        dropped.
        """
        seen_cat: set[str] = set()
        seen_vec: set[str] = set()
        for event_features in self._buffer:
            for fi, values in event_features.items():
                if fi.preprocessor in _EMBEDDING_PREPROCESSORS:
                    if fi.unique_key not in seen_cat:
                        self._cat_fi_order.append(fi)
                        seen_cat.add(fi.unique_key)
                else:
                    if fi.unique_key not in seen_vec:
                        self._vec_fi_order.append(fi)
                        self._vec_lengths[fi.unique_key] = len(values)
                        seen_vec.add(fi.unique_key)
        self._known_feature_keys = frozenset(seen_cat) | frozenset(seen_vec)

    def _assign_feature_slices(self) -> None:
        """Map each fi.unique_key to its contiguous slice in the AE input vector."""
        pos = 0
        for i, fi in enumerate(self._cat_fi_order):
            dim = self._model.embed_dims[i]
            self._feature_slices[fi.unique_key] = (pos, pos + dim)
            pos += dim
        for fi in self._vec_fi_order:
            vec_len = self._vec_lengths[fi.unique_key]
            self._feature_slices[fi.unique_key] = (pos, pos + vec_len + 1)
            pos += vec_len + 1

    def _update_cat_vocab_sizes(self, features: dict[FieldInfo, list[float]]) -> None:
        """Track the max observed LabelIndex per categorical field for embedding sizing."""
        for fi, values in features.items():
            if fi.preprocessor in _EMBEDDING_PREPROCESSORS:
                idx = int(values[0])
                expected = int(fi.limits[1]) if fi.limits is not None else 0
                current = self._cat_vocab_sizes.get(fi.original, 0)
                self._cat_vocab_sizes[fi.original] = max(current, idx, expected)

    def _encode_features(
        self, features: dict[FieldInfo, list[float]]
    ) -> tuple[list[int], list[float]]:
        """Encode a feature dict into (cat_indices, num_floats) with missing-value imputation.

        Categorical missing → index 0 (LabelIndex missing sentinel): the model sees
        a distinct "absent" embedding, not any real category.

        All other fields → [v0, ..., vN, is_present]: vec_len values + presence flag.
        """
        cat_indices = [int(features.get(fi, [0.0])[0]) for fi in self._cat_fi_order]
        num_floats: list[float] = []
        for fi in self._vec_fi_order:
            if fi in features:
                num_floats.extend(features[fi])
                num_floats.append(1.0)
            else:
                num_floats.extend([0.0] * self._vec_lengths[fi.unique_key])
                num_floats.append(0.0)
        return cat_indices, num_floats

    def _buffer_to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert all buffered events to batched (cat_tensor, num_tensor) for training."""
        rows = [self._encode_features(d) for d in self._buffer]
        return (
            torch.tensor([r[0] for r in rows], dtype=torch.long, device=self._device),
            torch.tensor([r[1] for r in rows], dtype=torch.float32, device=self._device),
        )

    # ------------------------------------------------------------------ #
    # Scoring                                                              #
    # ------------------------------------------------------------------ #

    def _mse_to_percentile_score(self, mse: float) -> float:
        """Convert reconstruction MSE to a 0–100 anomaly score via percentile rank.

        Score = (baseline events with lower MSE / total baseline events) × 100.
        Score 70 means this event's reconstruction error exceeds 70% of the baseline.
        """
        rank = bisect.bisect_left(self._sorted_baseline_mses, mse)
        return (rank / len(self._sorted_baseline_mses)) * 100.0

    def _build_explanation(
        self,
        features: dict[FieldInfo, list[float]],
        flat: dict[str, Any],
        reconstructed: torch.Tensor,
        original_x: torch.Tensor,
        score: float,
    ) -> CohortExplanation:
        """Attribute the anomaly score proportionally to each field's squared error.

        delta = (field squared error / total squared error) × score.
        """
        per_dim_sq_error = (reconstructed - original_x) ** 2  # [1, total_dim]
        total_sq_error = float(per_dim_sq_error.sum().item())

        groups: dict[str, list[FieldInfo]] = {}
        for fi in features:
            groups.setdefault(fi.original, []).append(fi)

        contributors = []
        for field_name, fis in groups.items():
            field_sq_error = 0.0
            for fi in fis:
                s, e = self._feature_slices.get(fi.unique_key, (0, 0))
                field_sq_error += float(per_dim_sq_error[0, s:e].sum().item())
            delta = (field_sq_error / total_sq_error * score) if total_sq_error > 0 else 0.0
            contributors.append(FieldContribution(
                field=field_name,
                value=flat.get(field_name),
                delta=round(delta, 4),
                preprocessed={fi.unique_key: features[fi] for fi in fis},
            ))

        contributors.sort(key=lambda c: abs(c.delta), reverse=True)
        return CohortExplanation(features=contributors, baseline_score=0.0)

    def _resolve_bottleneck(self, vocab_sizes: list[int], numeric_dim: int) -> int:
        """Derive the bottleneck dimension, auto-computing it if not configured."""
        if self._bottleneck is not None:
            return self._bottleneck
        embed_dims = [max(2, min(self._embed_dim, v // 2 + 1)) for v in vocab_sizes]
        total_input_dim = sum(embed_dims) + numeric_dim
        return max(total_input_dim // 2, 2)

    def _percentile_to_linear(self, p: float) -> float:
        """Map AutoEncoder percentile score (uniform over [0,100] for in-distribution events)
        to the shared linear 0–100 anomaly scale used by all models.

        The AE score is uniformly distributed, so label boundaries fall at higher
        percentile values. This piecewise linear map aligns AE percentile thresholds with the canonical
        label boundaries so Detector can apply a single score_label() function.

          Percentile → Linear   Label boundary
              0  →   0
             80  →  30          SLIGHTLY_ELEVATED
             90  →  50          ELEVATED
             95  →  70          ANOMALOUS
             99  →  85          HIGHLY_ANOMALOUS
            100  → 100
        """
        breakpoints = [(0.0, 0.0), (80.0, 30.0), (90.0, 50.0), (95.0, 70.0), (99.0, 85.0), (100.0, 100.0)]
        for i in range(len(breakpoints) - 1):
            p0, l0 = breakpoints[i]
            p1, l1 = breakpoints[i + 1]
            if p <= p1:
                t = (p - p0) / (p1 - p0)
                return l0 + t * (l1 - l0)
        return 100.0

    # ------------------------------------------------------------------ #
    # Self-checks                                                          #
    # ------------------------------------------------------------------ #

    def _assert_no_unknown_features(
        self, features: dict[FieldInfo, float], context: str
    ) -> None:
        """Raise if any feature was not present when feature order was fixed at model-build time.

        Feature order is fixed once at model-build time from the buffer union.
        New fields after that point would be silently zero-imputed — this check
        turns that silent data loss into an explicit error.
        """
        unknown = [fi for fi in features if fi.unique_key not in self._known_feature_keys]
        if unknown:
            names = [fi.unique_key for fi in unknown]
            raise ValueError(
                f"{context}: received {len(unknown)} feature(s) unknown to the model "
                f"(feature order is fixed at model-build time): {names}. "
                f"Reset the model to incorporate new fields."
            )

    def _assert_cat_indices_in_bounds(self, cat_indices: list[int]) -> None:
        """Raise if any categorical index would overflow its embedding table.

        A new vocabulary value that appears between the last retrain and the current
        score call can produce an index beyond the pre-allocated headroom rows. This
        raises a clear ValueError rather than letting PyTorch throw an opaque
        IndexError. Increase vocab_headroom or decrease retrain_every to fix this.
        """
        for i, (idx, fi) in enumerate(zip(cat_indices, self._cat_fi_order)):
            emb = self._model.embeddings[i]
            if idx >= emb.num_embeddings:
                raise ValueError(
                    f"Categorical index {idx} for field '{fi.original}' exceeds the "
                    f"embedding table capacity of {emb.num_embeddings} rows "
                    f"(vocab_headroom={self._vocab_headroom} exhausted). "
                    f"Increase vocab_headroom or reduce retrain_every so the table "
                    f"is validated before new vocabulary reaches score time."
                )
