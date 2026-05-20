from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils import get_device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np

from app.features.preprocessors.base import FieldInfo

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

@contextlib.contextmanager
def _silence_stderr():
    """Redirect stderr to devnull — suppresses tqdm bars regardless of tqdm version."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield


# Shared model cache — keyed by repo_id.
# SentenceTransformer inference is read-only, so one instance is safe to share
# across all cohorts and threads.
_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}


class SentenceTransformerEncoder:
    """Encodes a text field into a fixed-size dense float vector using a pre-trained
    sentence transformer, projected down to embed_dim dimensions.

    Two projection modes:

    Random projection (pca_warmup=0, default):
        A fixed matrix seeded by `seed` reduces the native embedding dimension to
        embed_dim. Stateless — no warmup required. Preserves approximate distances
        (Johnson-Lindenstrauss lemma) but does not focus on directions of maximum
        variance for your domain's vocabulary.

    PCA projection (pca_warmup=N, N >= embed_dim):
        learn_pre_transform accumulates N raw embeddings, then fits PCA via SVD and
        stores the top embed_dim principal components as the projection matrix.
        The field is treated as absent (returns {}) until PCA is fitted, so the model
        does not see the field for the first N events. Yields better variance coverage
        for domain-specific vocabulary at the cost of a stateful warmup period.

    Keep embed_dim comparable to other embedding fields (e.g. 8–32)

    Model caching:
        The model is downloaded once to cache_dir and reused on subsequent runs.
        cache_dir can be absolute or relative to the repository root.

    Absent / empty values:
        Returns an empty dict.

    Parameters
    ----------
    model
        HuggingFace model name. Defaults to all-MiniLM-L6-v2 (384-dim, fast, good quality).
    embed_dim
        Output dimensionality after projection.
    cache_dir
        Directory for the downloaded model. Relative paths are resolved from the
        repository root. Defaults to .local/modelcache/.
    seed
        Seed for the random projection matrix. Same seed always produces the same
        projection — consistent feature values across restarts. Ignored when
        warmup_count > 0.
    warmup_count
        Injected automatically from the cohort's warmup_count — not a user-facing
        config parameter. 0 uses the fixed random projection. Must be >= embed_dim
        when > 0.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        embed_dim: int = 8,
        cache_dir: str = ".local/modelcache",
        seed: int = 42,
        warmup_count: int = 0,
    ) -> None:
        if warmup_count > 0 and warmup_count < embed_dim:
            raise ValueError(
                f"warmup_count ({warmup_count}) must be >= embed_dim ({embed_dim}) "
                "so SVD can produce enough principal components."
            )

        from huggingface_hub import try_to_load_from_cache
        from sentence_transformers import SentenceTransformer

        resolved = Path(cache_dir) if Path(cache_dir).is_absolute() else _REPO_ROOT / cache_dir
        resolved.mkdir(parents=True, exist_ok=True)

        model_repo_id = model if "/" in model else f"sentence-transformers/{model}"
        if model_repo_id not in _MODEL_CACHE:
            cached = try_to_load_from_cache(repo_id=model_repo_id, filename="modules.json", cache_dir=str(resolved))
            local_files_only = isinstance(cached, str)
            device = get_device()
            for _noisy_logger in ("sentence_transformers", "transformers", "huggingface_hub"):
                logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
            with _silence_stderr():
                _MODEL_CACHE[model_repo_id] = SentenceTransformer(
                    model,
                    cache_folder=str(resolved),
                    local_files_only=local_files_only,
                    device=device,
                )
        self._st_model = _MODEL_CACHE[model_repo_id]
        self._embed_dim = embed_dim
        self._pca_warmup = warmup_count

        if warmup_count > 0:
            self._projection: np.ndarray | None = None
            self._pca_buffer: list[np.ndarray] = []
        else:
            source_dim = self._st_model.get_embedding_dimension()
            rng = np.random.default_rng(seed)
            projection = rng.standard_normal((source_dim, embed_dim))
            projection /= np.linalg.norm(projection, axis=0)
            self._projection = projection
            self._pca_buffer = []

    def learn_pre_transform(self, key: str, value: Any) -> None:
        if self._pca_warmup == 0 or self._projection is not None:
            return
        if not value or not str(value).strip():
            return
        embedding = self._st_model.encode(str(value), convert_to_numpy=True, show_progress_bar=False)
        self._pca_buffer.append(embedding)
        if len(self._pca_buffer) >= self._pca_warmup:
            self._fit_pca()

    def _fit_pca(self) -> None:
        X = np.stack(self._pca_buffer).astype(np.float64)
        X -= X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self._projection = Vt[: self._embed_dim].T  # (source_dim, embed_dim)
        self._pca_buffer.clear()

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, list[float]]:
        if self._projection is None:
            return {}
        if not value or not str(value).strip():
            return {}
        embedding = self._st_model.encode(str(value), convert_to_numpy=True, show_progress_bar=False)
        projected: list[float] = (embedding @ self._projection).tolist()
        return {FieldInfo(original=key, preprocessor="SentenceTransformerEncoder"): projected}
