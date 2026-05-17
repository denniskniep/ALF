from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np

from app.features.preprocessors.base import FieldInfo

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Shared model cache — keyed by (repo_id, resolved_cache_dir).
# SentenceTransformer inference is read-only, so one instance is safe to share
# across all cohorts and threads.
_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}


class SentenceTransformerEncoder:
    """Encodes a text field into a fixed-size dense float vector using a pre-trained
    sentence transformer, projected down to embed_dim dimensions via a fixed random
    projection matrix.

    The sentence transformer is pre-trained and frozen — no training or fine-tuning
    happens here. The random projection reduces the model's native embedding dimension
    (384 for all-MiniLM-L6-v2) to embed_dim using a fixed seed, preserving approximate
    distances (Johnson-Lindenstrauss lemma).

    Keep embed_dim comparable to other related embed_dim
    (e.g. categorical fields in Autoencoder default 8–32) so the text
    field does not dominate the AE input vector and gradient signal.

    Model caching:
        The model is downloaded once to cache_dir and reused on subsequent runs.
        cache_dir can be absolute or relative to the repository root.

    Absent / empty values:
        Returns an empty dict

    Potential optimization — PCA projection:
        The fixed random projection preserves approximate distances but does not focus
        on the directions of maximum variance for your domain's text. An alternative is
        to accumulate N embeddings during warm-up, fit PCA on them, and use the top
        embed_dim principal components as the projection matrix. This yields better
        variance coverage for domain-specific vocabulary at the cost of a stateful
        warm-up period (field is treated as absent until PCA is fitted). For most use
        cases the random projection is sufficient.

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
        projection — consistent feature values across restarts.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        embed_dim: int = 8,
        cache_dir: str = ".local/modelcache",
        seed: int = 42,
    ) -> None:
        from huggingface_hub import try_to_load_from_cache
        from sentence_transformers import SentenceTransformer

        resolved = Path(cache_dir) if Path(cache_dir).is_absolute() else _REPO_ROOT / cache_dir
        resolved.mkdir(parents=True, exist_ok=True)

        import torch

        model_repo_id = model if "/" in model else f"sentence-transformers/{model}"
        if model_repo_id not in _MODEL_CACHE:
            cached = try_to_load_from_cache(repo_id=model_repo_id, filename="modules.json", cache_dir=str(resolved))
            local_files_only = isinstance(cached, str)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL_CACHE[model_repo_id] = SentenceTransformer(
                model,
                cache_folder=str(resolved),
                local_files_only=local_files_only,
                device=device,
            )
        self._st_model = _MODEL_CACHE[model_repo_id]
        self._embed_dim = embed_dim

        source_dim = self._st_model.get_embedding_dimension()
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((source_dim, embed_dim))
        projection /= np.linalg.norm(projection, axis=0)
        self._projection: np.ndarray = projection

    def learn_pre_transform(self, key: str, value: Any) -> None:
        pass

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, list[float]]:
        if not value or not str(value).strip():
            return {}
        embedding = self._st_model.encode(str(value), convert_to_numpy=True)
        projected: list[float] = (embedding @ self._projection).tolist()
        return {FieldInfo(original=key, preprocessor="SentenceTransformerEncoder"): projected}
