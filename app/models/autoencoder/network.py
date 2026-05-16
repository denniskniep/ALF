from __future__ import annotations

import torch
import torch.nn as nn


class EntityEmbeddingAutoencoder(nn.Module):
    """Autoencoder with per-field entity embedding tables for categorical features.

    Categorical fields are looked up in separate embedding tables, concatenated with
    numeric features, then compressed through encoder → bottleneck → decoder.
    Reconstruction error (MSE) is the anomaly signal.
    """

    def __init__(
        self,
        vocab_sizes: list[int],
        numeric_dim: int,
        embed_dim: int = 8,
        bottleneck: int = 16,
        vocab_headroom: int = 50,
    ) -> None:
        super().__init__()

        # embed_dim is a global cap applied equally to all fields. This matters: if
        # different fields had different caps, higher-cap fields would get more dims in
        # the concatenated input vector and implicitly carry more weight in the anomaly
        # signal. Below the cap, small-vocab fields get fewer dims (vocab_size // 2 + 1)
        # since they have less cardinality to represent — that scaling is bounded by
        # actual cardinality, not an arbitrary choice.
        self.embed_dims = [max(2, min(embed_dim, v // 2 + 1)) for v in vocab_sizes]
        self.numeric_dim = numeric_dim
        self.bottleneck_dim = bottleneck

        # One embedding table per categorical field; vocab_headroom extra rows
        # pre-allocated for vocabulary that arrives after model-build time.
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=v + vocab_headroom, embedding_dim=d)
            for v, d in zip(vocab_sizes, self.embed_dims)
        ])

        total_input_dim = sum(self.embed_dims) + numeric_dim
        if total_input_dim == 0:
            raise ValueError("AutoencoderDetector requires at least one feature")

        hidden = max(total_input_dim * 2, 16)
        # Cap: bottleneck cannot exceed the input — compressing to more dims than
        # the input has would be expansion, not compression.
        actual_bottleneck = min(bottleneck, total_input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, actual_bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(actual_bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, total_input_dim),
        )

    def forward(
        self,
        cat_indices: torch.Tensor,  # [batch, n_cat]
        num_floats: torch.Tensor,   # [batch, numeric_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Embed each categorical field and concatenate with numeric features.
        parts: list[torch.Tensor] = []
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(cat_indices[:, i]))  # calls Embedding.forward() via nn.Module.__call__
        if self.numeric_dim > 0:
            parts.append(num_floats)
        x = torch.cat(parts, dim=1)

        # Encode to bottleneck, then decode back to input space.
        # Return both reconstructed and original x; MSE between them is the anomaly signal.
        reconstructed = self.decoder(self.encoder(x))  # calls Sequential.forward() via nn.Module.__call__
        return reconstructed, x
