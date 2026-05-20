from __future__ import annotations

from dataclasses import replace as _replace
from typing import Any

from app.features.preprocessors.base import FieldInfo
from app.features.preprocessors.hash_index import HashIndex
from app.features.preprocessors.label_index import LabelIndex


class Drain3Encoder:
    """Parses free-form log text with the Drain3 online template miner and encodes
    the result as two kinds of sub-fields.

    Output fields (given a source field named ``key``):

    ``<key>__template``
        Integer index of the matched Drain3 cluster, encoded via a shared
        ``preprocessor="LabelIndex"``
        An unseen template at score time returns no entry

    ``<key>__var_0`` … ``<key>__var_{max_vars-1}``
        Hash-bucketed variable values extracted from the matched template, encoded
        via a shared ``HashIndex`` instance.  Routed to the AE numeric pathway
        (``preprocessor="HashIndex"``).  Slots beyond the number of variables in
        the matched template are absent

    Drain3 tree learning happens in ``learn_pre_transform`` so that the full event
    batch is incorporated before any ``transform`` call resolves cluster IDs via
    ``match``.  At score time (``learn_pre_transform`` not called) ``match`` is
    used without mutating the tree; an unrecognised log line returns ``{}``.

    LIMITATION — variable slots are positional and cross-template:
        ``var_0`` is always the *first* wildcard of whatever template matched.
        For template "Failed login for user <*> from <*>" ``var_0`` is the
        username; for template "Service <*> restarted" ``var_0`` is the service
        name.

    Parameters
    ----------
    sim_th
        Drain3 similarity threshold (0–1).  Higher values create more specific
        templates (more clusters); lower values merge aggressively.
    max_clusters
        Upper bound on the number of Drain3 clusters.  Also used as the
        ``expected_categories`` hint for ``LabelIndex``
    depth
        Drain3 parse-tree depth.  Deeper trees produce finer-grained templates
        at the cost of more memory.
    max_vars
        Maximum number of variable slots emitted per event.  Variables beyond
        this index are silently dropped.
    n_hash_features
        Bucket count for the ``HashIndex`` used to encode variable values.
    seed
        Seed forwarded to ``HashIndex`` for deterministic hashing across restarts.
    """

    def __init__(
        self,
        sim_th: float = 0.4,
        max_clusters: int = 200,
        depth: int = 4,
        max_vars: int = 20,
        n_hash_features: int = 2_000,
        seed: int = 0,
    ) -> None:
        import logging

        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        logging.getLogger("drain3.template_miner").setLevel(logging.WARNING)

        cfg = TemplateMinerConfig()
        cfg.drain_sim_th = sim_th
        cfg.drain_depth = depth
        cfg.drain_max_clusters = max_clusters

        self._drain = TemplateMiner(config=cfg)
        self._label_index = LabelIndex(expected_categories=max_clusters)
        self._hash_index = HashIndex(n_features=n_hash_features, seed=seed)
        self._max_vars = max_vars

    def learn_pre_transform(self, key: str, value: Any) -> None:
        if not value or not str(value).strip():
            return
        self._drain.add_log_message(str(value))

    def learn_post_transform(self, key: str, value: Any) -> None:
        pass

    def transform(self, key: str, value: Any) -> dict[FieldInfo, list[float]]:
        if not value or not str(value).strip():
            return {}

        s = str(value)
        cluster = self._drain.match(s)
        if cluster is None:
            # Unseen template at score time — emit nothing
            return {}

        result: dict[FieldInfo, list[float]] = {}

        template_str = cluster.get_template()
        for fi, val in self._label_index.transform(key, str(cluster.cluster_id)).items():
            result[_replace(fi, bucket="template", raw=template_str)] = val

        params = self._drain.get_parameter_list(template_str, s)
        for i, param in enumerate(params[: self._max_vars]):
            for fi, val in self._hash_index.transform(key, param).items():
                result[_replace(fi, bucket=f"var_{i}", raw=param)] = val

        return result
