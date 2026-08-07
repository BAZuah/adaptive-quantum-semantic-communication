"""Map K semantic concepts to |S| states; Fs is information-theoretic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering


@dataclass
class CompressionResult:
    compression_level: float
    num_states: int
    concept_to_state: np.ndarray
    state_centroids: np.ndarray
    semantic_fidelity: float
    geometric_fidelity: float
    information_fidelity: float


class SemanticCompressor:
    """
    Adaptive semantic representation.

    Compression level c in (0, 1] sets |S| = ceil(c * K).
    Resource demand for SeQUeNCe is |S| entangled pairs.

    Primary Fs (used in reward):
        information fidelity = I(concept; state) / H(concept)
        = 1 - H(concept | state) / H(concept)

    This drops when distinct concepts are merged, without injecting the
    raw action |S|/K into the reward as a fake fidelity term.

    geometric_fidelity is reported for analysis only.
    """

    def __init__(self, num_concepts: int, rbf_scale: float | None = None):
        if num_concepts < 1:
            raise ValueError("num_concepts must be positive")
        self.num_concepts = num_concepts
        self.rbf_scale = rbf_scale

    def target_states(self, compression_level: float) -> int:
        if not 0.0 < compression_level <= 1.0:
            raise ValueError("compression_level must be in (0, 1]")
        return max(1, int(np.ceil(self.num_concepts * compression_level)))

    def compress(
        self,
        centroids: np.ndarray,
        probabilities: np.ndarray,
        compression_level: float,
    ) -> CompressionResult:
        centroids = np.asarray(centroids, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)

        if centroids.shape[0] != self.num_concepts:
            raise ValueError("centroid count mismatch")
        if probabilities.shape != (self.num_concepts,):
            raise ValueError("probability shape mismatch")
        if probabilities.sum() <= 0:
            raise ValueError("probabilities must sum to a positive value")

        probs = probabilities / probabilities.sum()
        num_states = self.target_states(compression_level)

        if self.rbf_scale is None:
            self.rbf_scale = self._default_scale(centroids)

        if num_states == self.num_concepts:
            mapping = np.arange(self.num_concepts, dtype=np.int64)
        else:
            mapping = AgglomerativeClustering(
                n_clusters=num_states,
                linkage="ward",
            ).fit_predict(centroids).astype(np.int64)

        state_centroids = self._state_centroids(centroids, probs, mapping, num_states)
        info_fs = self._information_fidelity(probs, mapping, num_states)
        geom_fs = self._geometric_fidelity(centroids, probs, mapping, state_centroids)

        return CompressionResult(
            compression_level=float(compression_level),
            num_states=num_states,
            concept_to_state=mapping,
            state_centroids=state_centroids.astype(np.float32),
            # Reward uses information fidelity so merging concepts lowers Fs.
            # Geometric fidelity is retained for analysis.
            semantic_fidelity=info_fs,
            geometric_fidelity=geom_fs,
            information_fidelity=info_fs,
        )

    @staticmethod
    def _default_scale(centroids: np.ndarray) -> float:
        if centroids.shape[0] < 2:
            return 1.0
        diffs = centroids[:, None, :] - centroids[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        upper = dists[np.triu_indices(len(centroids), k=1)]
        return max(float(np.median(upper)), 1e-6)

    @staticmethod
    def _state_centroids(
        centroids: np.ndarray,
        probs: np.ndarray,
        mapping: np.ndarray,
        num_states: int,
    ) -> np.ndarray:
        out = []
        for s in range(num_states):
            idx = np.where(mapping == s)[0]
            if len(idx) == 0:
                raise RuntimeError(f"empty semantic state {s}")
            w = probs[idx]
            if w.sum() <= 0:
                w = np.ones(len(idx), dtype=np.float64)
            out.append(np.average(centroids[idx], axis=0, weights=w))
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    def _information_fidelity(
        self,
        probs: np.ndarray,
        mapping: np.ndarray,
        num_states: int,
    ) -> float:
        h_concept = self._entropy(probs)
        if h_concept <= 1e-12:
            return 1.0

        cond = 0.0
        for s in range(num_states):
            idx = np.where(mapping == s)[0]
            mass = float(probs[idx].sum())
            if mass <= 0:
                continue
            post = probs[idx] / mass
            cond += mass * self._entropy(post)

        return float(np.clip(1.0 - cond / h_concept, 0.0, 1.0))

    def _geometric_fidelity(
        self,
        centroids: np.ndarray,
        probs: np.ndarray,
        mapping: np.ndarray,
        state_centroids: np.ndarray,
    ) -> float:
        scale = float(self.rbf_scale)
        score = 0.0
        for i, s in enumerate(mapping):
            err = float(np.linalg.norm(centroids[i] - state_centroids[s]))
            sim = float(np.exp(-0.5 * (err / scale) ** 2))
            score += probs[i] * sim
        return float(np.clip(score, 0.0, 1.0))
