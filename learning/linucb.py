"""Disjoint LinUCB contextual bandit."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LinUCB:
    """
    Online contextual MAB: each arm is a compression level.

    Training uses select_arm() + update(chosen_arm only).
    Evaluation uses select_greedy_arm() with no updates.
    """

    def __init__(self, arms, context_dim: int, alpha: float = 1.0):
        self.arms = list(arms)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.num_arms = len(self.arms)

        if self.num_arms == 0:
            raise ValueError("LinUCB requires at least one arm")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.A = [np.eye(self.context_dim) for _ in range(self.num_arms)]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.num_arms)]

    def _x(self, context) -> np.ndarray:
        x = np.asarray(context, dtype=float).reshape(-1, 1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context dim mismatch: expected {self.context_dim}, got {x.shape[0]}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def select_arm(self, context):
        x = self._x(context)
        scores = []
        for i in range(self.num_arms):
            a_inv = np.linalg.inv(self.A[i])
            theta = a_inv @ self.b[i]
            exploit = float((theta.T @ x).item())
            explore = self.alpha * np.sqrt(max(0.0, float((x.T @ a_inv @ x).item())))
            scores.append(exploit + explore)
        idx = int(np.argmax(scores))
        return idx, self.arms[idx]

    def select_greedy_arm(self, context):
        values = self.get_estimated_values(context)
        idx = int(np.argmax(values))
        return idx, self.arms[idx]

    def get_estimated_values(self, context) -> list[float]:
        x = self._x(context)
        values = []
        for i in range(self.num_arms):
            a_inv = np.linalg.inv(self.A[i])
            theta = a_inv @ self.b[i]
            values.append(float((theta.T @ x).item()))
        return values

    def update(self, arm_index: int, context, reward: float):
        idx = int(arm_index)
        if not 0 <= idx < self.num_arms:
            raise IndexError(f"invalid arm index {idx}")
        x = self._x(context)
        r = float(reward)
        if not np.isfinite(r):
            raise ValueError("reward must be finite")
        self.A[idx] = self.A[idx] + (x @ x.T)
        self.b[idx] = self.b[idx] + (r * x)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            arms=np.asarray(self.arms, dtype=float),
            context_dim=np.asarray([self.context_dim]),
            alpha=np.asarray([self.alpha]),
            A=np.stack(self.A),
            b=np.stack(self.b),
        )

    @classmethod
    def load(cls, path) -> "LinUCB":
        data = np.load(path, allow_pickle=False)
        agent = cls(
            arms=data["arms"].tolist(),
            context_dim=int(data["context_dim"][0]),
            alpha=float(data["alpha"][0]),
        )
        agent.A = [np.asarray(a, dtype=float) for a in data["A"]]
        agent.b = [np.asarray(b, dtype=float).reshape(-1, 1) for b in data["b"]]
        return agent
