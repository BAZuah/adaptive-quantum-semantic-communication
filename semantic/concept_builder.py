"""CIFAR-10 → ResNet embeddings → K-means semantic concepts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from semantic.dataset import create_cifar10_dataloader
from semantic.feature_extractor import ResNet18FeatureExtractor


@dataclass
class ConceptSet:
    embeddings: np.ndarray
    labels: np.ndarray
    assignments: np.ndarray
    centroids: np.ndarray
    probabilities: np.ndarray

    @property
    def num_concepts(self) -> int:
        return int(self.centroids.shape[0])


class ConceptBuilder:
    """Paper-aligned classical stand-in for quantum embedding + q-means."""

    def __init__(
        self,
        num_samples: int = 2000,
        num_clusters: int = 10,
        batch_size: int = 64,
        seed: int = 42,
        data_root: str = "./data",
        device: str | None = None,
    ):
        if num_clusters > num_samples:
            raise ValueError("num_clusters cannot exceed num_samples")
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.seed = seed
        self.data_root = data_root
        self.device = device

    def build(self) -> ConceptSet:
        from config import DOWNLOAD_CIFAR

        loader = create_cifar10_dataloader(
            root=self.data_root,
            train=True,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=False,
            num_workers=0,
            download=DOWNLOAD_CIFAR,
        )
        extractor = ResNet18FeatureExtractor(device=self.device)
        embeddings, labels = extractor.extract_dataloader(
            dataloader=loader,
            normalize_embeddings=True,
        )

        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.seed,
            n_init=10,
        )
        assignments = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_.astype(np.float64)

        counts = np.bincount(assignments, minlength=self.num_clusters).astype(
            np.float64
        )
        probabilities = counts / counts.sum()

        return ConceptSet(
            embeddings=embeddings.astype(np.float32),
            labels=labels.astype(np.int64),
            assignments=assignments.astype(np.int64),
            centroids=centroids,
            probabilities=probabilities,
        )
