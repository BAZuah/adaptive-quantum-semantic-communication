from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


class ResNet18FeatureExtractor:
    """
    Extract 512-dimensional semantic feature embeddings from images
    using an ImageNet-pretrained ResNet-18 model.

    The final classification layer is removed, so the output of the
    global average pooling layer is used as the feature embedding.
    """

    def __init__(
        self,
        device: str | None = None,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        # Remove the final fully connected classification layer.
        self.model = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = 512

    @torch.no_grad()
    def extract_batch(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract embeddings for one image batch.

        Parameters
        ----------
        images:
            Tensor with shape (batch_size, 3, 224, 224).

        Returns
        -------
        embeddings:
            Tensor with shape (batch_size, 512).
        """

        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape "
                "(batch_size, channels, height, width)."
            )

        images = images.to(
            self.device,
            non_blocking=True,
        )

        features = self.model(images)

        # ResNet output shape is (batch_size, 512, 1, 1).
        embeddings = torch.flatten(
            features,
            start_dim=1,
        )

        return embeddings

    def extract_dataloader(
        self,
        dataloader: DataLoader,
        normalize_embeddings: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and labels for an entire DataLoader.

        Parameters
        ----------
        dataloader:
            DataLoader yielding image and label batches.

        normalize_embeddings:
            If True, apply L2 normalization to each feature vector.

        Returns
        -------
        embeddings:
            NumPy array with shape (num_samples, 512).

        labels:
            NumPy array with shape (num_samples,).
        """

        embedding_batches = []
        label_batches = []

        for images, labels in dataloader:
            embeddings = self.extract_batch(images)

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                )

            embedding_batches.append(
                embeddings.cpu()
            )

            label_batches.append(
                labels.cpu()
            )

        if not embedding_batches:
            raise ValueError(
                "The DataLoader did not contain any samples."
            )

        all_embeddings = torch.cat(
            embedding_batches,
            dim=0,
        )

        all_labels = torch.cat(
            label_batches,
            dim=0,
        )

        return (
            all_embeddings.numpy().astype(
                np.float32
            ),
            all_labels.numpy().astype(
                np.int64
            ),
        )