from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_cifar10_transform():
    """
    Build the preprocessing transform used before feature extraction.

    CIFAR-10 images are originally 32 x 32 pixels. They are resized to
    224 x 224 pixels because the pretrained ResNet-18 feature extractor
    expects ImageNet-style inputs.
    """

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_cifar10_dataset(
    root: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = 2000,
    seed: int = 42,
    download: bool = True,
):
    """
    Load CIFAR-10 and optionally select a reproducible random subset.

    Parameters
    ----------
    root:
        Directory in which CIFAR-10 is stored or downloaded.

    train:
        If True, load the CIFAR-10 training set.
        If False, load the CIFAR-10 test set.

    num_samples:
        Number of samples to keep.

        If None, use the entire dataset.

    seed:
        Random seed used for reproducible subset selection.

    download:
        If True, download CIFAR-10 when it is not already available.

    Returns
    -------
    dataset:
        Full CIFAR-10 dataset or a reproducible subset.
    """

    data_root = Path(root)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(
        root=str(data_root),
        train=train,
        transform=build_cifar10_transform(),
        download=download,
    )

    if num_samples is None:
        return dataset

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")

    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} samples, but the dataset "
            f"contains only {len(dataset)} samples."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:num_samples].tolist()

    return Subset(
        dataset,
        selected_indices,
    )


def create_cifar10_dataloader(
    root: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = 2000,
    batch_size: int = 64,
    seed: int = 42,
    shuffle: bool = False,
    num_workers: int = 2,
    download: bool = True,
):
    """
    Create a DataLoader for CIFAR-10 feature extraction.

    Parameters
    ----------
    root:
        CIFAR-10 storage directory.

    train:
        Select training or test split.

    num_samples:
        Number of images to use.

    batch_size:
        Number of images processed per batch.

    seed:
        Seed for reproducible subset selection and shuffling.

    shuffle:
        Whether to shuffle the selected samples.

        For deterministic feature extraction, False is recommended.

    num_workers:
        Number of DataLoader worker processes.

        Use 0 if multiprocessing causes problems on the current system.

    download:
        Download the dataset if necessary.

    Returns
    -------
    dataloader:
        PyTorch DataLoader containing images and class labels.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    dataset = load_cifar10_dataset(
        root=root,
        train=train,
        num_samples=num_samples,
        seed=seed,
        download=download,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def get_cifar10_class_name(label: int) -> str:
    """
    Convert an integer CIFAR-10 label into its class name.
    """

    if not 0 <= label < len(CIFAR10_CLASS_NAMES):
        raise ValueError(
            f"Invalid CIFAR-10 label {label}. "
            f"Expected a value between 0 and 9."
        )

    return CIFAR10_CLASS_NAMES[label]