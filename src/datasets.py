import os
import pickle
from itertools import chain
from typing import List, NamedTuple

import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ModalityData(NamedTuple):
    features: torch.Tensor
    lengths: torch.Tensor
    name: str


class MultimodalDataset(Dataset):
    def __init__(self, modalities: List[ModalityData], labels: torch.Tensor):
        self.modalities = modalities
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return tuple(
            chain(*[(mod.features[idx], mod.lengths[idx]) for mod in self.modalities])
        ) + (self.labels[idx],)


def setup_data():
    print("--- 1. Setting up dataset (Original Version) ---")
    file_id = "1osjEm6mq2Aa9TBPsvprYA995n0DgPipl"
    data_dir = "data"
    output_filename = "mosi_data.pkl"
    data_file_path = os.path.join(data_dir, output_filename)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(data_file_path):
        print(f"Dataset not found. Downloading '{output_filename}' to '{data_dir}'...")
        gdown.download(id=file_id, output=data_file_path, quiet=False)
    else:
        print("Dataset already exists.")
    print("--- Dataset setup complete ---\n")
    return data_file_path


def load_mosi_data_regression(data_path: str, num_classes: int):
    print(f"Loading data from {data_path} for REGRESSION ({num_classes}-class eval)...")
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    processed_data = {}
    for split in ["train", "valid", "test"]:
        raw_labels = data[split]["labels"].squeeze()
        if num_classes == 7:
            labels = torch.tensor(np.round(raw_labels), dtype=torch.float32)
        else:
            labels = torch.tensor(raw_labels, dtype=torch.float32)
        modalities = {
            "text": torch.tensor(data[split]["text"], dtype=torch.float32),
            "audio": torch.tensor(data[split]["audio"], dtype=torch.float32),
            "vision": torch.tensor(data[split]["vision"], dtype=torch.float32),
        }
        processed_data[split] = {
            "labels": labels,
            "modalities": {
                name: ModalityData(
                    features, torch.full((len(features),), features.shape[1]), name
                )
                for name, features in modalities.items()
            },
        }
        print(f"Loaded '{split}' split with {len(labels)} samples.")
    return processed_data


def create_dataloaders(all_data: dict, order_keys: list, batch_size: int):
    """Creates DataLoaders for train, valid, and test splits."""
    initial_modalities_train = [all_data["train"]["modalities"][k] for k in order_keys]

    train_dataset = MultimodalDataset(
        initial_modalities_train, all_data["train"]["labels"]
    )
    valid_dataset = MultimodalDataset(
        [all_data["valid"]["modalities"][k] for k in order_keys],
        all_data["valid"]["labels"],
    )
    test_dataset = MultimodalDataset(
        [all_data["test"]["modalities"][k] for k in order_keys],
        all_data["test"]["labels"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, initial_modalities_train
