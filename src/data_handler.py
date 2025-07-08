import os
import pickle
from typing import Dict, List, NamedTuple

import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ModalityData(NamedTuple):
    features: torch.Tensor
    lengths: torch.Tensor
    name: str


class MOSIMultimodalDataset(Dataset):
    def __init__(self, modalities: List[ModalityData], labels: torch.Tensor):
        self.modalities = {mod.name: mod for mod in modalities}
        self.labels = labels
        self.modality_names = sorted(self.modalities.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        # Returns a dictionary for clarity, to be ordered by the collate_fn
        data = {
            name: (
                self.modalities[name].features[idx],
                self.modalities[name].lengths[idx],
            )
            for name in self.modality_names
        }
        data["labels"] = self.labels[idx]
        return data


def _setup_data(data_dir: str, gdrive_id: str):
    print("--- Setting up dataset ---")
    output_filename = "mosi_data.pkl"
    data_file_path = os.path.join(data_dir, output_filename)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(data_file_path):
        print(f"Dataset not found. Downloading to '{data_dir}'...")
        gdown.download(id=gdrive_id, output=data_file_path, quiet=False)
    else:
        print("Dataset already exists.")
    return data_file_path


def get_dataloaders(config: Dict):
    data_path = _setup_data(config["data"]["data_dir"], config["data"]["gdrive_id"])
    num_classes = config["num_classes"]

    print(f"Loading data from {data_path} for REGRESSION ({num_classes}-class eval)...")
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    loaders = {}
    all_modality_info = {}
    modality_names = ["text", "audio", "vision"]

    for split in ["train", "valid", "test"]:
        raw_labels = data[split]["labels"].squeeze()
        labels = torch.tensor(
            np.round(raw_labels) if num_classes == 7 else raw_labels,
            dtype=torch.float32,
        )

        modalities_list = [
            ModalityData(
                torch.tensor(data[split][name], dtype=torch.float32),
                torch.full((len(labels),), data[split][name].shape[1]),
                name,
            )
            for name in modality_names
        ]

        if split == "train":
            all_modality_info = {m.name: m for m in modalities_list}

        dataset = MOSIMultimodalDataset(modalities_list, labels)

        # The collate function handles the reordering based on the config['order']
        def collate_fn(batch):
            order = config["order"]
            # Reorder modalities according to config['order']
            ordered_features = []
            for mod_name in order:
                mod_features = torch.stack([item[mod_name][0] for item in batch])
                mod_lengths = torch.stack([item[mod_name][1] for item in batch])
                ordered_features.extend([mod_features, mod_lengths])

            labels = torch.stack([item["labels"] for item in batch])
            return tuple(ordered_features) + (labels,)

        loaders[split] = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=0,  # Can be increased for performance
            collate_fn=collate_fn,
        )
        print(f"Loaded '{split}' split with {len(labels)} samples.")

    return loaders["train"], loaders["valid"], loaders["test"], all_modality_info
