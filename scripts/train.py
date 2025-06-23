import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..src import UniModalDataset


def train(
    dataloader: DataLoader,
    optim: optim.Optimizer,
    loss_fn: nn.Module,
    fusion_head: nn.Module,
    classifier_head: nn.Module,
    modalities: list[UniModalDataset],
    epochs: int = 10,
):
    for epoch in range(epochs):
        for x, y, labels in tqdm(dataloader):
            optim.zero_grad()

            feature = fusion_head(x, y)
            pred = classifier_head(feature)

            loss = loss_fn(pred, labels)

            loss.backward()
            optim.step()

        fused_features = []
        with torch.no_grad():
            for x, y, _ in tqdm(dataloader):
                fused_feature = fusion_head(x, y)
                fused_features.append(fused_feature)

        new_feature = UniModalDataset(fused_features)
        modalities.append(new_feature)
