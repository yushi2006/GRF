from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Classifier, FusionMode, ModalityAwareFusion
from .multimodalDataset import MultiModalDataset


class Mode(Enum):
    TRAIN = 0
    EVAL = 1


class FusionPipeline:
    def __init__(
        self,
        modalities: list[list[torch.Tensor]],
        labels: list[float],
        num_heads: list[int],
        d_model: int,
        unimodal_encoders,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        num_classes: int,
        mode: Mode = Mode.EVAL,
        fusionMode: FusionMode = FusionMode.BI,
    ):
        self.modalities = modalities
        self.labels = labels
        self.num_modalities = len(self.modalities)

        self.modality_encoding()

        if len(modalities) != len(num_heads) + 1:
            raise ValueError(
                f"number of heads must be provided for {len(modalities)} modalities."
            )

        self.unimodal_encoders = unimodal_encoders

        self.d_model = d_model
        self.mode = mode
        self.fusion_heads = nn.ModuleList(
            [
                ModalityAwareFusion(
                    d_model,
                    self.num_modalities,
                    num_heads[i],
                    device="cpu",
                    mode=fusionMode,
                )
                for i in range(len(num_heads))
            ]
        )
        self.classifier_head = Classifier(d_model, num_classes=num_classes)

        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def modality_encoding(self):
        for i in range(self.num_modalities):
            self.modalities[i] = self.unimodal_encoders[i](self.modalities[i])

    def fuse(self, batch_size: int = 64, fuser: int = 0):
        # Here we need to do recursive modality fusion.
        # We will choose two different modalities and do modality fusion on it.
        # We can then remove the two modalities and replace them with the new modality.
        # and we just recursively call the function again on the new list.
        # This function should have two implemenations, one for eval and one for training.
        # So in training we do the training mode in pytorch and do a for loop on the batch
        # In eval we should run the eval thing for pytorch.
        # if we finished this pipeline correctly we can start experiementing the idea and then publish the results

        # Get two modalties
        if len(self.modalities) >= 2:
            X = self.modalities.pop()
            Y = self.modalities.pop()

            # So now we create a multimodal dataset that encapsulate the two modalties
            # Then we make a dataloader of it
            dataset = MultiModalDataset(X, Y, self.labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for x, y, labels in tqdm(loader):
                self.optimizer.zero_grad()

                feature = self.fusion_heads[fuser](x, y)
                pred = self.classifier_head(feature)

                loss = self.loss_fn(pred, labels)

                loss.backward()
                self.optimizer.step()

            fused_features = []
            eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            with torch.no_grad():
                for x, y, _ in eval_loader:
                    features = self.fusion_heads[fuser](x, y)
                    fused_features.append(features)

            # Append new modality to list
            self.modalities.append(fused_features)

            # Recurse with next fusion head
            return self.fuse(batch_size, fuser + 1)

        return
