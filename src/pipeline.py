from enum import Enum

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..scripts import eval, train
from .model import Classifier, FusionMode, ModalityAwareFusion
from .multimodalDataset import MultiModalDataset
from .unimodalDataset import UniModalDataset


class Mode(Enum):
    TRAIN = 0
    EVAL = 1


class FusionPipeline:
    def __init__(
        self,
        modalities: list[UniModalDataset],
        labels: list[float],
        num_heads: list[int],
        d_model: int,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        num_classes: int,
        fusionMode: FusionMode = FusionMode.BI,
    ):
        self.modalities = modalities
        self.labels = labels
        self.num_modalities = len(self.modalities)

        if len(modalities) != len(num_heads) + 1:
            raise ValueError(
                f"number of heads must be provided for {len(modalities)} modalities."
            )

        self.d_model = d_model
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

    def fuse(self, mode: Mode = Mode.EVAL, batch_size: int = 64, fuser: int = 0):
        # Get two modalties
        if len(self.modalities) >= 2:
            X = self.modalities.pop()
            Y = self.modalities.pop()

            # So now we create a multimodal dataset that encapsulate the two modalties
            # Then we make a dataloader of it
            dataset = MultiModalDataset(X, Y, self.labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            if mode == Mode.EVAL:
                eval(loader, self.fusion_heads[fuser], self.classifier_head)
            else:
                train(
                    loader,
                    self.optimizer,
                    self.loss_fn,
                    self.fusion_heads[fuser],
                    self.classifier_head,
                    self.modalities,
                )
            # Recurse with next fusion head
            return self.fuse(mode, batch_size, fuser + 1)

        return
