import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval(
    dataloader: DataLoader,
    fusion_head: nn.Module,
    classifier_head: nn.Module,
):
    fusion_head.eval()
    classifier_head.eval()

    with torch.no_grad():
        for x, y, _ in tqdm(dataloader):
            features = fusion_head(x, y)
            logits = classifier_head(features)

        predicted_classes = torch.argmax(logits, dim=1).item()  # type: ignore

        return predicted_classes
