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
) -> dict[str, list]:
    epoch_losses = []
    epoch_accuracies = []
    epoch_f1s = []

    device = next(fusion_head.parameters()).device  # Get device from model

    for epoch in range(epochs):
        fusion_head.train()
        classifier_head.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Training loop
        for x, y, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, y, labels = x.to(device), y.to(device), labels.to(device)
            optim.zero_grad()

            feature = fusion_head(x, y)
            pred = classifier_head(feature)
            loss = loss_fn(pred, labels)

            loss.backward()
            optim.step()

            # Accumulate loss
            total_loss += loss.item() * len(labels)

            # Collect predictions and true labels
            pred_labels = torch.argmax(pred.detach(), dim=1)
            all_preds.append(pred_labels.cpu())
            all_labels.append(labels.cpu())

        # Compute metrics for the epoch
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        n_samples = len(all_labels)

        # Average loss
        avg_loss = total_loss / n_samples
        epoch_losses.append(avg_loss)

        # Accuracy
        correct = (all_preds == all_labels).sum().item()
        accuracy = correct / n_samples
        epoch_accuracies.append(accuracy)

        # Macro F1 Score
        classes = torch.unique(all_labels).tolist()
        f1_scores = []
        for c in classes:
            tp = ((all_preds == c) & (all_labels == c)).sum().item()
            fp = ((all_preds == c) & (all_labels != c)).sum().item()
            fn = ((all_preds != c) & (all_labels == c)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores)
        epoch_f1s.append(macro_f1)

        # Generate fused features for new modality
        fused_features = []
        fusion_head.eval()
        with torch.no_grad():
            for x, y, _ in tqdm(dataloader, desc="Generating fused features"):
                x, y = x.to(device), y.to(device)
                fused_feature = fusion_head(x, y)
                fused_features.append(fused_feature.cpu())

        new_feature = UniModalDataset(fused_features)
        modalities.append(new_feature)

    return {"loss": epoch_losses, "accuracy": epoch_accuracies, "f1": epoch_f1s}
