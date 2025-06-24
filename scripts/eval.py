import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval(
    dataloader: DataLoader,
    fusion_head: nn.Module,
    classifier_head: nn.Module,
    loss_fn: nn.Module,
) -> dict[str, list]:
    fusion_head.eval()
    classifier_head.eval()

    device = next(fusion_head.parameters()).device
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, labels in tqdm(dataloader, desc="Evaluating"):
            x, y, labels = x.to(device), y.to(device), labels.to(device)

            features = fusion_head(x, y)
            logits = classifier_head(features)

            # Calculate and accumulate loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)

            # Collect predictions and labels
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all collected values
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {"loss": [avg_loss], "accuracy": [accuracy], "f1": [f1]}
