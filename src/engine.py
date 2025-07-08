import numpy as np
import torch
import torch.nn as nn
import tqdm

from .utils import calculate_regression_metrics


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    is_training: bool,
    config: dict,
):
    model.train(is_training)
    total_loss, all_preds, all_labels = 0.0, [], []
    desc = "Training" if is_training else "Evaluating"

    for batch in tqdm.tqdm(dataloader, desc=desc, leave=False):
        labels = batch[-1].to(device).unsqueeze(1)
        inputs = [t.to(device) for t in batch[:-1]]

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            main_preds, aux_preds = model(*inputs)
            main_loss = loss_fn(main_preds, labels)
            aux_loss = loss_fn(aux_preds, labels)
            loss = main_loss + config["model"]["aux_loss_weight"] * aux_loss

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["training"]["max_grad_norm"]
            )
            optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.append(main_preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = calculate_regression_metrics(all_preds, all_labels, config["num_classes"])
    metrics["loss"] = total_loss / len(all_labels)
    return metrics
