import json
from typing import Dict, Optional

import mlflow
import numpy as np
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from thop import profile


def load_config(path: str) -> Dict:
    """Loads a YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def log_dict_as_json(data: Dict, filename: str, artifact_path: Optional[str] = None):
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    with open(filename, "w") as f:
        json.dump(data, f, indent=4, default=convert)
    if mlflow.active_run():
        mlflow.log_artifact(filename, artifact_path)


def profile_model(model: nn.Module, sample_batch: tuple):
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cpu_inputs = tuple(t.cpu() for t in sample_batch[:-1])  # thop works on CPU
    macs, _ = profile(model, inputs=cpu_inputs, verbose=False)
    metrics = {"parameters_M": num_params / 1e6, "gflops": (macs * 2) / 1e9}
    print(
        f"--- Profiling: {metrics['parameters_M']:.2f} M Params, {metrics['gflops']:.2f} GFLOPs ---"
    )
    return metrics


def calculate_regression_metrics(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> Dict[str, float]:
    preds, labels = preds.squeeze(), labels.squeeze()
    mae = np.mean(np.abs(preds - labels))
    corr = (
        np.corrcoef(preds, labels)[0, 1]
        if np.std(preds) > 0 and np.std(labels) > 0
        else 0.0
    )

    if num_classes == 2:
        preds_class, labels_class = (preds >= 0).astype(int), (labels >= 0).astype(int)
        acc_key = "acc2"
    else:
        preds_class, labels_class = (
            np.round(preds).astype(int),
            np.round(labels).astype(int),
        )
        min_label, max_label = np.min(labels_class), np.max(labels_class)
        preds_class = np.clip(preds_class, min_label, max_label)
        acc_key = f"acc{num_classes}"

    accuracy = np.mean(preds_class == labels_class)
    f1 = f1_score(labels_class, preds_class, average="macro", zero_division=0)
    return {"mae": mae, "corr": corr, acc_key: accuracy, "f1_macro": f1}
