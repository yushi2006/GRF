import time
from enum import Enum

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..scripts import eval, train
from .model import Classifier, Fuser, FusionMode
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
        self.num_heads = num_heads  # Store for MLflow logging
        self.fusionMode = fusionMode  # Store for MLflow logging

        if len(modalities) != len(num_heads) + 1:
            raise ValueError(
                f"number of heads must be provided for {len(modalities)} modalities."
            )

        self.d_model = d_model
        self.fusion_heads = nn.ModuleList(
            [
                Fuser(
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
        # Start MLflow parent run only at the first fusion step
        if fuser == 0:
            mlflow.start_run()
            mlflow.log_params(
                {
                    "d_model": self.d_model,
                    "num_classes": self.classifier_head.num_classes,
                    "fusion_mode": self.fusionMode.name,
                    "optimizer": type(self.optimizer).__name__,
                    "loss_fn": type(self.loss_fn).__name__,
                    "total_modalities": self.num_modalities,
                    "num_heads": str(self.num_heads),
                }
            )

        # Get two modalties
        if len(self.modalities) >= 2:
            X = self.modalities.pop()
            Y = self.modalities.pop()

            dataset = MultiModalDataset(X, Y, self.labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Start nested MLflow run for this fusion step
            with mlflow.start_run(run_name=f"fusion_step_{fuser}", nested=True):
                mlflow.log_params(
                    {
                        "fusion_step": fuser,
                        "current_heads": self.num_heads[fuser],
                        "batch_size": batch_size,
                        "remaining_modalities": len(self.modalities),
                    }
                )

                # Reset CUDA memory stats and start timer
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                start_time = time.time()

                # Execute training/evaluation
                if mode == Mode.EVAL:
                    eval_metrics = eval(
                        loader,
                        self.fusion_heads[fuser],
                        self.classifier_head,
                        self.loss_fn,
                    )
                    metrics = eval_metrics
                else:
                    train_metrics = train(
                        loader,
                        self.optimizer,
                        self.loss_fn,
                        self.fusion_heads[fuser],
                        self.classifier_head,
                        self.modalities,
                    )
                    metrics = train_metrics

                # Collect measurements
                time_taken = time.time() - start_time
                if torch.cuda.is_available():
                    mem_peak = torch.cuda.max_memory_allocated()
                    mem_allocated = torch.cuda.memory_allocated()
                else:
                    mem_peak = 0
                    mem_allocated = 0

                # Log metrics and measurements
                mlflow.log_metrics(
                    {
                        "loss": metrics.get("loss", 0),
                        "accuracy": metrics.get("accuracy", 0),
                        "f1": metrics.get("f1", 0),
                        "time_taken": time_taken,
                        "gpu_memory_peak": mem_peak,
                        "gpu_memory_allocated": mem_allocated,
                    }
                )

            # Recurse with next fusion head
            return self.fuse(mode, batch_size, fuser + 1)

        # Finalize MLflow after last fusion step
        if fuser == 0:
            # Save final model artifacts
            model_state = {
                "fusion_heads": [head.state_dict() for head in self.fusion_heads],
                "classifier_head": self.classifier_head.state_dict(),
            }
            with open("model_checkpoint.pth", "wb") as f:
                torch.save(model_state, f)
            mlflow.log_artifact("model_checkpoint.pth")
            mlflow.end_run()
        return
