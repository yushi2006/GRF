import argparse
import os
from datetime import datetime

import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from src.data_handler import get_dataloaders
from src.engine import run_epoch
from src.model import HierarchicalFusionModel
from src.utils import load_config, log_dict_as_json, profile_model


def main(args):
    config = load_config(args.config)

    # --- Override config with command-line arguments ---
    config["order"] = args.order
    config["num_classes"] = args.num_classes

    device = torch.device(config.get("device", "cpu"))
    order_name = "-".join(config["order"])

    # --- 1. Setup Data ---
    train_loader, val_loader, test_loader, modality_info = get_dataloaders(config)

    # --- 2. Initialize Model & Optimization ---
    model = HierarchicalFusionModel(config, modality_info).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    warmup_epochs, total_epochs = (
        config["training"]["warmup_epochs"],
        config["training"]["epochs"],
    )
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: (e + 1) / warmup_epochs if e < warmup_epochs else 1,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    loss_fn = nn.L1Loss()

    # --- 3. MLflow Logging ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        config["run_name_template"]
        .replace("{order_name}", order_name)
        .replace("{num_classes}", str(config["num_classes"]))
    )
    full_run_name = f"{run_name}_{timestamp}"

    mlflow.set_experiment(config["project_name"])
    with mlflow.start_run(run_name=full_run_name) as run:
        print(f"\n--- Starting MLflow Run: {full_run_name} (ID: {run.info.run_id}) ---")
        log_dict_as_json(config, "config.json")
        if args.profile:
            sample_batch = next(iter(train_loader))
            profile_metrics = profile_model(model, sample_batch)
            mlflow.log_metrics({f"profile_{k}": v for k, v in profile_metrics.items()})

        # --- 4. Training Loop ---
        best_val_mae = float("inf")
        epochs_no_improve = 0
        patience = config["training"]["patience"]

        for epoch in range(total_epochs):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                is_training=True,
                config=config,
            )
            scheduler.step()
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                loss_fn,
                device,
                is_training=False,
                config=config,
            )

            print(
                f"Epoch {epoch + 1}/{total_epochs} | Val MAE: {val_metrics['mae']:.4f} | Val Corr: {val_metrics['corr']:.4f}"
            )
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()}, step=epoch
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()}, step=epoch
            )

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                epochs_no_improve = 0
                model_save_path = os.path.join(
                    config["model_save_dir"], f"{run_name}_best.pth"
                )
                os.makedirs(config["model_save_dir"], exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print("  -> New best validation MAE. Saving model.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(
                    f"--- Early stopping after {patience} epochs with no improvement. ---"
                )
                break

        # --- 5. Final Evaluation ---
        print("\n--- Loading best model for final evaluation on test set ---")
        model.load_state_dict(torch.load(model_save_path))
        test_metrics = run_epoch(
            model, test_loader, None, loss_fn, device, is_training=False, config=config
        )

        print(f"\n--- Final Test Results ({order_name}) ---")
        for key, val in test_metrics.items():
            print(f"  {key}: {val:.4f}")

        mlflow.log_metrics({f"final_test_{k}": v for k, v in test_metrics.items()})
        log_dict_as_json(test_metrics, "final_test_results.json")
        mlflow.log_artifact(model_save_path, "best_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Fusion Model for Sentiment Analysis."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--order",
        nargs="+",
        required=True,
        help="List of modalities in fusion order (e.g., text audio vision).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        choices=[2, 7],
        required=True,
        help="Number of classes for evaluation (2 or 7).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the model for GFLOPs and params on the first run.",
    )
    args = parser.parse_args()
    main(args)
