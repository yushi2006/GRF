import torch
import torch.nn as nn
from .modules import CrossModalAttentionEncoder, GatedFusionUnit, SentimentRegressionHead


class FusionPipeline(nn.Module):
    def __init__(self, initial_modalities, model_save_dir, hparams, device):
        super().__init__(); self.hparams = hparams; self.device = device; self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)
        modality_dims = [mod.features.shape[-1] for mod in initial_modalities]
        self.projectors = nn.ModuleList([nn.Linear(dim, hparams['d_model']) for dim in modality_dims])
        self.fusion_encoder_1 = CrossModalAttentionEncoder(d_model=hparams['d_model'], nhead=hparams['num_heads'], d_ffn=hparams['d_ffn'], num_layers=hparams['num_layers'], dropout=hparams['dropout'])
        self.fusion_encoder_2 = CrossModalAttentionEncoder(d_model=hparams['d_model'], nhead=hparams['num_heads'], d_ffn=hparams['d_ffn'], num_layers=hparams['num_layers'], dropout=hparams['dropout'])

        # --- MODIFIED: Using GatedFusionUnit for merging representations ---
        self.gfu_1 = GatedFusionUnit(hparams['d_model'], hparams['dropout'])
        self.gfu_2 = GatedFusionUnit(hparams['d_model'], hparams['dropout'])

        self.regressor = SentimentRegressionHead(hparams['d_model'], hparams['dropout'])
        self.auxiliary_regressor = SentimentRegressionHead(hparams['d_model'], hparams['dropout'])

    def setup_optimization(self):
        self.optimizer = AdamW(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay'])
        warmup_epochs, total_epochs = self.hparams['warmup_epochs'], self.hparams['epochs']
        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1)
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        self.loss_fn = nn.L1Loss()
        print(f"--- Using L1Loss (MAE) for regression. ---")

    def forward(self, *batch):
        modalities_in_batch = [(batch[i].to(self.device), batch[i+1].to(self.device)) for i in range(0, len(batch)-1, 2)]
        (mod_a_data, l_a), (mod_b_data, l_b), (mod_c_data, l_c) = modalities_in_batch
        mod_a_proj = self.projectors[0](mod_a_data); mod_b_proj = self.projectors[1](mod_b_data); mod_c_proj = self.projectors[2](mod_c_data)

        # First fusion step (e.g., Audio + Vision)
        state_repr_1, new_info_1 = self.fusion_encoder_1(mod_a_proj, l_a, mod_b_proj, l_b)
        fused_state_intermediate = self.gfu_1(state_repr_1, new_info_1) # Use GFU to merge

        # Prepare for second fusion step
        intermediate_state_seq = fused_state_intermediate.unsqueeze(1)
        l_intermediate = torch.ones(fused_state_intermediate.size(0), device=self.device)

        # Second fusion step (e.g., (Audio+Vision) + Text)
        state_repr_2, new_info_2 = self.fusion_encoder_2(intermediate_state_seq, l_intermediate, mod_c_proj, l_c)
        final_representation = self.gfu_2(state_repr_2, new_info_2) # Use GFU to merge

        return self.regressor(final_representation), self.auxiliary_regressor(fused_state_intermediate)

    def run_epoch(self, dataloader, is_training: bool):
        self.train(is_training)
        total_loss, all_preds, all_labels = 0.0, [], []
        desc = "Training" if is_training else "Evaluating"
        for batch in tqdm(dataloader, desc=desc, leave=False):
            labels = batch[-1].to(self.device).unsqueeze(1); inputs = batch[:-1]
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(is_training):
                main_preds, aux_preds = self(*inputs)
                loss = self.loss_fn(main_preds, labels) + self.hparams['aux_loss_weight'] * self.loss_fn(aux_preds, labels)
            if is_training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0); self.optimizer.step()
            total_loss += loss.item() * len(labels)
            all_preds.append(main_preds.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        if is_training: self.scheduler.step()
        all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
        metrics = calculate_regression_metrics(all_preds, all_labels, self.hparams['num_classes'])
        metrics["loss"] = total_loss / len(all_labels)
        return metrics

    def save_models(self, prefix=""):
        filename = os.path.join(self.model_save_dir, f"{prefix}models.pth"); torch.save(self.state_dict(), filename); return filename
    def load_models(self, prefix=""):
        filename = os.path.join(self.model_save_dir, f"{prefix}models.pth")
        if not os.path.exists(filename): raise FileNotFoundError(f"Model file not found at {filename}")
        self.load_state_dict(torch.load(filename, map_location=self.device)); print(f"Models loaded from {filename}")

    def run_training_loop(self, train_loader, val_loader, test_loader):
        self.to(self.device)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); order_name = self.hparams.get("order_name", "default")
        num_classes = self.hparams['num_classes']
        parent_run_name = f"CrossModal-Regression-{order_name}-{num_classes}class-eval_{timestamp}"; mlflow.set_experiment("Efficient_Fusion_CrossModal_Regression")
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            print(f"\n--- Starting MLflow Run: {parent_run_name} ---")
            log_dict_as_json(self.hparams, "hparams.json"); mlflow.log_params({k: v for k, v in self.hparams.items() if isinstance(v, (str, int, float, bool))})
            if self.hparams.get("profile_metrics"): mlflow.log_metrics({f"profile_{k}": v for k, v in self.hparams["profile_metrics"].items()})
            best_val_mae = float('inf')
            epochs_no_improve = 0
            for epoch in range(self.hparams['epochs']):
                train_metrics = self.run_epoch(train_loader, is_training=True)
                val_metrics = self.run_epoch(val_loader, is_training=False)
                print(f"Epoch {epoch+1}/{self.hparams['epochs']} | Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
                if val_metrics['mae'] < best_val_mae:
                    best_val_mae = val_metrics['mae']; epochs_no_improve = 0; model_path = self.save_models(f"{order_name}_best_")
                    print(f"  -> New best validation MAE. Saving models to {model_path}.")
                else:
                    epochs_no_improve += 1; print(f"  -> No improvement in {epochs_no_improve} epochs.")
                if epochs_no_improve >= self.hparams['patience']: print(f"--- Early stopping ---"); break
            mlflow.log_artifact(model_path, "best_model"); self.load_models(f"{order_name}_best_")
            print("\n--- Final Evaluation on Test Set ---")
            test_metrics = self.run_epoch(test_loader, is_training=False)
            mlflow.log_metrics({f"final_{k}": v for k, v in test_metrics.items()}); log_dict_as_json(test_metrics, "final_results.json")
            return test_metrics

