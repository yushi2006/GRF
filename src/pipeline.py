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
    TRAIN, TEST = 1, 2

class FusionPipeline:
    def __init__(self, initial_modalities, initial_labels, model_save_dir: str, num_heads_list, d_model, num_layers, d_ffn, optimizer_class, optimizer_params, num_classes, device, epochs_per_step):
        self.initial_modalities = initial_modalities
        self.initial_labels = initial_labels
        self.model_save_dir = model_save_dir
        self.num_fusion_steps = len(self.initial_modalities) - 1
        if self.num_fusion_steps != len(num_heads_list): raise ValueError("Head sizes list must match number of fusion steps.")
        self.d_model, self.num_heads_list, self.num_layers, self.d_ffn = d_model, num_heads_list, num_layers, d_ffn
        self.optimizer_class, self.optimizer_params = optimizer_class, optimizer_params
        self.device, self.epochs_per_step = device, epochs_per_step
        self.num_classes = num_classes
        os.makedirs(self.model_save_dir, exist_ok=True)

    def run(self, mode: Mode, batch_size: int, modalities: List[ModalityData] = None, labels: torch.Tensor = None) -> Dict[str, Any]:
        working_modalities = list(modalities) if modalities else list(self.initial_modalities)
        current_labels = labels if labels is not None else self.initial_labels
    
        class_counts = torch.bincount(current_labels)
        class_weights = len(current_labels) / (len(class_counts) * class_counts.float())
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        print(f"\nUsing class weights for {mode.name} run: {class_weights.numpy()}")
    
        total_execution_time = 0; max_peak_gpu_mem = 0; final_step_metrics = {}
    
        parent_run_name = f"CMU_MOSI_Fusion_{mode.name}_{self.num_classes}class"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_params({"mode": mode.name, "num_classes": self.num_classes, "epochs_per_step": self.epochs_per_step, "d_model": self.d_model})
    
            for step in range(self.num_fusion_steps):
                start_time = time.time()
                if self.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(self.device)
    
                print(f"\n--- Starting Fusion Step {step + 1}/{self.num_fusion_steps} [{mode.name}] ---")
                mod_a, mod_b = working_modalities.pop(0), working_modalities.pop(0)
                print(f"Fusing '{mod_a.name}' and '{mod_b.name}'")
    
                dataset = BimodalDataset(mod_a, mod_b, current_labels)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == Mode.TRAIN), num_workers=0, pin_memory=True)
    
                d_in_a, d_in_b = mod_a.features.shape[-1], mod_b.features.shape[-1]
                nhead = self.num_heads_list[step]
                encoder = TransformerEncoder(d_in_a, d_in_b, self.d_model, nhead, self.num_layers, self.d_ffn).to(self.device)
                classifier = SentimentClassifierHead(self.d_model, self.num_classes, dropout=0.3).to(self.device)
                total_params = count_parameters(encoder) + count_parameters(classifier)
                print(f"Step {step+1} Model Parameters: Total={total_params:,}")
    
                model_name_base = f"step{step+1}_{mod_a.name}_{mod_b.name}_{self.num_classes}class"
                encoder_filename = os.path.join(self.model_save_dir, f"encoder_{model_name_base}.pth")
                classifier_filename = os.path.join(self.model_save_dir, f"classifier_{model_name_base}.pth")
    
                with mlflow.start_run(run_name=f"fusion_step_{step+1}", nested=True):
                    mlflow.log_params({"step": step + 1, "fusing": f"{mod_a.name}+{mod_b.name}", "total_params": total_params})
    
                    if mode == Mode.TRAIN:
                        params = list(encoder.parameters()) + list(classifier.parameters())
                        optimizer = self.optimizer_class(params, **self.optimizer_params)
                        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs_per_step, eta_min=1e-6)
                        train_loop(loader, optimizer, scheduler, loss_fn, encoder, classifier, self.device, self.epochs_per_step)
                        print(f"Saving models for step {step+1} to '{self.model_save_dir}'...")
                        torch.save(encoder.state_dict(), encoder_filename)
                        torch.save(classifier.state_dict(), classifier_filename)
                        mlflow.log_artifact(encoder_filename, artifact_path="models")
                        mlflow.log_artifact(classifier_filename, artifact_path="models")
                    elif mode == Mode.TEST:
                        print(f"Loading pre-trained models for step {step+1} from '{self.model_save_dir}'...")
                        if not os.path.exists(encoder_filename) or not os.path.exists(classifier_filename):
                            raise FileNotFoundError(f"Models for {self.num_classes}-class task not found: {encoder_filename} or {classifier_filename}. Please run the TRAIN pipeline first for this task.")
                        encoder.load_state_dict(torch.load(encoder_filename, map_location=self.device))
                        classifier.load_state_dict(torch.load(classifier_filename, map_location=self.device))
    
                    metrics = eval_loop(loader, encoder, classifier, loss_fn, self.device)
                    mlflow.log_metrics({f"eval_loss": metrics['loss'], f"eval_accuracy": metrics['accuracy'], f"eval_f1": metrics['f1']})
    
                    full_dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                    fused_features = generate_fused_features(encoder, full_dataset_loader, self.device)
                    new_fused_modality = ModalityData(fused_features.unsqueeze(1), torch.ones(len(fused_features), dtype=torch.long), f"fused({mod_a.name},{mod_b.name})")
                    working_modalities.insert(0, new_fused_modality)
    
                duration = time.time() - start_time
                total_execution_time += duration
                print(f"Step {step+1} took {duration:.2f} seconds.")
                print(f"Step {step+1} Eval Metrics: Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
                if self.device.type == 'cuda':
                    peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    max_peak_gpu_mem = max(max_peak_gpu_mem, peak_mem_mb)
                    print(f"Peak GPU Memory for Step {step+1}: {peak_mem_mb:.2f} MB")
                    mlflow.log_metric("peak_gpu_mem_mb", peak_mem_mb)
    
                if step == self.num_fusion_steps - 1:
                    final_step_metrics = metrics
                    
                    sample_batch = next(iter(loader))
                    sample_input = [t.to(self.device) for t in sample_batch[:-1]]
    
                    class CombinedModel(nn.Module):
                        def __init__(self, enc, cls): super().__init__(); self.encoder = enc; self.classifier = cls
                        def forward(self, x_a, l_a, x_b, l_b):
                            return self.classifier(self.encoder(x_a, l_a, x_b, l_b))
                    
                    combined_model = CombinedModel(encoder, classifier)
                    
                    gflops_for_batch = calculate_gflops(combined_model, tuple(sample_input))
                    
                    gflops_per_sample = gflops_for_batch / batch_size if gflops_for_batch > 0 else -1.0
                    
                    print(f"Final Step GFLOPS ({mode.name}): {gflops_per_sample:.4f} per sample")
                    mlflow.log_metric("gflops_per_sample", gflops_per_sample)
                    final_step_metrics['gflops'] = gflops_per_sample
    
        print(f"\n--- Pipeline {mode.name} finished ---")
        return {
            "loss": final_step_metrics.get('loss'), "accuracy": final_step_metrics.get('accuracy'), "f1": final_step_metrics.get('f1'),
            "gflops": final_step_metrics.get('gflops', -1.0), "total_execution_time_sec": total_execution_time, "peak_gpu_memory_mb": max_peak_gpu_mem,
        }
