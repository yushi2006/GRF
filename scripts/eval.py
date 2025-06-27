import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_loop(dataloader, encoder, classifier, loss_fn, device):
    encoder.eval(); classifier.eval(); encoder.to(device); classifier.to(device)
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating", leave=False):
            x_a, l_a, x_b, l_b, labels = [t.to(device) for t in batch]
            mutual_rep = encoder(x_a, l_a, x_b, l_b)
            logits = classifier(mutual_rep)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)
            all_preds.append(torch.argmax(logits, dim=1).cpu()); all_labels.append(labels.cpu())
    all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
    n_samples = len(all_labels)
    accuracy = (all_preds == all_labels).sum().item() / n_samples
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": total_loss / n_samples, "accuracy": accuracy, "f1": macro_f1}

def run_testing(hparams: Dict, best_model_path: str, test_loader: DataLoader, device: torch.device) -> Dict:
    print("\n" + "="*20 + " STARTING TESTING PHASE " + "="*20)
    
    # Instantiate the models
    d_in_t = test_loader.dataset.mod_t.features.shape[-1]
    d_in_a = test_loader.dataset.mod_a.features.shape[-1]
    d_in_v = test_loader.dataset.mod_v.features.shape[-1]
    encoder = MULTModel(d_in_t, d_in_a, d_in_v, **hparams['model']).to(device)
    classifier = SentimentClassifierHead(hparams['model']['d_model'], hparams['num_classes']).to(device)

    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # --- Benchmarking ---
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    test_start_time = time.time()
    
    # 1. Performance Metrics
    loss_fn = nn.CrossEntropyLoss()
    test_metrics = evaluate(test_loader, encoder, classifier, loss_fn, device)

    # 2. Dynamic Cost Metrics
    test_duration = time.time() - test_start_time
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0

    # 3. Static Cost Metrics
    total_params = count_parameters(encoder) + count_parameters(classifier)
    
    # --- GFLOPS CALCULATION FIX ---
    class CombinedModel(nn.Module):
        def __init__(self, enc, cls): super().__init__(); self.encoder = enc; self.classifier = cls
        def forward(self, x_t, l_t, x_a, l_a, x_v, l_v): 
            return self.classifier(self.encoder(x_t, l_t, x_a, l_a, x_v, l_v))
    
    combined_model = CombinedModel(encoder, classifier).to(device)
    
    # Get a sample batch and prepare it correctly for fvcore
    sample_batch = next(iter(test_loader))
    
    # THE FIX: Create a TUPLE of the model inputs, excluding the label.
    # This ensures fvcore unpacks the arguments correctly for the model's forward pass.
    model_inputs_for_flops = tuple(t.to(device) for t in sample_batch[:-1])
    
    # Now call the calculator with the correctly formatted tuple of inputs
    gflops_for_batch = calculate_gflops(combined_model, model_inputs_for_flops)
    
    # Calculate GFLOPS per sample for easier comparison
    gflops_per_sample = gflops_for_batch / hparams['batch_size'] if gflops_for_batch > 0 else -1.0

    return {
        **test_metrics,
        "parameters": total_params,
        "gflops_per_sample": gflops_per_sample,
        "test_execution_time_sec": test_duration,
        "peak_gpu_memory_mb": peak_gpu_mem_mb
    }
