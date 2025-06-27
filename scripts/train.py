import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..src import UniModalDataset


def train_loop(dataloader, optimizer, scheduler, loss_fn, encoder, classifier, device, epochs):
    encoder.to(device); classifier.to(device)
    for epoch in range(epochs):
        encoder.train(); classifier.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False):
            x_a, l_a, x_b, l_b, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            mutual_rep = encoder(x_a, l_a, x_b, l_b)
            pred = classifier(mutual_rep)
            loss = loss_fn(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chain(encoder.parameters(), classifier.parameters()), max_norm=0.8)
            optimizer.step()
        scheduler.step()

def run_training(hparams: Dict, loaders: Dict, device: torch.device, save_dir: str) -> str:
    print("\n" + "="*20 + " STARTING TRAINING PHASE " + "="*20)
    train_start_time = time.time()
    
    encoder = MULTModel(d_in_t=loaders['d_in_t'], d_in_a=loaders['d_in_a'], d_in_v=loaders['d_in_v'], **hparams['model']).to(device)
    classifier = SentimentClassifierHead(hparams['model']['d_model'], hparams['num_classes']).to(device)
    
    class_counts = torch.bincount(loaders['train'].dataset.labels); class_weights = len(loaders['train'].dataset.labels) / (len(class_counts) * class_counts.float())
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(chain(encoder.parameters(), classifier.parameters()), lr=hparams['lr'], weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=hparams['epochs'], eta_min=1e-6)

    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, "best_mult_model.pth")
    os.makedirs(save_dir, exist_ok=True)
    
    with mlflow.start_run(run_name="MOSI_MULT_Baseline_Training"):
        loggable_hparams = {**hparams, **hparams['model']}
        del loggable_hparams['model']
        mlflow.log_params(loggable_hparams)
        total_params = count_parameters(encoder) + count_parameters(classifier)
        mlflow.log_metric("total_trainable_params", total_params)
        print(f"Total trainable parameters: {total_params:,}")

        for epoch in range(hparams['epochs']):
            encoder.train(); classifier.train()
            for batch in tqdm(loaders['train'], desc=f"Epoch {epoch + 1}/{hparams['epochs']}", leave=False):
                x_t, l_t, x_a, l_a, x_v, l_v, labels = [t.to(device) for t in batch]
                optimizer.zero_grad()
                pred = classifier(encoder(x_t, l_t, x_a, l_a, x_v, l_v))
                loss = loss_fn(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(chain(encoder.parameters(), classifier.parameters()), max_norm=0.8)
                optimizer.step()
            scheduler.step()

            val_metrics = evaluate(loaders['valid'], encoder, classifier, loss_fn, device)
            print(f"Epoch {epoch + 1} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            mlflow.log_metrics({"val_f1": val_metrics['f1'], "val_accuracy": val_metrics['accuracy']}, step=epoch)

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save({'encoder_state_dict': encoder.state_dict(), 'classifier_state_dict': classifier.state_dict()}, best_model_path)
                print(f"  -> New best model saved with F1: {best_val_f1:.4f}")
    
    total_train_duration = time.time() - train_start_time
    print(f"--- Training finished in {total_train_duration / 60:.2f} minutes. Best model saved to {best_model_path} ---")
    return best_model_path