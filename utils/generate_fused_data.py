import torch
from tqdm import tqdm

def generate_fused_features(encoder, dataloader, device) -> torch.Tensor:
    encoder.to(device).eval()
    fused_features_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Fused Features", leave=False):
            x_a, l_a, x_b, l_b, _ = [t.to(device) for t in batch]
            fused_feature = encoder(x_a, l_a, x_b, l_b)
            fused_features_list.append(fused_feature.cpu())
    return torch.cat(fused_features_list, dim=0)