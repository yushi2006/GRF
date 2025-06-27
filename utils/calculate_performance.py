import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

def count_parameters(model: nn.Module) -> int: return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_gflops(model: nn.Module, model_inputs: tuple) -> float:
    """
    Calculates GFLOPS for a model given a tuple of sample inputs.
    This version expects a pre-processed tuple of tensors, ready for the model's forward pass.
    """
    if not FVCORE_AVAILABLE: 
        return -1.0
    
    try:
        model.eval()
        # The 'model_inputs' is already a tuple of tensors ready to be passed.
        # fvcore will internally call model.forward(*model_inputs)
        flop_counter = FlopCountAnalysis(model, model_inputs)
        # Return total GFLOPS for the forward pass on the given batch.
        return flop_counter.total() / 1e9
    except Exception as e:
        print(f"Could not calculate GFLOPS: {e}")
        # This will now print the actual error instead of just failing silently.
        return -1.0
