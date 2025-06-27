def benchmark_scaling(architecture_type: str, d_model_sweep: List[int], config: Dict) -> List[Dict]:
    """
    Runs a one-epoch benchmark for a range of d_model sizes for a given architecture.
    This version fixes the GFLOPS calculation for both architectures.
    """
    print(f"\n\n{'='*20} SCALING ANALYSIS FOR: {architecture_type.upper()} {'='*20}")
    
    benchmark_results = []
    all_modalities = config['modalities']
    
    for d_model in d_model_sweep:
        print(f"\n--- Benchmarking d_model = {d_model} ---")
        
        nhead = 8
        if d_model % nhead != 0:
            print(f"Skipping d_model={d_model}, not divisible by nhead={nhead}."); continue
        
        # --- MODEL-SPECIFIC SETUP ---
        if architecture_type == 'recursive':
            dataset = BimodalDataset(all_modalities['audio'], all_modalities['vision'], config['labels'])
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
            
            encoder = TransformerEncoder(
                d_in_a=all_modalities['audio'].features.shape[-1], 
                d_in_b=all_modalities['vision'].features.shape[-1], 
                d_model=d_model, nhead=nhead, 
                num_layers=config['num_layers'], 
                d_ffn=d_model // 4
            ).to(config['device'])
            
            classifier = SentimentClassifierHead(d_model, config['num_classes']).to(config['device'])
            
            class CombinedModel(nn.Module):
                def __init__(self, e, c): super().__init__(); self.encoder = e; self.classifier = c
                def forward(self, x_a, l_a, x_b, l_b):
                    return self.classifier(self.encoder(x_a, l_a, x_b, l_b))

        elif architecture_type == 'mult':
            dataset = TrimodalDataset(all_modalities['text'], all_modalities['audio'], all_modalities['vision'], config['labels'])
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

            encoder = MULTModel(
                d_in_t=all_modalities['text'].features.shape[-1],
                d_in_a=all_modalities['audio'].features.shape[-1],
                d_in_v=all_modalities['vision'].features.shape[-1],
                d_model=d_model, nhead=nhead,
                num_layers=config['num_layers'],
                d_ffn=d_model // 4
            ).to(config['device'])

            classifier = SentimentClassifierHead(d_model, config['num_classes']).to(config['device'])

            class CombinedModel(nn.Module):
                def __init__(self, e, c): super().__init__(); self.encoder = e; self.classifier = c
                def forward(self, x_t, l_t, x_a, l_a, x_v, l_v):
                    return self.classifier(self.encoder(x_t, l_t, x_a, l_a, x_v, l_v))
        else: 
            raise ValueError(f"Unknown architecture_type: {architecture_type}")
        
        # --- STATIC METRICS (Params & GFLOPS) ---
        total_params = count_parameters(encoder) + count_parameters(classifier)
        combined_model = CombinedModel(encoder, classifier).to(config['device'])
        
        # --- GFLOPS CALCULATION FIX ---
        # 1. Get a sample batch
        sample_batch = next(iter(loader))
        # 2. THE FIX: Create a TUPLE of the model inputs, excluding the label.
        model_inputs_for_flops = tuple(t.to(config['device']) for t in sample_batch[:-1])
        # 3. Call the calculator with the correctly formatted tuple of inputs
        gflops_for_batch = calculate_gflops(combined_model, model_inputs_for_flops)
        gflops_per_sample = gflops_for_batch / config['batch_size'] if gflops_for_batch > 0 else -1.0

        # --- DYNAMIC METRICS (Time & Memory) ---
        if config['device'].type == 'cuda': torch.cuda.reset_peak_memory_stats(config['device'])
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Benchmarking (d_model={d_model})", leave=False):
                # Pass only the model inputs, excluding the label
                inputs = [t.to(config['device']) for t in batch[:-1]] 
                _ = combined_model(*inputs)
                
        duration = time.time() - start_time
        peak_mem_mb = torch.cuda.max_memory_allocated(config['device']) / (1024*1024) if config['device'].type == 'cuda' else 0
        
        print(f"  Params: {total_params:,} | GFLOPS/sample: {gflops_per_sample:.4f} | Time: {duration:.2f}s | Memory: {peak_mem_mb:.2f}MB")
        benchmark_results.append({
            'architecture': architecture_type, 
            'd_model': d_model, 
            'parameters': total_params, 
            'gflops': gflops_per_sample, # Storing per-sample GFLOPS
            'one_epoch_time_sec': duration, 
            'peak_gpu_memory_mb': peak_mem_mb
        })
        
    return benchmark_results