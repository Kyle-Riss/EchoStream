#!/usr/bin/env python3
"""
Check encoder output statistics and diversity.
"""
import sys
sys.path.insert(0, '.')
import torch
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import EchoStreamConfig, build_echostream_model

def check_encoder_output():
    """Check encoder output statistics."""
    
    # Load model with Conformer
    config = EchoStreamConfig()
    config.use_conformer = True  # Use Conformer
    config.encoder_embed_dim = 128
    config.encoder_layers = 4
    config.encoder_ffn_embed_dim = 512
    config.decoder_embed_dim = 128
    config.mt_decoder_layers = 2
    config.unit_decoder_layers = 2
    config.st_decoder_layers = 2
    config.dropout = 0.1
    
    model = build_echostream_model(config)
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints_conformer_overfit/checkpoint_best.pt', map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path='data/train_mini_10.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    # Get 3 samples
    samples = [dataset[i] for i in range(min(3, len(dataset)))]
    batch = collate_s2st_batches(samples)
    
    print("="*70)
    print("🔍 Encoder Output Analysis")
    print("="*70)
    
    with torch.no_grad():
        # Forward through encoder only
        speech = batch['speech']
        speech_lengths = batch['speech_lengths']
        
        # Call encoder directly
        encoder_result = model.encoder(speech, speech_lengths)
        encoder_out = encoder_result['encoder_out'][0]  # List[Tensor] -> [T, B, D]
        encoder_padding_mask = (
            encoder_result['encoder_padding_mask'][0] 
            if encoder_result['encoder_padding_mask'] 
            else None
        )
        
        print(f"\n1️⃣ Encoder output shape & statistics:")
        print("-"*70)
        print(f"Shape: {encoder_out.shape}")  # [T, B, D]
        print(f"Mean: {encoder_out.mean().item():.6f}")
        print(f"Std: {encoder_out.std().item():.6f}")
        print(f"Min: {encoder_out.min().item():.6f}")
        print(f"Max: {encoder_out.max().item():.6f}")
        
        # Per-frame std (variation across time)
        per_frame_std = encoder_out.std(dim=0).mean().item()  # [B, D] -> scalar
        print(f"Per-frame std (mean): {per_frame_std:.6f}")
        
        # Per-dim std (variation across dimensions)
        per_dim_std = encoder_out.std(dim=-1).mean().item()  # [T, B] -> scalar
        print(f"Per-dim std (mean): {per_dim_std:.6f}")
        
        print("\n2️⃣ Diversity check (cosine similarity between samples):")
        print("-"*70)
        
        if encoder_out.size(1) >= 2:
            # Compare first two samples
            x1 = encoder_out[:, 0, :]  # [T1, D]
            x2 = encoder_out[:, 1, :]  # [T2, D]
            
            # Pool over time
            x1_mean = x1.mean(dim=0)  # [D]
            x2_mean = x2.mean(dim=0)  # [D]
            
            cos = torch.nn.functional.cosine_similarity(x1_mean, x2_mean, dim=0)
            print(f"Cosine similarity (sample 0 vs 1): {cos.item():.4f}")
            
            # Also check per-frame similarity
            min_len = min(x1.size(0), x2.size(0))
            cos_per_frame = torch.nn.functional.cosine_similarity(
                x1[:min_len], x2[:min_len], dim=-1
            ).mean()
            print(f"Cosine similarity (per-frame avg): {cos_per_frame.item():.4f}")
        
        if encoder_out.size(1) >= 3:
            x3 = encoder_out[:, 2, :]
            x3_mean = x3.mean(dim=0)
            cos_13 = torch.nn.functional.cosine_similarity(x1_mean, x3_mean, dim=0)
            cos_23 = torch.nn.functional.cosine_similarity(x2_mean, x3_mean, dim=0)
            print(f"Cosine similarity (sample 0 vs 2): {cos_13.item():.4f}")
            print(f"Cosine similarity (sample 1 vs 2): {cos_23.item():.4f}")
        
        print("\n3️⃣ Temporal variation (same sample, different frames):")
        print("-"*70)
        
        # Check if different frames have different values
        x1 = encoder_out[:, 0, :]  # [T, D]
        if x1.size(0) >= 10:
            # Compare frame 0 vs frame T//2
            frame_0 = x1[0]
            frame_mid = x1[x1.size(0) // 2]
            frame_end = x1[-1]
            
            cos_0_mid = torch.nn.functional.cosine_similarity(frame_0, frame_mid, dim=0)
            cos_0_end = torch.nn.functional.cosine_similarity(frame_0, frame_end, dim=0)
            cos_mid_end = torch.nn.functional.cosine_similarity(frame_mid, frame_end, dim=0)
            
            print(f"Cosine (frame 0 vs mid): {cos_0_mid.item():.4f}")
            print(f"Cosine (frame 0 vs end): {cos_0_end.item():.4f}")
            print(f"Cosine (frame mid vs end): {cos_mid_end.item():.4f}")
            
            # L2 distance
            l2_0_mid = (frame_0 - frame_mid).norm().item()
            l2_0_end = (frame_0 - frame_end).norm().item()
            print(f"L2 distance (frame 0 vs mid): {l2_0_mid:.4f}")
            print(f"L2 distance (frame 0 vs end): {l2_0_end:.4f}")
    
    print("\n" + "="*70)
    print("🎯 Diagnosis:")
    
    if encoder_out.std().item() < 0.01:
        print(f"  ❌ Encoder output std is too low ({encoder_out.std().item():.6f})")
        print(f"     → Encoder is outputting near-constant values")
        print(f"     → This is likely the root cause!")
    elif per_dim_std < 0.01:
        print(f"  ❌ Per-dim std is too low ({per_dim_std:.6f})")
        print(f"     → Encoder output has no temporal variation")
    elif encoder_out.size(1) >= 2 and cos.item() > 0.99:
        print(f"  ⚠️  Samples are too similar (cos={cos.item():.4f})")
        print(f"     → Encoder may not be learning meaningful representations")
    elif encoder_out.size(1) >= 2 and cos_per_frame.item() > 0.99:
        print(f"  ⚠️  Per-frame similarity is too high ({cos_per_frame.item():.4f})")
        print(f"     → Encoder may be outputting similar frames")
    elif x1.size(0) >= 10 and cos_0_mid.item() > 0.99:
        print(f"  ⚠️  Temporal variation is too low (cos={cos_0_mid.item():.4f})")
        print(f"     → Encoder may not be capturing temporal dynamics")
    else:
        print(f"  ✅ Encoder output looks healthy:")
        print(f"     - Std: {encoder_out.std().item():.4f}")
        print(f"     - Sample diversity: cos={cos.item():.4f}")
        if x1.size(0) >= 10:
            print(f"     - Temporal variation: cos={cos_0_mid.item():.4f}")
        print(f"     → Problem is likely in ST decoder or loss calculation")
    
    print("="*70)


if __name__ == "__main__":
    check_encoder_output()

