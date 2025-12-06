#!/usr/bin/env python3
"""
Check if input features are already similar before encoder.
"""
import sys
sys.path.insert(0, '.')
import torch
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import EchoStreamConfig, build_echostream_model

def check_input_features():
    """Check input feature statistics."""
    
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
    
    features = batch['speech']  # [B, T, F]
    
    print("="*70)
    print("🔍 Input Feature Analysis (Before Encoder)")
    print("="*70)
    
    print(f"\n1️⃣ Feature shape & overall statistics:")
    print("-"*70)
    print(f"Shape: {features.shape}")  # [B, T, F]
    print(f"Overall mean: {features.mean().item():.6f}")
    print(f"Overall std: {features.std().item():.6f}")
    print(f"Overall min: {features.min().item():.6f}")
    print(f"Overall max: {features.max().item():.6f}")
    
    print(f"\n2️⃣ Per-utterance statistics:")
    print("-"*70)
    
    for i in range(features.size(0)):
        utt = features[i]  # [T, F]
        print(f"Sample {i}:")
        print(f"  Mean: {utt.mean().item():.6f}")
        print(f"  Std: {utt.std().item():.6f}")
        print(f"  Min: {utt.min().item():.6f}")
        print(f"  Max: {utt.max().item():.6f}")
    
    print(f"\n3️⃣ Diversity check (cosine similarity between samples):")
    print("-"*70)
    
    if features.size(0) >= 2:
        # Compare first two samples
        # Pool over time to get [F] vectors
        x1 = features[0].mean(dim=0)  # [F]
        x2 = features[1].mean(dim=0)  # [F]
        
        cos = torch.nn.functional.cosine_similarity(x1, x2, dim=0)
        print(f"Cosine similarity (sample 0 vs 1): {cos.item():.4f}")
        
        # L2 distance
        l2 = (x1 - x2).norm().item()
        print(f"L2 distance (sample 0 vs 1): {l2:.4f}")
    
    if features.size(0) >= 3:
        x3 = features[2].mean(dim=0)
        cos_13 = torch.nn.functional.cosine_similarity(x1, x3, dim=0)
        cos_23 = torch.nn.functional.cosine_similarity(x2, x3, dim=0)
        print(f"Cosine similarity (sample 0 vs 2): {cos_13.item():.4f}")
        print(f"Cosine similarity (sample 1 vs 2): {cos_23.item():.4f}")
    
    print(f"\n4️⃣ Temporal variation (within same sample):")
    print("-"*70)
    
    x1_full = features[0]  # [T, F]
    if x1_full.size(0) >= 10:
        frame_0 = x1_full[0]
        frame_mid = x1_full[x1_full.size(0) // 2]
        frame_end = x1_full[-1]
        
        cos_0_mid = torch.nn.functional.cosine_similarity(frame_0, frame_mid, dim=0)
        cos_0_end = torch.nn.functional.cosine_similarity(frame_0, frame_end, dim=0)
        
        print(f"Cosine (frame 0 vs mid): {cos_0_mid.item():.4f}")
        print(f"Cosine (frame 0 vs end): {cos_0_end.item():.4f}")
        
        l2_0_mid = (frame_0 - frame_mid).norm().item()
        l2_0_end = (frame_0 - frame_end).norm().item()
        print(f"L2 distance (frame 0 vs mid): {l2_0_mid:.4f}")
        print(f"L2 distance (frame 0 vs end): {l2_0_end:.4f}")
    
    print("\n" + "="*70)
    print("🎯 Diagnosis:")
    
    # Check if features are too similar
    if features.size(0) >= 2:
        if cos.item() > 0.99:
            print(f"  ❌ Input features are too similar (cos={cos.item():.4f})!")
            print(f"     → Problem is in audio/feature pipeline (before encoder)")
            print(f"     → Check: wav loading, CMVN, normalization")
        elif features.std().item() < 0.1:
            print(f"  ⚠️  Feature std is low ({features.std().item():.4f})")
            print(f"     → Aggressive normalization may be collapsing features")
        else:
            print(f"  ✅ Input features look diverse:")
            print(f"     - Cosine similarity: {cos.item():.4f}")
            print(f"     - Overall std: {features.std().item():.4f}")
            print(f"     → Problem is in encoder, not input features")
    
    print("="*70)


if __name__ == "__main__":
    check_input_features()


