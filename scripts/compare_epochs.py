#!/usr/bin/env python3
"""
Compare encoder output between Epoch 10 and Epoch 30.
"""
import sys
sys.path.insert(0, '.')
import torch
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import EchoStreamConfig, build_echostream_model

def compare_epochs():
    """Compare encoder output across epochs."""
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path='data/train_mini_10.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    # Get one sample
    sample = dataset[0]
    batch = collate_s2st_batches([sample])
    
    speech = batch['speech']
    speech_lengths = batch['speech_lengths']
    
    print("="*70)
    print("🔍 Epoch 10 vs Epoch 30 Comparison")
    print("="*70)
    
    # Load model config
    config = EchoStreamConfig()
    config.encoder_embed_dim = 128
    config.encoder_layers = 4
    config.encoder_ffn_embed_dim = 512
    config.decoder_embed_dim = 128
    config.mt_decoder_layers = 2
    config.unit_decoder_layers = 2
    config.st_decoder_layers = 2
    config.dropout = 0.1
    
    # Epoch 10
    print(f"\n1️⃣ Loading Epoch 10...")
    model = build_echostream_model(config)
    checkpoint10 = torch.load('checkpoints_overfit_no_unk/checkpoint_epoch_10.pt', map_location='cpu')
    state_dict10 = checkpoint10['model'] if 'model' in checkpoint10 else checkpoint10
    filtered_state_dict10 = {k: v for k, v in state_dict10.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict10, strict=False)
    model.eval()
    
    with torch.no_grad():
        encoder_result10 = model.encoder(speech, speech_lengths)
        enc10 = encoder_result10['encoder_out'][0]  # [T, B, D]
    
    print(f"Epoch 10 encoder output:")
    print(f"  Shape: {enc10.shape}")
    print(f"  Mean: {enc10.mean().item():.6f}")
    print(f"  Std: {enc10.std().item():.6f}")
    
    # Epoch 30
    print(f"\n2️⃣ Loading Epoch 30...")
    model = build_echostream_model(config)
    checkpoint30 = torch.load('checkpoints_overfit_no_unk/checkpoint_epoch_30.pt', map_location='cpu')
    state_dict30 = checkpoint30['model'] if 'model' in checkpoint30 else checkpoint30
    filtered_state_dict30 = {k: v for k, v in state_dict30.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict30, strict=False)
    model.eval()
    
    with torch.no_grad():
        encoder_result30 = model.encoder(speech, speech_lengths)
        enc30 = encoder_result30['encoder_out'][0]  # [T, B, D]
    
    print(f"Epoch 30 encoder output:")
    print(f"  Shape: {enc30.shape}")
    print(f"  Mean: {enc30.mean().item():.6f}")
    print(f"  Std: {enc30.std().item():.6f}")
    
    # Compare
    print(f"\n3️⃣ Comparison:")
    print("-"*70)
    
    diff_abs = (enc30 - enc10).abs().mean().item()
    diff_rel = diff_abs / (enc10.abs().mean().item() + 1e-8)
    
    print(f"Mean absolute difference: {diff_abs:.6f}")
    print(f"Relative difference: {diff_rel*100:.2f}%")
    
    # Cosine similarity
    enc10_flat = enc10.reshape(-1)
    enc30_flat = enc30.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(enc10_flat, enc30_flat, dim=0)
    print(f"Cosine similarity: {cos.item():.6f}")
    
    # Check parameter difference
    print(f"\n4️⃣ Parameter difference:")
    print("-"*70)
    
    param_diffs = []
    for (name10, p10), (name30, p30) in zip(
        filtered_state_dict10.items(), 
        filtered_state_dict30.items()
    ):
        if 'encoder' in name10:
            diff = (p30 - p10).abs().mean().item()
            param_diffs.append((name10, diff))
    
    # Sort by difference
    param_diffs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 encoder parameters with largest changes:")
    for name, diff in param_diffs[:10]:
        print(f"  {name}: {diff:.6f}")
    
    print(f"\nBottom 10 encoder parameters with smallest changes:")
    for name, diff in param_diffs[-10:]:
        print(f"  {name}: {diff:.6f}")
    
    avg_param_diff = sum(d for _, d in param_diffs) / len(param_diffs)
    print(f"\nAverage parameter change: {avg_param_diff:.6f}")
    
    print("\n" + "="*70)
    print("🎯 Diagnosis:")
    
    if diff_abs < 0.01:
        print(f"  ❌ Encoder output barely changed ({diff_abs:.6f})")
        print(f"     → Encoder is not learning effectively")
    elif avg_param_diff < 0.0001:
        print(f"  ❌ Encoder parameters barely changed ({avg_param_diff:.6f})")
        print(f"     → Gradient is not flowing to encoder")
    elif cos.item() > 0.999:
        print(f"  ⚠️  Encoder output direction unchanged (cos={cos.item():.6f})")
        print(f"     → Encoder is learning but converging to similar representations")
    else:
        print(f"  ✅ Encoder is learning:")
        print(f"     - Output diff: {diff_abs:.6f}")
        print(f"     - Param diff: {avg_param_diff:.6f}")
        print(f"     - Cosine: {cos.item():.6f}")
        print(f"     → Problem is in representation quality, not learning")
    
    print("="*70)


if __name__ == "__main__":
    compare_epochs()


