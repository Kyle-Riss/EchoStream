#!/usr/bin/env python3
"""
Check encoder internals: padding mask, lengths, streaming state.
"""
import sys
sys.path.insert(0, '.')
import torch
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import EchoStreamConfig, build_echostream_model

def check_encoder_internals():
    """Check encoder internal state."""
    
    # Load model
    config = EchoStreamConfig()
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
    checkpoint = torch.load('checkpoints_overfit_no_unk/checkpoint_best.pt', map_location='cpu')
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
    
    speech = batch['speech']
    speech_lengths = batch['speech_lengths']
    
    print("="*70)
    print("🔍 Encoder Internals Analysis")
    print("="*70)
    
    print(f"\n1️⃣ Input information:")
    print("-"*70)
    print(f"Speech shape: {speech.shape}")  # [B, T, F]
    print(f"Speech lengths: {speech_lengths}")
    print(f"Min/Max lengths: {speech_lengths.min().item()}, {speech_lengths.max().item()}")
    
    with torch.no_grad():
        # Call encoder
        encoder_result = model.encoder(speech, speech_lengths)
        encoder_out = encoder_result['encoder_out'][0]  # [T, B, D]
        encoder_padding_mask = (
            encoder_result['encoder_padding_mask'][0] 
            if encoder_result['encoder_padding_mask'] 
            else None
        )
        
        print(f"\n2️⃣ Encoder output information:")
        print("-"*70)
        print(f"Encoder output shape: {encoder_out.shape}")  # [T, B, D]
        print(f"Expected: [T_enc, B, D] where T_enc ≈ T_speech / 4 (downsampling)")
        print(f"Actual T_enc: {encoder_out.size(0)}")
        print(f"Expected T_enc: {speech.size(1) // 4} (approx)")
        
        if encoder_padding_mask is not None:
            print(f"\n3️⃣ Padding mask information:")
            print("-"*70)
            print(f"Padding mask shape: {encoder_padding_mask.shape}")  # [B, T_enc]
            print(f"Padding mask unique values: {encoder_padding_mask.unique()}")
            print(f"Padding mask dtype: {encoder_padding_mask.dtype}")
            
            # Check how many frames are padded for each sample
            for i in range(encoder_padding_mask.size(0)):
                mask = encoder_padding_mask[i]
                num_valid = (~mask).sum().item()
                num_padded = mask.sum().item()
                total = mask.size(0)
                print(f"Sample {i}: valid={num_valid}/{total} ({num_valid/total*100:.1f}%), padded={num_padded}")
                
                # Check if all frames are padded (BUG!)
                if num_valid == 0:
                    print(f"  ❌ WARNING: All frames are padded for sample {i}!")
                elif num_valid < 5:
                    print(f"  ⚠️  WARNING: Very few valid frames ({num_valid}) for sample {i}!")
        else:
            print(f"\n3️⃣ Padding mask: None")
        
        print(f"\n4️⃣ Encoder output statistics per sample:")
        print("-"*70)
        
        # Transpose to [B, T, D] for easier analysis
        encoder_out_batchfirst = encoder_out.transpose(0, 1)  # [B, T, D]
        
        for i in range(encoder_out_batchfirst.size(0)):
            enc = encoder_out_batchfirst[i]  # [T, D]
            
            # If padding mask exists, only look at valid frames
            if encoder_padding_mask is not None:
                mask = encoder_padding_mask[i]
                valid_frames = enc[~mask]  # [T_valid, D]
                print(f"Sample {i} (valid frames only):")
                print(f"  Mean: {valid_frames.mean().item():.6f}")
                print(f"  Std: {valid_frames.std().item():.6f}")
                print(f"  Std over time: {valid_frames.std(dim=0).mean().item():.6f}")
            else:
                print(f"Sample {i}:")
                print(f"  Mean: {enc.mean().item():.6f}")
                print(f"  Std: {enc.std().item():.6f}")
                print(f"  Std over time: {enc.std(dim=0).mean().item():.6f}")
        
        print(f"\n5️⃣ Check for streaming state issues:")
        print("-"*70)
        
        # Check if encoder has cache/state
        if hasattr(model.encoder, 'emformer'):
            emformer = model.encoder.emformer
            if hasattr(emformer, '_memory_bank'):
                print(f"Emformer has memory bank: {emformer._memory_bank is not None}")
                if emformer._memory_bank is not None:
                    print(f"  Memory bank shape: {emformer._memory_bank.shape}")
            if hasattr(emformer, 'reset_cache'):
                print(f"Emformer has reset_cache method: True")
                # Try resetting
                model.encoder.reset_cache()
                print(f"  Cache reset called")
        
        # Forward again after reset and compare
        encoder_result2 = model.encoder(speech, speech_lengths)
        encoder_out2 = encoder_result2['encoder_out'][0]
        
        diff = (encoder_out - encoder_out2).abs().mean().item()
        print(f"\n  Difference after cache reset: {diff:.6f}")
        if diff > 0.001:
            print(f"  ⚠️  WARNING: Output changed after reset! Streaming state issue!")
        else:
            print(f"  ✅ Output consistent (no streaming state issue)")
    
    print("\n" + "="*70)
    print("🎯 Diagnosis:")
    
    if encoder_padding_mask is not None:
        all_padded = []
        for i in range(encoder_padding_mask.size(0)):
            if (~encoder_padding_mask[i]).sum().item() == 0:
                all_padded.append(i)
        
        if all_padded:
            print(f"  ❌ CRITICAL: Samples {all_padded} have ALL frames padded!")
            print(f"     → Encoder sees no valid input")
            print(f"     → This is the root cause!")
        elif encoder_padding_mask.all():
            print(f"  ❌ CRITICAL: All frames in all samples are padded!")
            print(f"     → Encoder sees no valid input at all")
            print(f"     → Check length calculation in encoder")
        else:
            print(f"  ✅ Padding mask looks reasonable")
    
    if diff > 0.001:
        print(f"  ⚠️  Streaming state issue detected")
        print(f"     → Encoder may be reusing cache between samples")
    
    print("="*70)


if __name__ == "__main__":
    check_encoder_internals()


