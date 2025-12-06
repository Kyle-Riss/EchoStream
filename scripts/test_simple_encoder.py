#!/usr/bin/env python3
"""
Test SimpleCTCEncoder: Check encoder diversity and decoding results.
"""
import sys
sys.path.insert(0, '.')
import torch
import sentencepiece as spm
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import EchoStreamConfig, build_echostream_model

def test_simple_encoder():
    """Test SimpleCTCEncoder diversity and decoding."""
    
    # Load model with SimpleCTCEncoder
    config = EchoStreamConfig()
    config.use_simple_encoder = True  # Use SimpleCTCEncoder
    config.encoder_embed_dim = 128
    config.encoder_layers = 4
    config.encoder_ffn_embed_dim = 512
    config.decoder_embed_dim = 128
    config.mt_decoder_layers = 2
    config.unit_decoder_layers = 2
    config.st_decoder_layers = 2
    config.dropout = 0.1
    
    model = build_echostream_model(config)
    
    # Load checkpoint (best = epoch 41)
    checkpoint = torch.load('checkpoints_simple_encoder/checkpoint_best.pt', map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load('data/tgt_unigram6000/spm_unigram_en.model')
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path='data/train_mini_10.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    print("="*70)
    print("🧪 SimpleCTCEncoder Test - Epoch 10")
    print("="*70)
    
    # ========================================
    # Part 1: Encoder Diversity Check
    # ========================================
    print("\n" + "="*70)
    print("1️⃣ ENCODER DIVERSITY CHECK")
    print("="*70)
    
    # Get 3 samples
    samples = [dataset[i] for i in range(min(3, len(dataset)))]
    batch = collate_s2st_batches(samples)
    
    speech = batch['speech']
    speech_lengths = batch['speech_lengths']
    
    with torch.no_grad():
        encoder_result = model.encoder(speech, speech_lengths)
        encoder_out = encoder_result['encoder_out'][0]  # [T, B, D]
    
    print(f"\nEncoder output shape: {encoder_out.shape}")
    print(f"Mean: {encoder_out.mean().item():.6f}")
    print(f"Std: {encoder_out.std().item():.6f}")
    
    # Transpose to [B, T, D]
    enc = encoder_out.transpose(0, 1)
    
    # Sample diversity
    x1 = enc[0].mean(dim=0)
    x2 = enc[1].mean(dim=0)
    x3 = enc[2].mean(dim=0) if enc.size(0) >= 3 else None
    
    cos_12 = torch.nn.functional.cosine_similarity(x1, x2, dim=0).item()
    print(f"\n📊 Sample diversity:")
    print(f"  Cosine (sample 0 vs 1): {cos_12:.4f}")
    
    if x3 is not None:
        cos_13 = torch.nn.functional.cosine_similarity(x1, x3, dim=0).item()
        cos_23 = torch.nn.functional.cosine_similarity(x2, x3, dim=0).item()
        print(f"  Cosine (sample 0 vs 2): {cos_13:.4f}")
        print(f"  Cosine (sample 1 vs 2): {cos_23:.4f}")
        avg_cos = (cos_12 + cos_13 + cos_23) / 3
        print(f"  Average cosine: {avg_cos:.4f}")
    
    # Temporal variation
    x1_full = enc[0]  # [T, D]
    std_over_time = x1_full.std(dim=0).mean().item()
    print(f"\n📊 Temporal variation (sample 0):")
    print(f"  Std over time: {std_over_time:.4f}")
    
    print("\n🎯 Diversity Assessment:")
    if cos_12 > 0.999:
        print(f"  ❌ Still collapsed (cos={cos_12:.4f})")
    elif cos_12 > 0.98:
        print(f"  ⚠️  Somewhat similar (cos={cos_12:.4f})")
    else:
        print(f"  ✅ Good diversity (cos={cos_12:.4f})")
        print(f"     → Encoder is learning meaningful representations!")
    
    # ========================================
    # Part 2: CTC Decoding Test
    # ========================================
    print("\n" + "="*70)
    print("2️⃣ CTC DECODING TEST (10 samples)")
    print("="*70)
    
    decoded_results = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        batch_single = collate_s2st_batches([sample])
        
        with torch.no_grad():
            output = model(
                src_tokens=batch_single['speech'],
                src_lengths=batch_single['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch_single.get('target_lengths'),
            )
        
        st_log_probs = output['st_log_probs']  # [T, B, V]
        
        # Greedy decoding
        tokens = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
        
        # CTC collapse
        collapsed = []
        prev = None
        blank_id = 1
        for t in tokens.tolist():
            if t == blank_id:
                continue
            if t == prev:
                continue
            collapsed.append(t)
            prev = t
        
        # Decode
        try:
            text = sp.decode_ids(collapsed)
        except:
            text = "(decode failed)"
        
        # Reference
        ref_tokens = sample['tgt_tokens']
        ref_text = sp.decode_ids(ref_tokens.tolist())
        
        decoded_results.append({
            'idx': idx,
            'pred_tokens': collapsed,
            'pred_text': text,
            'ref_text': ref_text,
            'pred_len': len(collapsed),
            'ref_len': len(ref_tokens),
        })
        
        print(f"\n{'='*70}")
        print(f"Sample {idx+1}/10")
        print(f"{'='*70}")
        print(f"📝 Reference: {ref_text[:80]}...")
        print(f"   Tokens: {len(ref_tokens)}")
        print(f"\n🎯 Prediction: {text[:80]}...")
        print(f"   Tokens: {len(collapsed)}")
        
        # Blank ratio
        blank_count = (tokens == blank_id).sum().item()
        blank_ratio = blank_count / len(tokens)
        print(f"\n📊 Stats:")
        print(f"   Frames: {len(tokens)}, Blank: {blank_count} ({blank_ratio*100:.1f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    total_pred = sum(r['pred_len'] for r in decoded_results)
    total_ref = sum(r['ref_len'] for r in decoded_results)
    empty_count = sum(1 for r in decoded_results if r['pred_len'] == 0)
    has_content = len(decoded_results) - empty_count
    
    print(f"\nTotal samples: {len(decoded_results)}")
    print(f"  Has content: {has_content} ({has_content/len(decoded_results)*100:.0f}%)")
    print(f"  Empty: {empty_count} ({empty_count/len(decoded_results)*100:.0f}%)")
    
    print(f"\nAverage metrics:")
    print(f"  Predicted tokens: {total_pred/len(decoded_results):.1f}")
    print(f"  Reference tokens: {total_ref/len(decoded_results):.1f}")
    
    print("\n🎯 Assessment:")
    if empty_count == len(decoded_results):
        print(f"  ❌ All samples are empty - still collapsed")
    elif has_content >= len(decoded_results) * 0.8:
        print(f"  ✅ Most samples have content ({has_content}/{len(decoded_results)})")
        print(f"     → CTC + ST + SPM pipeline is WORKING!")
        if total_pred / len(decoded_results) > 5:
            print(f"     → Average {total_pred/len(decoded_results):.1f} tokens per sample")
            print(f"     → This is a MAJOR breakthrough! 🎉")
    else:
        print(f"  ⚠️  Some samples have content ({has_content}/{len(decoded_results)})")
        print(f"     → Partial success, needs more training")
    
    print("="*70)


if __name__ == "__main__":
    test_simple_encoder()

