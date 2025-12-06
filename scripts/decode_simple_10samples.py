#!/usr/bin/env python3
"""
Decode SimpleCTCEncoder 10-sample results with detailed analysis
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import sentencepiece as spm

def ctc_collapse(tokens, blank_id=1):
    """CTC collapse: remove blanks and consecutive duplicates"""
    collapsed = []
    prev = None
    for t in tokens:
        if t == blank_id:
            continue
        if t == prev:
            continue
        collapsed.append(t)
        prev = t
    return collapsed

def main():
    print("="*70)
    print("🔍 SimpleCTCEncoder 10-Sample Decoding Results")
    print("="*70)
    
    # Load SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load('data/tgt_unigram6000/spm_unigram_en.model')
    print(f"\n✅ SentencePiece loaded (vocab={sp.get_piece_size()})\n")
    
    # Load dataset
    from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
    dataset = S2STManifestDataset(
        manifest_path='data/train_mini_10.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    # Load model
    from models.echostream_model import EchoStreamConfig, build_echostream_model
    config = EchoStreamConfig()
    config.use_simple_encoder = True
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
    checkpoint = torch.load('checkpoints_simple_encoder/checkpoint_best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("✅ Model loaded\n")
    
    print("="*70)
    print("📊 Decoding Results")
    print("="*70)
    
    stats = {
        'total': 0,
        'has_content': 0,
        'total_pred_len': 0,
        'total_ref_len': 0,
        'blank_ratios': [],
    }
    
    for i in range(10):
        sample = dataset[i]
        batch = collate_s2st_batches([sample])
        ref_text = dataset.entries[i].tgt_text
        
        with torch.no_grad():
            output = model(
                src_tokens=batch['speech'],
                src_lengths=batch['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch.get('target_lengths'),
            )
        
        # Get predictions
        st_log_probs = output['st_log_probs']  # [T, B, V]
        tokens = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
        
        # CTC collapse
        collapsed = ctc_collapse(tokens.tolist(), blank_id=1)
        pred_text = sp.decode_ids(collapsed) if collapsed else ""
        
        # Stats
        blank_count = (tokens == 1).sum().item()
        blank_ratio = blank_count / len(tokens) * 100
        
        stats['total'] += 1
        stats['total_pred_len'] += len(collapsed)
        stats['total_ref_len'] += len(sample['tgt_tokens'])
        stats['blank_ratios'].append(blank_ratio)
        
        if len(collapsed) > 0:
            stats['has_content'] += 1
        
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/10")
        print(f"{'='*70}")
        print(f"\n📝 Reference:")
        print(f"   {ref_text}")
        print(f"   Tokens: {len(sample['tgt_tokens'])}")
        
        print(f"\n🎯 Prediction:")
        print(f"   {pred_text if pred_text else '(empty)'}")
        print(f"   Tokens: {len(collapsed)}")
        
        print(f"\n📊 Stats:")
        print(f"   Total frames: {len(tokens)}")
        print(f"   Blank ratio: {blank_ratio:.1f}%")
        print(f"   Predicted token IDs: {collapsed[:30]}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    avg_blank = sum(stats['blank_ratios']) / len(stats['blank_ratios'])
    avg_pred = stats['total_pred_len'] / stats['total']
    avg_ref = stats['total_ref_len'] / stats['total']
    
    print(f"\nSamples with content: {stats['has_content']}/10 ({stats['has_content']*10}%)")
    print(f"Average blank ratio: {avg_blank:.1f}%")
    print(f"Average prediction length: {avg_pred:.1f} tokens")
    print(f"Average reference length: {avg_ref:.1f} tokens")
    print(f"Length ratio: {avg_pred/avg_ref*100:.1f}%")
    
    print(f"\n🎯 Assessment:")
    if avg_pred > avg_ref * 0.5 and stats['has_content'] >= 8:
        print(f"  ✅ EXCELLENT: Model works well!")
        print(f"     - Most samples decoded successfully")
        print(f"     - Length ratio is reasonable")
    elif avg_pred > avg_ref * 0.2 and stats['has_content'] >= 5:
        print(f"  ⚠️  GOOD: Model learning but needs improvement")
    else:
        print(f"  ❌ POOR: Still has issues")
    
    print("="*70)

if __name__ == "__main__":
    main()

