"""
Test decoding on the 10-sample overfit set
"""
import sys
sys.path.insert(0, '.')

import torch
import sentencepiece as spm
from pathlib import Path

def ctc_collapse(tokens, blank_id):
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
    BLANK_ID = 1  # SentencePiece PAD token
    
    print("="*70)
    print("🧪 10-Sample Overfit Test - Decoding Results")
    print("="*70)
    
    # Load SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load('data/tgt_unigram6000/spm_unigram_en.model')
    print(f"\n✅ Loaded SentencePiece (vocab={sp.get_piece_size()}, blank={BLANK_ID})")
    
    # Load checkpoint
    checkpoint_path = 'checkpoints_simple_encoder/checkpoint_best.pt'
    print(f"✅ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # We need to manually build model with correct config
    # Let's load the checkpoint's config
    from models.echostream_model import EchoStreamModel
    
    # Build model with 128 dim (matching checkpoint)
    # Use build function with config
    from models.echostream_model import EchoStreamConfig, build_echostream_model
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
    
    # Load state dict (ignore vocoder)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print(f"✅ Model loaded")
    
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
    print(f"✅ Dataset loaded: {len(dataset)} samples\n")
    
    # Statistics
    stats = {
        'total': 0,
        'empty': 0,
        'has_content': 0,
        'total_pred_tokens': 0,
        'total_ref_tokens': 0,
        'blank_ratios': [],
    }
    
    # Test all 10 samples
    for i in range(10):
        sample = dataset[i]
        batch = collate_s2st_batches([sample])
        ref_text = dataset.entries[i].tgt_text
        ref_tokens = sample['tgt_tokens']
        
        with torch.no_grad():
            output = model(
                src_tokens=batch['speech'],
                src_lengths=batch['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch.get('target_lengths'),
            )
        
        st_log_probs = output['st_log_probs']  # [T, B, V]
        tokens = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
        
        # CTC collapse
        collapsed = ctc_collapse(tokens.tolist(), BLANK_ID)
        
        # SPM decode
        decoded_text = sp.decode_ids(collapsed) if collapsed else ""
        
        # Stats
        total_frames = len(tokens)
        blank_count = (tokens == BLANK_ID).sum().item()
        blank_ratio = blank_count / total_frames
        
        stats['total'] += 1
        stats['total_pred_tokens'] += len(collapsed)
        stats['total_ref_tokens'] += len(ref_tokens)
        stats['blank_ratios'].append(blank_ratio)
        
        if len(collapsed) > 0:
            stats['has_content'] += 1
        else:
            stats['empty'] += 1
        
        print("="*70)
        print(f"Sample {i+1}/10")
        print("="*70)
        print(f"\n📝 Reference:")
        print(f"   {ref_text}")
        print(f"   Tokens: {len(ref_tokens)}")
        
        print(f"\n🎯 Prediction:")
        print(f"   {decoded_text if decoded_text else '(empty)'}")
        print(f"   Tokens: {len(collapsed)}")
        
        print(f"\n📊 Stats:")
        print(f"   Frames: {total_frames}, Blank: {blank_count} ({blank_ratio:.1%})")
        print(f"   Collapsed tokens: {collapsed[:20]}")
        
        # Similarity check
        if decoded_text and len(decoded_text) > 5:
            print(f"   ✅ Non-trivial output!")
        elif len(collapsed) > 5:
            print(f"   ⚠️  Has tokens but decode failed")
        else:
            print(f"   ❌ Almost empty")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"\nTotal samples: {stats['total']}")
    print(f"  Has content: {stats['has_content']} ({stats['has_content']/stats['total']*100:.0f}%)")
    print(f"  Empty: {stats['empty']} ({stats['empty']/stats['total']*100:.0f}%)")
    print(f"\nAverage metrics:")
    print(f"  Predicted tokens: {stats['total_pred_tokens']/stats['total']:.1f}")
    print(f"  Reference tokens: {stats['total_ref_tokens']/stats['total']:.1f}")
    print(f"  Blank ratio: {sum(stats['blank_ratios'])/len(stats['blank_ratios'])*100:.1f}%")
    
    print(f"\n🎯 Assessment:")
    avg_pred = stats['total_pred_tokens']/stats['total']
    avg_ref = stats['total_ref_tokens']/stats['total']
    
    if avg_pred > avg_ref * 0.5:  # At least 50% of reference length
        print(f"  ✅ CASE A: Model can learn! (pred={avg_pred:.1f}, ref={avg_ref:.1f})")
        print(f"     → Structure is OK, issue is multi-task/optimization")
    elif avg_pred > 5:
        print(f"  ⚠️  CASE A/B: Partial learning (pred={avg_pred:.1f}, ref={avg_ref:.1f})")
        print(f"     → Structure likely OK, needs stronger signal")
    else:
        print(f"  ❌ CASE B: Stuck in blank mode (pred={avg_pred:.1f}, ref={avg_ref:.1f})")
        print(f"     → Structural issue remains - check CTC/blank/downsampling")
    
    print("="*70)

if __name__ == "__main__":
    main()

