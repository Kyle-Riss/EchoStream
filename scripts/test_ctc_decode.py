"""
Test CTC decoding with actual text output
"""
import sys
sys.path.insert(0, '.')

import torch
import yaml
import sentencepiece as spm
from pathlib import Path
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import build_echostream_model, EchoStreamConfig

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
    print("="*70)
    print("CTC Decoding Test - Real Text Output")
    print("="*70)
    
    BLANK_ID = 1  # SentencePiece PAD token
    
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load('data/tgt_unigram6000/spm_unigram_en.model')
    print(f"\n✅ Loaded SentencePiece model")
    print(f"   Vocab size: {sp.get_piece_size()}")
    print(f"   Blank ID: {BLANK_ID} ('{sp.id_to_piece(BLANK_ID)}')")
    
    # Load config
    config_path = 'configs/echostream_config.mini.yaml'
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Build config
    config_overrides = {}
    encoder_cfg = config_dict.get('encoder', {})
    if encoder_cfg:
        if 'embed_dim' in encoder_cfg:
            config_overrides['encoder_embed_dim'] = encoder_cfg['embed_dim']
        if 'layers' in encoder_cfg:
            config_overrides['encoder_layers'] = encoder_cfg['layers']
        if 'ffn_embed_dim' in encoder_cfg:
            config_overrides['encoder_ffn_embed_dim'] = encoder_cfg['ffn_embed_dim']
    
    mt_decoder_cfg = config_dict.get('mt_decoder', {})
    if mt_decoder_cfg:
        if 'embed_dim' in mt_decoder_cfg:
            config_overrides['decoder_embed_dim'] = mt_decoder_cfg['embed_dim']
        if 'layers' in mt_decoder_cfg:
            config_overrides['mt_decoder_layers'] = mt_decoder_cfg['layers']
    
    unit_decoder_cfg = config_dict.get('unit_decoder', {})
    if unit_decoder_cfg:
        if 'embed_dim' in unit_decoder_cfg and 'decoder_embed_dim' not in config_overrides:
            config_overrides['decoder_embed_dim'] = unit_decoder_cfg['embed_dim']
        if 'layers' in unit_decoder_cfg:
            config_overrides['unit_decoder_layers'] = unit_decoder_cfg['layers']
    
    st_decoder_cfg = config_dict.get('st_decoder', {})
    if st_decoder_cfg and 'layers' in st_decoder_cfg:
        config_overrides['st_decoder_layers'] = st_decoder_cfg['layers']
    
    config = EchoStreamConfig.from_dict(config_overrides)
    
    # Build model
    model = build_echostream_model(config)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints_st_weight_high/checkpoint_best.pt'
    print(f"\n✅ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path='data/train_sampled.units.streamspeech_format_final.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    print(f"\n✅ Dataset loaded: {len(dataset)} samples")
    
    # Test on 20 random samples from dev set
    import random
    random.seed(42)
    
    # Load dev set
    dev_dataset = S2STManifestDataset(
        manifest_path='data/dev_sampled.units.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    print(f"✅ Dev dataset loaded: {len(dev_dataset)} samples")
    
    # Sample 20 random indices
    num_test = min(20, len(dev_dataset))
    test_indices = random.sample(range(len(dev_dataset)), num_test)
    
    # Statistics
    stats = {
        'total': 0,
        'empty': 0,
        'only_space_unk': 0,
        'has_real_tokens': 0,
        'total_predicted_tokens': 0,
        'total_reference_tokens': 0,
        'blank_ratios': [],
    }
    
    for i, sample_idx in enumerate(test_indices):
        print("\n" + "="*70)
        print(f"Sample {i+1}/{num_test} (idx={sample_idx})")
        print("="*70)
        
        sample = dev_dataset[sample_idx]
        batch = collate_s2st_batches([sample])
        
        # Get reference text
        ref_text = dev_dataset.entries[sample_idx].tgt_text
        ref_tokens = sample['tgt_tokens']
        
        print(f"\n📝 Reference:")
        print(f"   Text: {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
        print(f"   Tokens: {ref_tokens[:20].tolist()}...")
        
        # Forward pass
        with torch.no_grad():
            output = model(
                src_tokens=batch['speech'],
                src_lengths=batch['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch.get('target_lengths'),
            )
        
        st_log_probs = output['st_log_probs']  # [T, B, V]
        
        # 1. Argmax
        tokens = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
        
        # 2. CTC collapse
        collapsed = ctc_collapse(tokens.tolist(), BLANK_ID)
        
        # 3. SPM decode
        decoded_text = sp.decode_ids(collapsed)
        
        # Stats
        total_frames = len(tokens)
        blank_count = (tokens == BLANK_ID).sum().item()
        blank_ratio = blank_count / total_frames
        unique_tokens = len(set(collapsed))
        
        print(f"\n🔍 Prediction:")
        print(f"   Total frames: {total_frames}")
        print(f"   Blank frames: {blank_count} ({blank_ratio:.1%})")
        print(f"   Non-blank tokens (after collapse): {len(collapsed)}")
        print(f"   Unique tokens: {unique_tokens}")
        print(f"   Collapsed tokens: {collapsed[:30]}")
        print(f"\n🎯 DECODED TEXT:")
        print(f"   >>> {decoded_text}")
        
        # Analyze output
        stats['total'] += 1
        stats['total_predicted_tokens'] += len(collapsed)
        stats['total_reference_tokens'] += len(ref_tokens)
        stats['blank_ratios'].append(blank_ratio)
        
        # Categorize
        if not decoded_text.strip():
            stats['empty'] += 1
            category = "❌ Empty"
        elif set(collapsed) <= {11, 3}:  # Only space(11) and UNK(3)
            stats['only_space_unk'] += 1
            category = "⚠️  Space/UNK only"
        else:
            stats['has_real_tokens'] += 1
            category = "✅ Has real tokens"
        
        print(f"\n📊 Analysis:")
        print(f"   Category: {category}")
        print(f"   Reference length: {len(ref_text)}")
        print(f"   Decoded length: {len(decoded_text)}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal samples tested: {stats['total']}")
    print(f"\nOutput categories:")
    print(f"  ❌ Empty output: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)")
    print(f"  ⚠️  Space/UNK only: {stats['only_space_unk']} ({stats['only_space_unk']/stats['total']*100:.1f}%)")
    print(f"  ✅ Has real tokens: {stats['has_real_tokens']} ({stats['has_real_tokens']/stats['total']*100:.1f}%)")
    
    print(f"\nToken statistics:")
    print(f"  Avg predicted tokens per sample: {stats['total_predicted_tokens']/stats['total']:.1f}")
    print(f"  Avg reference tokens per sample: {stats['total_reference_tokens']/stats['total']:.1f}")
    print(f"  Avg blank ratio: {sum(stats['blank_ratios'])/len(stats['blank_ratios'])*100:.1f}%")
    
    print(f"\n🎯 Assessment:")
    if stats['has_real_tokens'] > 0:
        print(f"  ✅ Model is starting to predict real tokens!")
        print(f"     Continue training - likely to improve with more epochs.")
    elif stats['only_space_unk'] > stats['empty']:
        print(f"  ⚠️  Model predicts space/UNK but no real tokens yet.")
        print(f"     Structure is working, needs more training.")
    else:
        print(f"  ❌ Most outputs are empty.")
        print(f"     May need to check ST weight or training setup.")

if __name__ == "__main__":
    main()

