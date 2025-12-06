#!/usr/bin/env python3
"""
Decode 2,400-sample V3 model results
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import sentencepiece as spm
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import build_echostream_model, EchoStreamConfig

def decode_ctc_greedy(log_probs, blank_id=1):
    """CTC greedy decoding with blank removal and repeat collapse"""
    # log_probs: [T, V]
    pred_ids = log_probs.argmax(dim=-1)  # [T]
    
    # Remove consecutive duplicates
    collapsed = []
    prev = None
    for token_id in pred_ids.tolist():
        if token_id != prev:
            collapsed.append(token_id)
            prev = token_id
    
    # Remove blanks
    filtered = [t for t in collapsed if t != blank_id]
    
    return filtered

def main():
    print("="*70)
    print("🔍 2,400-Sample V3 Model - Decoding Results")
    print("="*70)
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tgt_unigram6000/spm_unigram_en.model')
    print(f"\n✅ Tokenizer loaded (vocab={tokenizer.get_piece_size()})")
    
    # Load dev dataset
    dev_dataset = S2STManifestDataset(
        manifest_path='data/dev_sampled.units.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    print(f"✅ Dev dataset loaded: {len(dev_dataset)} samples")
    
    # Load model config (matching training config)
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
    checkpoint_path = 'checkpoints_simple_2400_v3/checkpoint_epoch_100.pt'
    print(f"\n✅ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Filter vocoder keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print("✅ Model loaded")
    
    print("\n" + "="*70)
    print("📊 Decoding Dev Set Samples...")
    print("="*70)
    
    # Statistics
    stats = {
        'total': 0,
        'empty': 0,
        'has_content': 0,
        'total_pred_len': 0,
        'total_ref_len': 0,
        'blank_ratios': [],
    }
    
    # Decode first 20 samples
    num_samples = min(20, len(dev_dataset))
    
    for idx in range(num_samples):
        sample = dev_dataset[idx]
        batch = collate_s2st_batches([sample])
        
        with torch.no_grad():
            # Forward through model
            output = model(
                src_tokens=batch['speech'],
                src_lengths=batch['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch.get('target_lengths'),
            )
            
            # Get ST CTC log_probs: [T, B, V]
            st_log_probs = output['st_log_probs']  # [T, B, V]
            
            # Transpose to [B, T, V] for decoding
            log_probs = st_log_probs.transpose(0, 1)  # [B, T, V]
            
            # Greedy decode
            pred_tokens = decode_ctc_greedy(log_probs[0], blank_id=1)
            
            # Decode text
            if len(pred_tokens) > 0:
                pred_text = tokenizer.decode(pred_tokens)
            else:
                pred_text = ""
            
            # GT
            gt_tokens = batch['target_text'][0][:batch['target_lengths'][0]-1].tolist()
            gt_text = tokenizer.decode(gt_tokens)
            
            # Blank ratio (from original [T, B, V] format)
            argmax_ids = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
            blank_count = (argmax_ids == 1).sum().item()
            total_frames = argmax_ids.size(0)
            blank_ratio = blank_count / total_frames * 100 if total_frames > 0 else 0
            
            # Statistics
            stats['total'] += 1
            stats['total_pred_len'] += len(pred_tokens)
            stats['total_ref_len'] += len(gt_tokens)
            stats['blank_ratios'].append(blank_ratio)
            
            if len(pred_tokens) == 0:
                stats['empty'] += 1
            else:
                stats['has_content'] += 1
            
            print(f"\n[Sample {idx+1}/{num_samples}]")
            print(f"  GT:   {gt_text}")
            print(f"  Pred: {pred_text}")
            print(f"  Tokens: {len(pred_tokens)} (GT: {len(gt_tokens)})")
            print(f"  Blank ratio: {blank_ratio:.1f}%")
    
    print("\n" + "="*70)
    print("📊 Summary:")
    print("="*70)
    print(f"  Total samples: {stats['total']}")
    print(f"  Empty outputs: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)")
    print(f"  Has content: {stats['has_content']} ({stats['has_content']/stats['total']*100:.1f}%)")
    print(f"  Avg pred length: {stats['total_pred_len']/stats['total']:.1f} tokens")
    print(f"  Avg ref length: {stats['total_ref_len']/stats['total']:.1f} tokens")
    print(f"  Avg blank ratio: {sum(stats['blank_ratios'])/len(stats['blank_ratios']):.1f}%")
    print("="*70)

if __name__ == '__main__':
    main()

