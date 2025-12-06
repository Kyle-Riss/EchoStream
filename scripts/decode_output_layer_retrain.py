#!/usr/bin/env python3
"""
Decode Output Layer Re-trained model results
Compare Epoch 4 vs Epoch 6 (or latest) checkpoints
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

def decode_checkpoint(checkpoint_path, num_samples=10, dataset_type='dev'):
    """Decode samples using a checkpoint"""
    print(f"\n{'='*70}")
    print(f"🔍 Decoding: {checkpoint_path}")
    print(f"{'='*70}")
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('data/tgt_unigram6000/spm_unigram_en.model')
    
    # Load dataset
    if dataset_type == 'dev':
        dataset = S2STManifestDataset(
            manifest_path='data/dev_sampled.units.tsv',
            data_root='data',
            sample_rate=16000,
            num_mel_bins=80,
            tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
            text_level='word',
        )
    else:  # train
        dataset = S2STManifestDataset(
            manifest_path='data/train_sampled.units.streamspeech_format_final.tsv',
            data_root='data',
            sample_rate=16000,
            num_mel_bins=80,
            tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
            text_level='word',
        )
    
    # Load model
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # Fix unit_decoder positional encoding
    original_forward = model.forward
    def forward_without_unit(src_tokens, src_lengths, prev_output_tokens=None, target_lengths=None, **kwargs):
        try:
            return original_forward(src_tokens, src_lengths, prev_output_tokens, target_lengths, **kwargs)
        except RuntimeError as e:
            if 'size of tensor a' in str(e) and 'size of tensor b' in str(e):
                encoder_out = model.encoder(src_tokens, src_lengths)
                encoder_hidden = encoder_out['encoder_out'][0]
                encoder_padding_mask = encoder_out['encoder_padding_mask'][0] if encoder_out['encoder_padding_mask'] else None
                asr_out = model.asr_ctc_decoder(encoder_out=encoder_hidden, encoder_padding_mask=encoder_padding_mask)
                st_out = model.st_ctc_decoder(encoder_out=encoder_hidden, encoder_padding_mask=encoder_padding_mask)
                return {
                    'asr_log_probs': asr_out['log_probs'],
                    'st_log_probs': st_out['log_probs'],
                    'mt_logits': None,
                    'unit_log_probs': None,
                }
            else:
                raise
    model.forward = forward_without_unit
    
    # Statistics
    stats = {
        'total': 0,
        'empty': 0,
        'has_content': 0,
        'total_pred_len': 0,
        'total_ref_len': 0,
        'blank_ratios': [],
        'pred_lengths': [],
        'ref_lengths': [],
    }
    
    # Decode samples
    num_samples = min(num_samples, len(dataset))
    
    print(f"\n📊 Decoding {num_samples} samples from {dataset_type} set...")
    print("="*70)
    
    for idx in range(num_samples):
        sample = dataset[idx]
        batch = collate_s2st_batches([sample])
        
        with torch.no_grad():
            output = model(
                src_tokens=batch['speech'],
                src_lengths=batch['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch.get('target_lengths'),
            )
            
            st_log_probs = output['st_log_probs']  # [T, B, V]
            log_probs = st_log_probs.transpose(0, 1)  # [B, T, V]
            
            pred_tokens = decode_ctc_greedy(log_probs[0], blank_id=1)
            
            if len(pred_tokens) > 0:
                pred_text = tokenizer.decode(pred_tokens)
            else:
                pred_text = ""
            
            gt_tokens = batch['target_text'][0][:batch['target_lengths'][0]-1].tolist()
            gt_text = tokenizer.decode(gt_tokens)
            
            # Blank ratio
            argmax_ids = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
            blank_count = (argmax_ids == 1).sum().item()
            total_frames = argmax_ids.size(0)
            blank_ratio = blank_count / total_frames * 100 if total_frames > 0 else 0
            
            # Statistics
            stats['total'] += 1
            stats['total_pred_len'] += len(pred_tokens)
            stats['total_ref_len'] += len(gt_tokens)
            stats['pred_lengths'].append(len(pred_tokens))
            stats['ref_lengths'].append(len(gt_tokens))
            stats['blank_ratios'].append(blank_ratio)
            
            if len(pred_tokens) == 0:
                stats['empty'] += 1
            else:
                stats['has_content'] += 1
            
            print(f"\n[Sample {idx+1}/{num_samples}]")
            print(f"  GT:   {gt_text[:100]}{'...' if len(gt_text) > 100 else ''}")
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
    print(f"  Length ratio: {stats['total_pred_len']/stats['total_ref_len']*100:.1f}%")
    print(f"  Min pred length: {min(stats['pred_lengths'])}")
    print(f"  Max pred length: {max(stats['pred_lengths'])}")
    print(f"  Avg blank ratio: {sum(stats['blank_ratios'])/len(stats['blank_ratios']):.1f}%")
    print("="*70)
    
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='dev', choices=['dev', 'train'])
    args = parser.parse_args()
    
    stats = decode_checkpoint(args.checkpoint, args.num_samples, args.dataset)
    
    # Diagnosis
    print("\n" + "="*70)
    print("🔍 진단 결과:")
    print("="*70)
    avg_pred_len = stats['total_pred_len'] / stats['total']
    avg_ref_len = stats['total_ref_len'] / stats['total']
    length_ratio = avg_pred_len / avg_ref_len * 100
    
    if avg_pred_len < 2:
        print("❌ 여전히 's'만 출력 → Output Layer 재학습만으로는 부족")
        print("   → 추가 조치 필요 (Beam Search, Length Penalty 등)")
    elif length_ratio < 20:
        print("⚠️  출력이 매우 짧음 → 부분적 개선")
        print("   → Beam Search 또는 Length Penalty 적용 권장")
    elif length_ratio < 50:
        print("✅ 출력 길이가 개선됨 → 괜찮은 진전")
        print("   → 추가 epoch 또는 디코딩 개선으로 더 향상 가능")
    else:
        print("✅ 출력 길이가 충분함 → 좋은 진전!")
        print("   → 현재 방향이 올바름")
    
    print("="*70)

if __name__ == '__main__':
    main()

