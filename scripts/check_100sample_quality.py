#!/usr/bin/env python3
"""
Check 100-sample fine-tuning quality before scaling to 2,400
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import sentencepiece as spm
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import build_echostream_model, EchoStreamConfig

def ctc_collapse(tokens, blank_id=1):
    """CTC greedy decode"""
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
    print("🔍 100-Sample Quality Check")
    print("="*70)
    
    # Load SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.load('data/tgt_unigram6000/spm_unigram_en.model')
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path='data/train_100.tsv',
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
    
    # Load checkpoint (최신 100-sample)
    checkpoint = torch.load('checkpoints_simple_100_penalty3/checkpoint_best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print("\n" + "="*70)
    print("1️⃣ 디코딩 결과 (첫 5개 샘플)")
    print("="*70)
    
    # CTC loss function
    ctc_loss_fn = torch.nn.CTCLoss(blank=1, reduction='none', zero_infinity=True)
    
    normalized_losses = []
    
    for i in range(min(5, len(dataset))):
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
        
        st_log_probs = output['st_log_probs']  # [T, B, V]
        
        # Greedy decode
        tokens = st_log_probs.argmax(dim=-1)[:, 0]  # [T]
        collapsed = ctc_collapse(tokens.tolist(), blank_id=1)
        pred_text = sp.decode_ids(collapsed) if collapsed else ""
        
        # Compute per-sample loss
        tgt_tokens = batch['target_text'][0][:batch['target_lengths'][0]-1]
        # Filter (same as training)
        valid_mask = (tgt_tokens != 1) & (tgt_tokens != 11) & (tgt_tokens != 3)
        clean_target = tgt_tokens[valid_mask]
        
        if clean_target.numel() > 0:
            input_len = torch.tensor([st_log_probs.size(0)], dtype=torch.long)
            target_len = torch.tensor([clean_target.numel()], dtype=torch.long)
            
            # Add blank penalty (same as training)
            st_log_probs_adj = st_log_probs.clone()
            st_log_probs_adj[:, :, 1] = st_log_probs_adj[:, :, 1] - 3.0
            
            loss = ctc_loss_fn(st_log_probs_adj, clean_target.unsqueeze(0), input_len, target_len)
            normalized_loss = loss.item() / clean_target.numel()
            normalized_losses.append(normalized_loss)
        else:
            normalized_loss = 0.0
        
        # Blank ratio
        blank_ratio = (tokens == 1).float().mean().item() * 100
        
        print(f"\n[Sample {i+1}]")
        print(f"  GT:   {ref_text}")
        print(f"  Pred: {pred_text if pred_text else '(empty)'}")
        print(f"  Stats:")
        print(f"    - Pred tokens: {len(collapsed)}")
        print(f"    - GT tokens: {len(sample['tgt_tokens'])}")
        print(f"    - blank_ratio: {blank_ratio:.1f}%")
        print(f"    - Normalized loss: {normalized_loss:.2f}")
    
    print("\n" + "="*70)
    print("2️⃣ Normalized Loss 분석 (전체 100개)")
    print("="*70)
    
    # Compute for all 100 samples
    all_normalized_losses = []
    blank_ratios = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        batch = collate_s2st_batches([sample])
        
        with torch.no_grad():
            output = model(
                src_tokens=batch['speech'],
                src_lengths=batch['speech_lengths'],
                prev_output_tokens=None,
                target_lengths=batch.get('target_lengths'),
            )
        
        st_log_probs = output['st_log_probs']
        
        # Blank ratio
        tokens = st_log_probs.argmax(dim=-1)[:, 0]
        blank_ratio = (tokens == 1).float().mean().item() * 100
        blank_ratios.append(blank_ratio)
        
        # Normalized loss
        tgt_tokens = batch['target_text'][0][:batch['target_lengths'][0]-1]
        valid_mask = (tgt_tokens != 1) & (tgt_tokens != 11) & (tgt_tokens != 3)
        clean_target = tgt_tokens[valid_mask]
        
        if clean_target.numel() > 0:
            input_len = torch.tensor([st_log_probs.size(0)], dtype=torch.long)
            target_len = torch.tensor([clean_target.numel()], dtype=torch.long)
            
            st_log_probs_adj = st_log_probs.clone()
            st_log_probs_adj[:, :, 1] = st_log_probs_adj[:, :, 1] - 3.0
            
            loss = ctc_loss_fn(st_log_probs_adj, clean_target.unsqueeze(0), input_len, target_len)
            normalized_loss = loss.item() / clean_target.numel()
            all_normalized_losses.append(normalized_loss)
    
    # Statistics
    import numpy as np
    avg_norm_loss = np.mean(all_normalized_losses)
    std_norm_loss = np.std(all_normalized_losses)
    avg_blank = np.mean(blank_ratios)
    
    print(f"\n📊 통계 (100 samples):")
    print(f"  평균 normalized loss: {avg_norm_loss:.2f}")
    print(f"  표준편차: {std_norm_loss:.2f}")
    print(f"  평균 blank_ratio: {avg_blank:.1f}%")
    
    print(f"\n🎯 평가:")
    if avg_norm_loss < 4:
        print(f"  ✅ EXCELLENT! (< 4.0)")
        print(f"     → 매우 잘 학습됨")
    elif avg_norm_loss < 6:
        print(f"  ✅ GOOD! (4-6)")
        print(f"     → 그럭저럭 배우고 있음")
    elif avg_norm_loss < 10:
        print(f"  ⚠️  OK (6-10)")
        print(f"     → 학습 중이지만 개선 필요")
    else:
        print(f"  ❌ POOR (> 10)")
        print(f"     → 거의 랜덤 수준")
    
    if avg_blank < 5:
        print(f"  ✅ blank_ratio 정상! (< 5%)")
    elif avg_blank < 30:
        print(f"  ⚠️  blank_ratio 약간 높음 (5-30%)")
    else:
        print(f"  ❌ blank_ratio 높음! (> 30%)")
    
    print("="*70)

if __name__ == '__main__':
    main()

