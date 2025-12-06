#!/usr/bin/env python3
"""
Decode SimpleCTCEncoder overfit results
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import build_echostream_model, EchoStreamConfig
from utils.tokenizer import load_tokenizer

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
    print("🔍 SimpleCTCEncoder Decoding Results")
    print("="*70)
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path='data/train_mini_10.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    # Load tokenizer
    tokenizer = load_tokenizer('data/tgt_unigram6000/spm_unigram_en.model')
    
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
    checkpoint = torch.load('checkpoints_simple_encoder/checkpoint_best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print("\n" + "="*70)
    print("📊 Decoding 10 samples...")
    print("="*70)
    
    total_blank_ratio = 0
    total_samples = 0
    
    for idx in range(min(10, len(dataset))):
        sample = dataset[idx]
        batch = collate_s2st_batches([sample])
        
        with torch.no_grad():
            # Forward
            speech = batch['speech_features']  # [B, T, F]
            speech_lengths = batch['speech_lengths']  # [B]
            
            # Encoder
            encoder_out = model.encoder(speech, speech_lengths)
            enc = encoder_out['encoder_out'][0].transpose(0, 1)  # [B, T, H]
            
            # ST CTC head
            logits = model.st_ctc_head(enc)  # [B, T, V]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Greedy decode
            pred_tokens = decode_ctc_greedy(log_probs[0], blank_id=1)
            
            # Decode text
            pred_text = tokenizer.decode(pred_tokens)
            
            # GT
            gt_tokens = batch['target_text'][0][:batch['target_lengths'][0]-1].tolist()
            gt_text = tokenizer.decode(gt_tokens)
            
            # Blank ratio
            argmax_ids = log_probs[0].argmax(dim=-1)
            blank_count = (argmax_ids == 1).sum().item()
            total_frames = argmax_ids.size(0)
            blank_ratio = blank_count / total_frames * 100
            total_blank_ratio += blank_ratio
            total_samples += 1
            
            print(f"\n[Sample {idx}]")
            print(f"  GT:   {gt_text}")
            print(f"  Pred: {pred_text}")
            print(f"  Tokens: {len(pred_tokens)} (GT: {len(gt_tokens)})")
            print(f"  Blank ratio: {blank_ratio:.1f}%")
    
    print("\n" + "="*70)
    print(f"📊 Summary:")
    print(f"  Average blank ratio: {total_blank_ratio / total_samples:.1f}%")
    print("="*70)

if __name__ == '__main__':
    main()


