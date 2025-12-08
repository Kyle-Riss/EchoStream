#!/usr/bin/env python3
"""
Smoke test: Decode a few dev samples after 1 epoch training.

Checks for:
- UNK spam patterns
- s-only spam patterns
- Basic word generation

Usage:
  python scripts/smoke_test_decode.py \
    --checkpoint checkpoints_st_lm_vocab5000/checkpoint_epoch_1.pt \
    --manifest data/dev_sampled.retokenized.tsv \
    --vocab-path data/tgt_unigram5000/spm_unigram_en.model \
    --num-samples 3
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sentencepiece as spm
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.st_ctc_lm import STCTCWithLM
from torch.utils.data import DataLoader


def decode_ctc_greedy(log_probs, blank_id=1):
    """Greedy CTC decoding."""
    # log_probs: [T, V]
    tokens = log_probs.argmax(dim=-1)  # [T]
    
    # Remove blanks and collapse repeats
    filtered = []
    prev = blank_id
    for t in tokens:
        if t != blank_id and t != prev:
            filtered.append(t.item())
        prev = t
    
    return filtered


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, default="data/tgt_unigram5000/spm_unigram_en.model")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.vocab_path)
    vocab_size = sp.get_piece_size()
    unk_id = sp.unk_id()
    pad_id = sp.pad_id()
    
    print("="*70)
    print("🔥 Smoke Test: Decoding Check")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocab size: {vocab_size}")
    print(f"UNK ID: {unk_id}")
    print()
    
    # Load model
    model = STCTCWithLM(
        vocab_size=vocab_size,
        encoder_dim=128,
        encoder_layers=4,
        decoder_dim=128,
        decoder_layers=3,
        decoder_heads=4,
        decoder_ffn_dim=512,
        mt_decoder_layers=4,
        mt_decoder_heads=4,
        mt_decoder_ffn_dim=1024,
        use_mt_decoder=True,
        dropout=0.1,
        pad_idx=pad_id,
        bos_idx=sp.bos_id(),
        eos_idx=sp.eos_id(),
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Model loaded\n")
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path=args.manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.vocab_path,
        text_level="word",
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_s2st_batches,
        num_workers=0,
    )
    
    print("="*70)
    print("📊 Decoding Results (First {} samples)".format(args.num_samples))
    print("="*70)
    print()
    
    unk_count_total = 0
    token_count_total = 0
    samples_checked = 0
    
    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break
        
        src_feats = batch["speech"].to(device)
        src_lengths = batch["speech_lengths"].to(device)
        tgt_text = batch.get("target_text", [""])[0]
        
        # Forward
        outputs = model(src_feats, src_lengths, compute_loss=False)
        
        # CTC decoding
        st_log_probs = outputs["st_log_probs"]  # [T, B, V]
        st_log_probs = st_log_probs.squeeze(1)  # [T, V]
        
        hyp_tokens = decode_ctc_greedy(st_log_probs, blank_id=pad_id)
        
        # Decode to text
        hyp_text = sp.decode(hyp_tokens)
        
        # Count UNK
        unk_count = sum(1 for t in hyp_tokens if t == unk_id)
        token_count = len(hyp_tokens)
        unk_count_total += unk_count
        token_count_total += token_count
        samples_checked += 1
        
        # Check for problematic patterns
        issues = []
        if unk_count > token_count * 0.5:
            issues.append("⚠️  UNK spam (>50%)")
        if len(set(hyp_tokens)) < 3:
            issues.append("⚠️  Low diversity")
        if hyp_text.strip() == "":
            issues.append("⚠️  Empty output")
        
        status = "✅" if not issues else "❌"
        
        print(f"Sample {i+1}:")
        print(f"  Reference: {tgt_text[:80]}...")
        print(f"  Hypothesis: {hyp_text[:80]}...")
        print(f"  Tokens: {len(hyp_tokens)} | UNK: {unk_count} ({unk_count/token_count*100:.1f}%)")
        if issues:
            print(f"  {status} Issues: {', '.join(issues)}")
        else:
            print(f"  {status} OK")
        print()
    
    # Summary
    avg_unk_ratio = (unk_count_total / token_count_total * 100) if token_count_total > 0 else 0
    
    print("="*70)
    print("📈 Summary")
    print("="*70)
    print(f"  Samples checked: {samples_checked}")
    print(f"  Total tokens: {token_count_total}")
    print(f"  Total UNK: {unk_count_total}")
    print(f"  Average UNK ratio: {avg_unk_ratio:.2f}%")
    print()
    
    if avg_unk_ratio < 5.0:
        print("  ✅ UNK ratio is acceptable (< 5%)")
        print("  ✅ No UNK spam detected")
    elif avg_unk_ratio < 20.0:
        print("  ⚠️  UNK ratio is moderate (5-20%)")
        print("  → May need further tuning")
    else:
        print("  ❌ UNK ratio is HIGH (> 20%)")
        print("  ❌ UNK spam detected - check training")
    
    print("="*70)


if __name__ == "__main__":
    main()

