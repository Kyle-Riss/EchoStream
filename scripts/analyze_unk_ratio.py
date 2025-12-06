#!/usr/bin/env python3
"""
Analyze UNK ratio in the dataset to diagnose CTC blank collapse.
"""
import sys
sys.path.insert(0, '.')
import torch
import sentencepiece as spm
from datasets.s2st_dataset import S2STManifestDataset
from collections import defaultdict
import numpy as np

def analyze_unk_ratio(manifest_path, spm_model_path, data_root='data'):
    """Analyze UNK token ratio in the dataset."""
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    
    # Load dataset
    dataset = S2STManifestDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    print("="*70)
    print(f"📊 UNK Ratio Analysis: {manifest_path}")
    print("="*70)
    
    unk_ratios = []
    space_ratios = []
    total_tokens = 0
    total_unk = 0
    total_space = 0
    
    unk_histogram = defaultdict(int)  # Bucket by 10%
    
    clean_samples = []  # Samples with low UNK ratio
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        tokens = sample['tgt_tokens']  # Changed from 'target_text'
        
        # Count UNK (3) and space (11)
        unk_count = (tokens == 3).sum().item()
        space_count = (tokens == 11).sum().item()
        total_count = len(tokens)
        
        unk_ratio = unk_count / total_count if total_count > 0 else 0
        space_ratio = space_count / total_count if total_count > 0 else 0
        
        unk_ratios.append(unk_ratio)
        space_ratios.append(space_ratio)
        
        total_tokens += total_count
        total_unk += unk_count
        total_space += space_count
        
        # Histogram bucket (0-10%, 10-20%, ...)
        bucket = int(unk_ratio * 10)
        unk_histogram[bucket] += 1
        
        # Track clean samples (UNK < 10%)
        if unk_ratio < 0.1:
            clean_samples.append((idx, unk_ratio, tokens.tolist()))
    
    # Statistics
    unk_ratios = np.array(unk_ratios)
    space_ratios = np.array(space_ratios)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total UNK(3): {total_unk} ({total_unk/total_tokens*100:.1f}%)")
    print(f"  Total Space(11): {total_space} ({total_space/total_tokens*100:.1f}%)")
    
    print(f"\n📊 UNK Ratio Distribution:")
    print(f"  Mean: {unk_ratios.mean()*100:.1f}%")
    print(f"  Median: {np.median(unk_ratios)*100:.1f}%")
    print(f"  Std: {unk_ratios.std()*100:.1f}%")
    print(f"  Min: {unk_ratios.min()*100:.1f}%")
    print(f"  Max: {unk_ratios.max()*100:.1f}%")
    
    print(f"\n📊 Histogram (by 10% buckets):")
    for bucket in range(11):
        count = unk_histogram[bucket]
        pct = count / len(dataset) * 100 if len(dataset) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket*10:3d}-{(bucket+1)*10:3d}%: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n✅ Clean Samples (UNK < 10%): {len(clean_samples)}/{len(dataset)} ({len(clean_samples)/len(dataset)*100:.1f}%)")
    
    if len(clean_samples) > 0:
        print(f"\n🎯 Top 5 Cleanest Samples:")
        clean_samples.sort(key=lambda x: x[1])
        for idx, unk_ratio, tokens in clean_samples[:5]:
            # Decode
            try:
                text = sp.decode_ids(tokens)
            except:
                text = "(decode failed)"
            print(f"  Sample {idx}: UNK={unk_ratio*100:.1f}%, tokens={len(tokens)}")
            print(f"    Text: {text[:80]}...")
    
    print("\n" + "="*70)
    print("🎯 Diagnosis:")
    avg_unk = unk_ratios.mean()
    if avg_unk > 0.3:
        print(f"  ❌ UNK ratio is VERY HIGH ({avg_unk*100:.1f}%)")
        print(f"     → This is almost noise, not training data!")
        print(f"     → CTC collapse is expected with this data quality")
    elif avg_unk > 0.2:
        print(f"  ⚠️  UNK ratio is HIGH ({avg_unk*100:.1f}%)")
        print(f"     → Data quality is poor, likely causing collapse")
    elif avg_unk > 0.1:
        print(f"  ⚠️  UNK ratio is moderate ({avg_unk*100:.1f}%)")
        print(f"     → Some impact on training quality")
    else:
        print(f"  ✅ UNK ratio is acceptable ({avg_unk*100:.1f}%)")
    
    print("\n📋 Recommendations:")
    if avg_unk > 0.2:
        print(f"  1. Filter out samples with UNK > 30%")
        print(f"  2. Retrain SentencePiece model with domain-specific corpus")
        print(f"  3. Test with cleanest samples first (UNK < 10%)")
    elif len(clean_samples) > 0:
        print(f"  1. Test overfit with cleanest sample (idx={clean_samples[0][0]})")
        print(f"  2. If successful, gradually add more samples")
    
    print("="*70)
    
    return clean_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/train_mini_10.tsv")
    parser.add_argument("--spm-model", type=str, default="data/tgt_unigram6000/spm_unigram_en.model")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()
    
    clean_samples = analyze_unk_ratio(args.manifest, args.spm_model, args.data_root)

