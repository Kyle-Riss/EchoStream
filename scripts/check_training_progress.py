#!/usr/bin/env python3
"""
Check training progress and automatically run smoke test when epoch 10 is complete.

Usage:
  python scripts/check_training_progress.py --checkpoint-dir checkpoints_st_lm_vocab5000
"""
import argparse
import sys
import os
from pathlib import Path
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_st_lm_vocab5000")
    parser.add_argument("--target-epoch", type=int, default=10)
    parser.add_argument("--manifest", type=str, default="data/dev_sampled.retokenized.tsv")
    parser.add_argument("--vocab-path", type=str, default="data/tgt_unigram5000/spm_unigram_en.model")
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    target_checkpoint = checkpoint_dir / f"checkpoint_epoch_{args.target_epoch}.pt"
    
    print("="*70)
    print("🔍 Training Progress Check")
    print("="*70)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Target epoch: {args.target_epoch}")
    print()
    
    # Check if target checkpoint exists
    if target_checkpoint.exists():
        print(f"✅ Epoch {args.target_epoch} checkpoint found!")
        print(f"   Path: {target_checkpoint}")
        print(f"   Size: {target_checkpoint.stat().st_size / 1024 / 1024:.2f} MB")
        print()
        
        # List all checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        print(f"📋 Available checkpoints: {len(checkpoints)}")
        for ckpt in checkpoints:
            epoch_num = ckpt.stem.split("_")[-1]
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"   - Epoch {epoch_num}: {size_mb:.2f} MB")
        print()
        
        # Run smoke test
        print("="*70)
        print("🔥 Running Smoke Test")
        print("="*70)
        print()
        
        cmd = [
            "python", "scripts/smoke_test_decode.py",
            "--checkpoint", str(target_checkpoint),
            "--manifest", args.manifest,
            "--vocab-path", args.vocab_path,
            "--num-samples", "5",
            "--device", "cpu",
        ]
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print()
            print("="*70)
            print("✅ Smoke test completed successfully!")
            print("="*70)
        else:
            print()
            print("="*70)
            print("⚠️  Smoke test completed with warnings")
            print("="*70)
    else:
        print(f"⏳ Epoch {args.target_epoch} checkpoint not found yet.")
        print(f"   Waiting for: {target_checkpoint}")
        print()
        
        # List available checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            print(f"📋 Current progress: {len(checkpoints)} checkpoints found")
            latest = checkpoints[-1]
            epoch_num = latest.stem.split("_")[-1]
            print(f"   Latest: Epoch {epoch_num}")
        else:
            print("   No checkpoints found yet.")


if __name__ == "__main__":
    main()

