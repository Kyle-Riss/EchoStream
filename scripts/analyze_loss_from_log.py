#!/usr/bin/env python3
"""
Analyze normalized loss from training log
"""
import re

log_file = 'results/train_100_penalty3_20251206_004352.log'

print("="*70)
print("📊 100-Sample Normalized Loss 분석 (로그 기반)")
print("="*70)

# Parse log
losses = []
target_lens = []

with open(log_file, 'r') as f:
    for line in f:
        # Find lines with both loss and target_len
        if 'target_len=' in line and 'loss=' in line:
            # Extract target_len
            target_match = re.search(r'target_len=(\d+)', line)
            # Extract loss
            loss_match = re.search(r'loss=([\d.]+)', line)
            
            if target_match and loss_match:
                target_len = int(target_match.group(1))
                loss = float(loss_match.group(1))
                
                if target_len > 0:
                    normalized_loss = loss / target_len
                    losses.append(loss)
                    target_lens.append(target_len)

if losses:
    import numpy as np
    normalized = [l/t for l, t in zip(losses, target_lens)]
    
    print(f"\n샘플 수: {len(losses)}")
    print(f"\n평균 normalized loss: {np.mean(normalized):.2f}")
    print(f"표준편차: {np.std(normalized):.2f}")
    print(f"최소: {np.min(normalized):.2f}")
    print(f"최대: {np.max(normalized):.2f}")
    print(f"중앙값: {np.median(normalized):.2f}")
    
    print(f"\n🎯 평가:")
    avg = np.mean(normalized)
    if avg < 4:
        print(f"  ✅ EXCELLENT! (< 4.0)")
        print(f"     → 매우 잘 학습됨")
    elif avg < 6:
        print(f"  ✅ GOOD! (4-6)")
        print(f"     → 그럭저럭 배우고 있음")
    elif avg < 10:
        print(f"  ⚠️  OK (6-10)")
        print(f"     → 학습 중이지만 개선 필요")
    else:
        print(f"  ❌ POOR (> 10)")
        print(f"     → 거의 랜덤 수준")
    
    # Distribution
    print(f"\n분포:")
    bins = [0, 4, 6, 10, 20, 999]
    labels = ["<4", "4-6", "6-10", "10-20", ">20"]
    for i in range(len(bins)-1):
        count = sum(1 for n in normalized if bins[i] <= n < bins[i+1])
        pct = count / len(normalized) * 100
        print(f"  {labels[i]:>6}: {count:3d} ({pct:5.1f}%)")
else:
    print("❌ 로그에서 loss/target_len 정보를 찾을 수 없습니다.")

print("="*70)

# Blank ratio analysis
print("\n" + "="*70)
print("📊 Blank Ratio 분석")
print("="*70)

blank_ratios = []
with open(log_file, 'r') as f:
    for line in f:
        if 'blank_ratio=' in line:
            match = re.search(r'blank_ratio=([\d.]+)%', line)
            if match:
                ratio = float(match.group(1))
                blank_ratios.append(ratio)

if blank_ratios:
    # Last 50 samples
    recent = blank_ratios[-50:]
    avg_blank = np.mean(recent)
    max_blank = np.max(recent)
    
    print(f"\n최근 50개 배치:")
    print(f"  평균 blank_ratio: {avg_blank:.1f}%")
    print(f"  최대 blank_ratio: {max_blank:.1f}%")
    
    if avg_blank < 5:
        print(f"  ✅ 완벽! (< 5%)")
    elif avg_blank < 30:
        print(f"  ⚠️  약간 높음 (5-30%)")
    else:
        print(f"  ❌ 높음! (> 30%)")

print("="*70)
