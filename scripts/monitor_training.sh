#!/bin/bash

LOG_FILE="results/train_100_finetune_v2_20251206_002232.log"

echo "========================================================================="
echo "📊 Training Monitor - Curriculum Learning (10→100)"
echo "========================================================================="
echo ""

# 현재 Epoch
echo "Current Epoch:"
tail -50 $LOG_FILE | grep "Epoch [0-9]" | tail -1
echo ""

# 최근 5개 batch의 blank_ratio
echo "Recent blank_ratio (last 5):"
tail -100 $LOG_FILE | grep "blank_ratio" | tail -5
echo ""

# 최근 Loss
echo "Recent Loss (last 5 batches):"
tail -50 $LOG_FILE | grep "Batch.*Loss" | tail -5
echo ""

# Dev Loss (if any)
echo "Dev Loss:"
tail -200 $LOG_FILE | grep "\[Dev\]" | tail -3
echo ""

# blank_log_prob 추이
echo "blank_log_prob trend:"
tail -100 $LOG_FILE | grep "blank_ratio" | tail -5 | awk '{for(i=1;i<=NF;i++){if($i~/blank_log_prob/){print $(i+1)}}}'
echo ""

echo "========================================================================="
echo "Legend:"
echo "  ✅ blank_ratio=0.0% : 정상 (목표)"
echo "  ⚠️  blank_ratio=1-50% : 위험 (일부 collapse)"
echo "  ❌ blank_ratio=100% : 실패 (완전 collapse)"
echo ""
echo "  blank_log_prob < -5.0 : 정상 (blank 선호 안 함)"
echo "  blank_log_prob > -2.0 : 위험 (blank 강하게 선호)"
echo "========================================================================="

