#!/bin/bash

LOG="results/train_100_penalty3_20251206_004352.log"

echo "========================================================================"
echo "📊 100-Sample 품질 체크 (로그 기반)"
echo "========================================================================"
echo ""

# 1. Blank ratio 분석
echo "1️⃣ Blank Ratio 분석:"
echo ""
BLANK_RATIOS=$(grep "blank_ratio=" $LOG | tail -50 | grep -o "blank_ratio=[0-9.]*%" | cut -d= -f2 | cut -d% -f1)

if [ ! -z "$BLANK_RATIOS" ]; then
  AVG=$(echo "$BLANK_RATIOS" | awk '{sum+=$1; count++} END {print sum/count}')
  MAX=$(echo "$BLANK_RATIOS" | sort -n | tail -1)
  
  echo "  최근 50개 배치:"
  echo "  - 평균 blank_ratio: $AVG%"
  echo "  - 최대 blank_ratio: $MAX%"
  
  if (( $(echo "$AVG < 5" | bc -l) )); then
    echo "  ✅ 완벽! (< 5%)"
  elif (( $(echo "$AVG < 30" | bc -l) )); then
    echo "  ⚠️  약간 높음 (5-30%)"
  else
    echo "  ❌ 높음! (> 30%)"
  fi
else
  echo "  ❌ 데이터 없음"
fi

echo ""
echo "2️⃣ Loss 추세:"
echo ""
echo "  Dev Loss 변화:"
grep "\[Dev\]" $LOG | tail -10 | awk '{print "    Epoch", NR+60": "$9}'

echo ""
echo "3️⃣ 학습 안정성:"
echo ""
# Epoch 60-70의 평균 blank_ratio
RECENT_BLANK=$(grep "blank_ratio=0.0%" $LOG | tail -100 | wc -l)
TOTAL_RECENT=$(grep "blank_ratio=" $LOG | tail -100 | wc -l)

if [ $TOTAL_RECENT -gt 0 ]; then
  ZERO_RATIO=$((RECENT_BLANK * 100 / TOTAL_RECENT))
  echo "  최근 100개 배치 중:"
  echo "  - blank_ratio=0.0%: $RECENT_BLANK / $TOTAL_RECENT ($ZERO_RATIO%)"
  
  if [ $ZERO_RATIO -gt 80 ]; then
    echo "  ✅ 매우 안정적!"
  elif [ $ZERO_RATIO -gt 50 ]; then
    echo "  ⚠️  보통"
  else
    echo "  ❌ 불안정"
  fi
fi

echo ""
echo "========================================================================"
