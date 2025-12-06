#!/bin/bash

LOG=$(ls -t results/train_100_penalty3_*.log | head -1)

while true; do
  clear
  echo "========================================================================="
  echo "📊 Blank Penalty=3.0 실시간 모니터링"
  echo "========================================================================="
  echo "Time: $(date +%H:%M:%S)"
  echo ""
  
  # Current epoch
  echo "Current Status:"
  tail -50 $LOG | grep "Epoch [0-9]" | tail -1
  echo ""
  
  # Recent blank_ratio (last 8)
  echo "Recent blank_ratio:"
  tail -150 $LOG | grep "blank_ratio" | tail -8
  echo ""
  
  # Dev Loss
  echo "Dev Loss:"
  tail -100 $LOG | grep "\[Dev\]" | tail -2
  echo ""
  
  echo "========================================================================="
  echo "✅ = blank_ratio=0.0% (목표)"
  echo "⚠️  = blank_ratio=1-50% (위험)"
  echo "❌ = blank_ratio=100% (collapse)"
  echo ""
  echo "Press Ctrl+C to stop monitoring"
  echo "========================================================================="
  
  sleep 60
done
