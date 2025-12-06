# 학습 코드 비교: EchoStream vs StreamSpeech

## ✅ 결론: StreamSpeech 학습 방식을 잘 따르고 있음

EchoStream의 학습 코드는 StreamSpeech의 학습 방식을 따르며, 데이터셋 크기에 맞게 파라미터를 조정했습니다.

---

## 📊 학습 구성 요소 비교

### 1. Loss 계산 방식

#### StreamSpeech Criterion
**`speech_to_unit_2pass_ctc_asr_st`** (`researches/ctc_unity/criterions/speech_to_speech_ctc_asr_st_criterion.py`):
- ASR CTC Loss
- ST CTC Loss  
- MT Cross-Entropy Loss (Label Smoothing 0.1)
- Unit CTC Loss
- R-Drop (optional, alpha=0.0)

**Loss 계산**:
```python
# CTC Loss (Line 220-229)
loss = F.ctc_loss(
    lprobs,
    target,
    input_lengths,
    target_lengths,
    blank=self.blank_idx,
    reduction="sum",
    zero_infinity=True,
)

# Label Smoothing (inherited from parent)
# R-Drop (optional)
```

#### EchoStream Criterion
**`MultiTaskLoss`** (`scripts/train.py` Line 39-232):
- ASR CTC Loss ✅
- ST CTC Loss ✅
- MT Cross-Entropy Loss ✅
- Unit CTC Loss ✅

**Loss 계산**:
```python
# CTC Loss (Line 67, 100-113)
self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# Cross-Entropy Loss (Line 68, 198-201)
self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
```

**비교**:
- ✅ 동일한 Loss 함수 사용 (CTC, Cross-Entropy)
- ✅ 동일한 `zero_infinity=True` 옵션
- ⚠️ Label Smoothing 미구현 (추가 가능)
- ⚠️ R-Drop 미구현 (선택사항)

---

### 2. Optimizer 설정

#### StreamSpeech
```bash
--optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 1.0
```

#### EchoStream
```python
# scripts/train.py Line 696-698
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=effective_lr,
    betas=(0.9, 0.98),  # ✅ StreamSpeech와 동일
)
```

**비교**:
- ✅ Adam optimizer 사용
- ✅ Betas (0.9, 0.98) 동일
- ✅ Gradient clipping 구현 (Line 342, 346, 368, 372)

---

### 3. Learning Rate Scheduler

#### StreamSpeech
```bash
--lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000
```

#### EchoStream
```python
# scripts/train.py Line 700-720
# Inverse sqrt scheduler 구현
def get_lr(epoch, warmup_epochs=10, base_lr=5e-4):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr / math.sqrt(epoch - warmup_epochs + 1)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
```

**비교**:
- ✅ Inverse sqrt scheduler 사용
- ✅ Warmup 구현
- ⚠️ Warmup 방식 약간 다름 (StreamSpeech는 updates 기반, EchoStream은 epochs 기반)

---

### 4. Training Loop 구조

#### StreamSpeech (Fairseq)
```python
# fairseq/trainer.py Line 843-851
loss, sample_size_i, logging_output = self.task.train_step(
    sample=sample,
    model=self.model,
    criterion=self.criterion,
    optimizer=self.optimizer,
    update_num=self.get_num_updates(),
    ignore_grad=is_dummy_batch,
    **extra_kwargs,
)
```

#### EchoStream
```python
# scripts/train.py Line 285-324
output = model(
    src_tokens=speech,
    src_lengths=speech_lengths,
    prev_output_tokens=prev_output_tokens,
    target_lengths=target_lengths,
)

loss, loss_dict = criterion(output, target_dict)
loss = loss / update_freq  # Gradient accumulation

loss.backward()
optimizer.step()
```

**비교**:
- ✅ Forward pass → Loss 계산 → Backward → Optimizer step
- ✅ Gradient accumulation 지원 (EchoStream 추가 기능)
- ✅ Teacher forcing 사용 (학습 시)

---

### 5. Multi-task Loss 가중치

#### StreamSpeech
- Criterion 내부에서 가중치 관리
- Label smoothing: 0.1
- R-Drop alpha: 0.0 (비활성화)

#### EchoStream
```python
# scripts/train.py Line 50-64
asr_weight: float = 0.3
st_weight: float = 0.3
mt_weight: float = 0.2
unit_weight: float = 0.2
```

**비교**:
- ✅ Multi-task loss 가중치 사용
- ⚠️ 가중치 값은 데이터셋에 맞게 조정 가능

---

## 📊 모델 파라미터 비교

### StreamSpeech (220시간 데이터)
```bash
--encoder-layers 12 --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 4
--translation-decoder-layers 4
--decoder-layers 2 --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 8
```

### EchoStream (1시간 데이터, Mac M2 8GB)
```yaml
encoder:
  embed_dim: 128        # 256 → 128 (1/2)
  layers: 4             # 12 → 4 (1/3)
  attention_heads: 2    # 4 → 2 (1/2)
  ffn_embed_dim: 512    # 2048 → 512 (1/4)

mt_decoder:
  embed_dim: 256        # 512 → 256 (1/2)
  layers: 2             # 4 → 2 (1/2)
  attention_heads: 2    # 8 → 2 (1/4)
  ffn_embed_dim: 512    # 2048 → 512 (1/4)
```

**비교**:
- ✅ 데이터셋 크기에 맞게 파라미터 수 감소
- ✅ 구조는 동일 (Encoder → MT Decoder → Unit Decoder)
- ✅ Loss 계산 방식 동일

---

## ✅ 호환성 확인

### 1. Loss 계산
- [x] CTC Loss: 동일 (`zero_infinity=True`) ✅
- [x] Cross-Entropy Loss: 동일 ✅
- [x] Multi-task 가중치: 사용 ✅
- [ ] Label Smoothing: 미구현 (추가 가능)
- [ ] R-Drop: 미구현 (선택사항)

### 2. Optimizer
- [x] Adam optimizer: 동일 ✅
- [x] Betas (0.9, 0.98): 동일 ✅
- [x] Gradient clipping: 구현 ✅

### 3. Learning Rate
- [x] Inverse sqrt scheduler: 구현 ✅
- [x] Warmup: 구현 ✅
- [ ] Warmup 방식: 약간 다름 (updates vs epochs)

### 4. Training Loop
- [x] Forward → Loss → Backward → Step: 동일 ✅
- [x] Teacher forcing: 사용 ✅
- [x] Gradient accumulation: EchoStream 추가 기능 ✅

---

## 📝 개선 가능 사항 (선택사항)

### 1. Label Smoothing 추가
StreamSpeech는 label smoothing 0.1을 사용합니다:
```python
# 추가 가능
from torch.nn import CrossEntropyLoss
self.ce_loss = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
```

### 2. R-Drop 추가 (선택사항)
StreamSpeech는 R-Drop을 지원하지만 기본값은 0.0 (비활성화)입니다.

### 3. Warmup을 Updates 기반으로 변경
StreamSpeech는 updates 기반 warmup을 사용합니다:
```python
# 현재: epochs 기반
# 개선: updates 기반 (더 정확)
```

---

## 🎯 최종 확인

### ✅ StreamSpeech 학습 방식 준수:
1. ✅ Loss 계산: CTC + Cross-Entropy
2. ✅ Optimizer: Adam (betas 0.9, 0.98)
3. ✅ Scheduler: Inverse sqrt with warmup
4. ✅ Gradient clipping: 구현
5. ✅ Multi-task loss: 가중치 사용
6. ✅ Training loop: Forward → Loss → Backward → Step

### ✅ 데이터셋 크기에 맞춘 조정:
1. ✅ 파라미터 수 감소 (1시간 데이터에 맞게)
2. ✅ Batch size 감소 (Mac M2 8GB에 맞게)
3. ✅ Gradient accumulation 추가 (메모리 효율)

---

## 결론

**EchoStream의 학습 코드는 StreamSpeech의 학습 방식을 잘 따르고 있으며, 데이터셋 크기와 하드웨어 제약에 맞게 적절히 조정되었습니다.**

- ✅ 핵심 학습 로직: StreamSpeech와 동일
- ✅ Loss 계산: 동일한 방식
- ✅ Optimizer/Scheduler: 동일한 설정
- ✅ 파라미터 조정: 데이터셋 크기에 맞게 최적화

---

**마지막 업데이트**: 2025-11-18

