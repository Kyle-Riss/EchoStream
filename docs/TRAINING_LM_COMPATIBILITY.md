# 학습 시 LM 후처리 호환성 확인

## ✅ 결론: 학습에는 전혀 문제 없음

LM 후처리는 **추론 시에만** 사용되며, 학습 시에는 사용되지 않습니다.

---

## 📊 학습 vs 추론 동작 비교

### 1. 학습 시 (`model.training = True`)

**`scripts/train.py`**:
```python
# Line 256
model.train()  # 학습 모드 설정

# Line 285-290: Teacher forcing으로 학습
output = model(
    src_tokens=speech,
    src_lengths=speech_lengths,
    prev_output_tokens=prev_output_tokens,  # Teacher forcing
    target_lengths=target_lengths,
)
```

**`models/echostream_model.py`**:
```python
# Line 213: 학습 시에는 greedy decoding 스킵
if not self.training:
    # 추론 시에만 greedy decoding 수행
    st_tokens_greedy = st_log_probs.argmax(dim=-1)
    # ... CTC collapse ...
else:
    # 학습 시: 이 블록 실행 안 됨

# Line 250-251: Teacher forcing 사용
if prev_output_tokens is not None:
    mt_input_tokens = prev_output_tokens  # Teacher forcing
    mt_out = self.mt_decoder(
        prev_output_tokens=mt_input_tokens,
        encoder_out=encoder_out,
    )
```

**학습 시 동작**:
- ✅ `model.train()` 설정
- ✅ `prev_output_tokens`를 teacher forcing으로 사용
- ✅ Greedy decoding 스킵 (`if not self.training:` 조건)
- ✅ LM 후처리 사용 안 함 (`WordLevelTranslator` 사용 안 함)

---

### 2. 추론 시 (`model.training = False`)

**`WordLevelTranslator.translate_word()`**:
```python
# Line 169-171: LM 후처리 (추론 시에만 호출)
# 5-1. Language Model 후처리 (영향 최소화: 번역 텍스트 개선)
if self.lm_model is not None and translation:
    translation = self._lm_postprocess(translation, mt_tokens)
```

**추론 시 동작**:
- ✅ `model.eval()` 설정
- ✅ Greedy decoding 수행
- ✅ `WordLevelTranslator` 사용 (서버, 평가 등)
- ✅ LM 후처리 적용

---

## 🔍 StreamSpeech 학습 방식 확인

**StreamSpeech 학습** (`researches/ctc_unity/models/streamspeech_model.py`):
```python
def forward(
    self,
    src_tokens,
    src_lengths,
    prev_output_tokens,      # Teacher forcing
    prev_output_tokens_mt,  # Teacher forcing
    streaming_config=None,
    ...
):
    # 학습 시: prev_output_tokens를 teacher forcing으로 사용
    # 추론 시: greedy decoding 또는 streaming agent 사용
```

**StreamSpeech LM 사용**:
- 학습 시: LM 사용 안 함
- 추론 시: `agent/sequence_generator.py`에서 LM 사용 (Line 348-354)

---

## ✅ EchoStream 학습 호환성

### 학습 시:
1. ✅ **Teacher forcing 사용**: `prev_output_tokens` 제공
2. ✅ **Greedy decoding 스킵**: `if not self.training:` 조건
3. ✅ **LM 후처리 사용 안 함**: `WordLevelTranslator` 호출 안 됨
4. ✅ **StreamSpeech와 동일**: Teacher forcing으로 학습

### 추론 시:
1. ✅ **Greedy decoding 수행**: `if not self.training:` 블록 실행
2. ✅ **LM 후처리 적용**: `WordLevelTranslator._lm_postprocess()` 호출
3. ✅ **StreamSpeech와 동일**: 추론 시에만 LM 사용

---

## 📝 확인 사항

### ✅ 확인 완료:
- [x] 학습 시 `model.train()` 설정
- [x] 학습 시 teacher forcing 사용
- [x] 학습 시 greedy decoding 스킵
- [x] 학습 시 `WordLevelTranslator` 사용 안 함
- [x] LM 후처리는 추론 시에만 사용
- [x] StreamSpeech 학습 방식과 동일

### 🎯 결론:
**LM 후처리는 학습에 전혀 영향을 주지 않습니다!**

- 학습: Teacher forcing으로 정상 학습
- 추론: LM 후처리로 번역 품질 개선

---

**마지막 업데이트**: 2025-11-18

