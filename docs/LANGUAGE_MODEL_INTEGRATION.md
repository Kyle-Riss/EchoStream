# Language Model (LM) 통합 가이드

## 📍 LM 통합 위치

**후처리 방식 (현재 구현)** - 영향 최소화

MT Decoder 출력 후 텍스트를 개선하는 방식입니다.

**위치**: `models/word_level_translator.py`의 `translate_word` 메서드 (Line 168-170)

**구현**:
```python
# 5. Decode translation
translation = self.tokenizer.decode(mt_tokens.tolist())
translation = self._clean_text(translation)

# 5-1. Language Model 후처리 (영향 최소화: 번역 텍스트 개선)
if self.lm_model is not None and translation:
    translation = self._lm_postprocess(translation, mt_tokens)
```

**장점**:
- ✅ **영향 최소화**: MT Decoder 로직 변경 없음
- ✅ 구현이 간단
- ✅ LM 모델을 독립적으로 교체 가능
- ✅ 기존 코드와 독립적으로 동작

**단점**:
- ⚠️ 디코딩 타임 통합보다 덜 효율적 (하지만 안정적)
- ⚠️ Beam search를 별도로 구현해야 함 (선택사항)

---

### 참고: 디코딩 타임 통합 (StreamSpeech 방식)

StreamSpeech는 디코딩 중에 LM을 사용하지만, EchoStream은 후처리 방식을 채택했습니다.

**StreamSpeech 방식** (`sequence_generator.py` Line 348-354):
```python
if self.lm_model is not None and not aux_task_name:
    lm_out = self.lm_model(tokens[:, : step + 1])
    probs = self.lm_model.get_normalized_probs(
        lm_out, log_probs=True, sample=None
    )
    probs = probs[:, -1, :] * self.lm_weight
    lprobs += probs  # MT logits에 LM 확률 추가
```

**EchoStream 후처리 방식**:
- MT Decoder는 그대로 동작
- 출력 텍스트를 LM으로 개선
- 더 안전하고 독립적

## 🔧 구현 방법

### 후처리 방식 (현재 구현)

#### 1. LM 모델 로드

```python
# models/word_level_translator.py
class WordLevelTranslator:
    def __init__(
        self,
        ...,
        lm_model=None,  # Fairseq LM 모델 또는 LanguageModelWrapper
        lm_weight=0.5,  # LM 가중치
    ):
        # Language Model (후처리 방식: 영향 최소화)
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()
```

#### 2. `translate_word`에서 후처리 호출

```python
def translate_word(...):
    # ... (기존 MT decoding) ...
    
    # 5. Decode translation
    translation = self.tokenizer.decode(mt_tokens.tolist())
    translation = self._clean_text(translation)
    
    # 5-1. Language Model 후처리 (영향 최소화)
    if self.lm_model is not None and translation:
        translation = self._lm_postprocess(translation, mt_tokens)
    
    # ... (기존 코드) ...
```

#### 3. `_lm_postprocess` 구현

```python
def _lm_postprocess(
    self,
    translation: str,
    mt_tokens: torch.Tensor,
) -> str:
    """
    Language Model 후처리로 번역 텍스트 개선.
    
    영향 최소화: MT Decoder 출력 후 텍스트만 개선
    """
    if self.lm_model is None:
        return translation
    
    try:
        # TODO: 여러 후보 생성 후 LM rescore
        # 현재는 원본 반환 (추후 구현)
        return translation
        
    except Exception as e:
        logger.warning(f"LM postprocessing failed: {e}")
        return translation
```

## 📊 StreamSpeech 참고

StreamSpeech에서는 `sequence_generator.py`의 Line 348-354에서 LM을 사용합니다:

```python
if self.lm_model is not None and not aux_task_name:
    lm_out = self.lm_model(tokens[:, : step + 1])
    probs = self.lm_model.get_normalized_probs(
        lm_out, log_probs=True, sample=None
    )
    probs = probs[:, -1, :] * self.lm_weight
    lprobs += probs  # MT logits에 LM 확률 추가
```

## 🎯 현재 구현: 후처리 방식

**EchoStream은 후처리 방식을 채택했습니다:**

1. ✅ **영향 최소화**: MT Decoder 로직 변경 없음
2. ✅ **안정성**: 기존 코드와 독립적으로 동작
3. ✅ **유연성**: LM 모델을 쉽게 교체 가능
4. ✅ **단순성**: 구현이 간단하고 유지보수 용이

**디코딩 타임 통합 (StreamSpeech 방식)과 비교:**
- StreamSpeech: 디코딩 중에 LM 사용 → 더 정확하지만 복잡
- EchoStream: 후처리로 텍스트 개선 → 안정적이고 독립적

## 📝 구현 체크리스트

- [ ] LM 모델 로드 (Fairseq LM 또는 다른 LM)
- [ ] `WordLevelTranslator.__init__`에 `lm_model`, `lm_weight` 추가
- [ ] `_generate_mt_tokens`에서 LM 확률 추가
- [ ] Beam search 구현 (선택사항, greedy도 가능)
- [ ] Config 파일에 LM 경로 및 가중치 추가
- [ ] 테스트 및 성능 평가

---

**마지막 업데이트**: 2025-11-18

