# EchoStream - Speech-to-Speech Translation

**EchoStream**은 StreamSpeech 아키텍처를 기반으로 한 Speech-to-Speech Translation 시스템입니다.

> ⚠️ **중요**: 이 프로젝트는 두 가지 인코더를 지원합니다:
> - **Emformer Encoder** (원래 계획, 실험 단계)
> - **SimpleCTCEncoder** (현재 실제 사용 중, BiLSTM 기반)

---

## 📋 목차

1. [현재 실제 사용 중인 아키텍처](#현재-실제-사용-중인-아키텍처)
2. [Emformer Encoder (원래 계획)](#emformer-encoder-원래-계획)
3. [빠른 시작](#빠른-시작)
4. [프로젝트 구조](#프로젝트-구조)

---

## 현재 실제 사용 중인 아키텍처

### ✅ 현재 상태

**현재 학습 및 추론에 사용되는 모델**: `STCTCWithLM` (SimpleCTCEncoder 기반)

```
Speech Input [B, T, 80]
    ↓
SimpleCTCEncoder (BiLSTM, 4 layers)
    ↓
ST CTC Decoder (CTC head)
    ↓
MT Decoder (TransformerMTDecoder, 4 layers)
    ↓
Unit Decoder (CTCTransformerUnitDecoder, 별도 학습)
    ↓
CodeHiFiGAN Vocoder
    ↓
Speech Output
```

### 핵심 컴포넌트

| 컴포넌트 | 상태 | 설명 |
|---------|------|------|
| **SimpleCTCEncoder** | ✅ 사용 중 | BiLSTM 기반 인코더 (4 layers, 128-dim) |
| **ST CTC Decoder** | ✅ 사용 중 | CTC head for alignment (vocab: 5000) |
| **MT Decoder** | ✅ 사용 중 | TransformerMTDecoder (4 layers) |
| **Unit Decoder** | ✅ 사용 중 | CTCTransformerUnitDecoder (별도 학습) |
| **CodeHiFiGAN** | ✅ 사용 중 | Units → Audio 합성 |
| **CT-Transformer** | ❌ 미사용 | 구두점 예측 기능 (코드만 존재) |

### 현재 학습 스크립트

```bash
# ST 모델 학습 (SimpleCTCEncoder 사용)
python scripts/train_st_lm.py \
  --train-manifest data/train_clean_unk20_with_units.retokenized.tsv \
  --dev-manifest data/dev_sampled.retokenized.tsv \
  --vocab-path data/tgt_unigram5000/spm_unigram_en.model \
  --save-dir checkpoints_st_lm_vocab5000 \
  --epochs 10 \
  --batch-size 4 \
  --device cpu

# Unit Decoder 별도 학습
python scripts/train_unit_decoder.py \
  --train-manifest data/train_clean_unk20_with_units.retokenized.tsv \
  --dev-manifest data/dev_sampled.retokenized.tsv \
  --checkpoint-st checkpoints_st_lm_vocab5000/checkpoint_epoch_10.pt \
  --save-dir checkpoints_unit_decoder \
  --epochs 20 \
  --use-mt-decoder
```

### 주요 특징

- ✅ **UNK 토큰 문제 해결**: SentencePiece vocab 5000 + byte_fallback
- ✅ **Vocab Migration 완료**: 모든 manifest 재토큰화 (UNK 0%)
- ✅ **실용적인 파이프라인**: SimpleCTCEncoder로 안정적인 학습
- ✅ **체계적인 실험**: 56개 스크립트, 23개 문서

---

## Emformer Encoder (원래 계획)

### 📌 상태: 실험 단계 (현재 사용 안 함)

**원래 목표**: Conformer의 O(T²) 복잡도를 Emformer의 O(1)로 개선

```
Emformer Encoder (16 layers)
  - Left Context Cache (K, V 재사용)
  - Memory Bank (장거리 의존성)
  - Segment-wise processing
  - Complexity: O(1) per segment
```

### ⚠️ 현재 문제

- **Representation Collapse**: 모든 샘플이 동일한 벡터로 수렴
- **작은 데이터셋**: 2.4k 샘플에서 streaming 구조가 맞지 않음
- **학습 불안정**: Memory bank가 batch 학습과 충돌

### 📁 관련 파일

- `models/echostream_encoder.py`: Emformer 구현
- `models/emformer_layer.py`: Emformer 레이어
- `docs/STREAMSPEECH_VS_ECHOSTREAM.md`: Emformer vs Conformer 비교
- `docs/EMFORMER_INTEGRATION_PLAN.md`: 통합 계획

### 🔄 사용 방법 (실험용)

```python
# EchoStreamModel에서 Emformer 사용
model = EchoStreamModel(
    use_simple_encoder=False,  # Emformer 사용
    use_conformer=False,
    encoder_layers=16,
    segment_length=4,
    left_context_length=30,
    memory_size=8,
)
```

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/Kyle-Riss/EchoStream.git
cd EchoStream
```

### 2. 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 데이터 준비

필요한 파일:
- `data/train_clean_unk20_with_units.retokenized.tsv`
- `data/dev_sampled.retokenized.tsv`
- `data/tgt_unigram5000/spm_unigram_en.model`
- `pretrain_models/unit-based_HiFi-GAN_vocoder/` (vocoder)

### 4. 학습

```bash
# ST 모델 학습
python scripts/train_st_lm.py \
  --train-manifest data/train_clean_unk20_with_units.retokenized.tsv \
  --dev-manifest data/dev_sampled.retokenized.tsv \
  --vocab-path data/tgt_unigram5000/spm_unigram_en.model \
  --save-dir checkpoints_st_lm_vocab5000 \
  --epochs 10 \
  --batch-size 4 \
  --device cpu
```

### 5. 추론

```bash
python scripts/decode_with_wpb.py \
  --checkpoint checkpoints_st_lm_vocab5000/checkpoint_epoch_10.pt \
  --manifest data/dev_sampled.retokenized.tsv \
  --vocab-path data/tgt_unigram5000/spm_unigram_en.model \
  --num-samples 5 \
  --generate-audio
```

---

## 프로젝트 구조

```
EchoStream/
├── models/
│   ├── st_ctc_lm.py          # 현재 사용 중인 모델 (SimpleCTCEncoder)
│   ├── simple_encoder.py     # SimpleCTCEncoder (BiLSTM)
│   ├── echostream_model.py   # Emformer 기반 모델 (실험용)
│   ├── echostream_encoder.py # Emformer 인코더 (실험용)
│   └── decoders/
│       ├── transformer_decoder.py  # MT Decoder
│       ├── unit_decoder.py         # Unit Decoder
│       └── vocoder.py              # CodeHiFiGAN
├── scripts/
│   ├── train_st_lm.py        # ST 모델 학습 (현재 사용)
│   ├── train_unit_decoder.py # Unit Decoder 학습
│   └── decode_with_wpb.py    # 추론 스크립트
├── data/
│   ├── tgt_unigram5000/      # SentencePiece 모델 (vocab 5000)
│   └── *.retokenized.tsv     # 재토큰화된 manifest
└── docs/
    ├── STREAMSPEECH_VS_ECHOSTREAM.md  # Emformer 관련 문서
    └── VOCAB_MIGRATION_GUIDE.md       # Vocab migration 가이드
```

---

## 주요 성과

### ✅ 완료된 작업

1. **UNK 토큰 문제 해결**
   - SentencePiece vocab 4000 → 5000
   - byte_fallback 활성화
   - UNK 비율: 18.9% → 0%

2. **Vocab Migration**
   - 모든 manifest 재토큰화 완료
   - 모델 config 업데이트 (vocab_size 5000)

3. **안정적인 학습 파이프라인**
   - SimpleCTCEncoder 기반 학습
   - MT Decoder 통합
   - Unit Decoder 별도 학습

### 🔄 진행 중

- ST 모델 학습 (10 epochs)
- Unit Decoder 학습
- 디코딩 품질 개선

### 📋 계획

- Emformer 문제 해결 (representation collapse)
- 더 큰 데이터셋으로 실험
- 실시간 스트리밍 최적화

---

## 참고 문서

- [Vocab Migration Guide](VOCAB_MIGRATION_GUIDE.md): Vocab 변경 가이드
- [StreamSpeech vs EchoStream](docs/STREAMSPEECH_VS_ECHOSTREAM.md): Emformer 관련 문서
- [Architecture Summary](docs/ARCHITECTURE_SUMMARY.md): 전체 아키텍처 요약

---

## 라이선스

본 프로젝트는 원본 StreamSpeech 및 Fairseq의 라이선스를 따릅니다.

---

## 기여

기여를 환영합니다! 이슈와 풀 리퀘스트를 통해 참여해 주세요.

---

**EchoStream** - Speech-to-Speech Translation 실험 및 개발 프로젝트 🌊
