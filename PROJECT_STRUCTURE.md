# EchoStream 프로젝트 구조

StreamSpeech 구조를 참고하여 정리된 EchoStream 프로젝트 구조입니다.

## 📁 디렉토리 구조

```
EchoStream/
├── agent/                    # SimulEval 에이전트
│   ├── echostream_agent.py
│   └── echostream_simul_agent.py
│
├── configs/                  # 설정 파일
│   ├── echostream_config.yaml
│   ├── echostream_config.mini.yaml
│   └── echostream_config.mac_m2_8gb.yaml
│
├── datasets/                 # 데이터셋
│   ├── __init__.py
│   └── s2st_dataset.py
│
├── docs/                     # 문서
│   ├── README.md
│   ├── ARCHITECTURE_SUMMARY.md
│   ├── BENCHMARK_RESULTS.md
│   ├── COMPARISON_STREAMSPEECH_VS_ECHOSTREAM.md
│   ├── DETAILED_STRUCTURE_ANALYSIS.md
│   ├── ECHOSTREAM_ARCHITECTURE.md
│   ├── LATENCY_ANALYSIS_STREAMSPEECH_VS_ECHOSTREAM.md
│   ├── MAC_M2_8GB_TRAINING_GUIDE.md
│   ├── MODEL_SUMMARY.md
│   ├── STREAMSPEECH_COMPATIBILITY.md
│   └── TRAINING_RECOMMENDATION.md
│
├── models/                   # 모델 코드
│   ├── decoders/             # 디코더들
│   │   ├── ctc_decoder.py
│   │   ├── transformer_decoder.py
│   │   ├── unit_decoder.py
│   │   └── vocoder.py
│   ├── echostream_encoder.py # Emformer 인코더
│   ├── echostream_model.py   # 메인 모델
│   └── emformer_layer.py     # Emformer 레이어
│
├── scripts/                  # 스크립트
│   ├── train.py              # 학습
│   ├── evaluate.py           # 평가
│   ├── compute_gcmvn.py      # CMVN 계산
│   └── ...
│
├── server/                   # 서버 (FastAPI)
│   ├── fastapi_app.py
│   └── client_ws.py
│
├── tests/                    # 테스트
│   ├── test_echostream.py
│   └── test_zipformer_integration.py
│
├── training/                 # 학습 관련
│   └── echostream_criterion.py
│
├── data/                     # 데이터
│   ├── train_sampled.units.tsv
│   ├── dev_sampled.units.tsv
│   ├── units/                # Units 파일들
│   └── ...
│
├── checkpoints/              # 체크포인트
│   └── ...
│
├── pretrain_models/          # 사전 학습 모델
│   ├── mHuBERT/
│   └── unit-based_HiFi-GAN_vocoder/
│
├── results/                  # 결과 파일
│   └── ...
│
└── StreamSpeech_analysis/    # StreamSpeech 분석 (참고용)
    └── ...
```

## 🔑 주요 디렉토리 설명

### `agent/`
SimulEval을 위한 에이전트 코드. StreamSpeech의 `agent/` 구조를 따릅니다.

### `configs/`
모델 설정 파일들. StreamSpeech의 `configs/` 구조를 따릅니다.

### `models/`
- `echostream_encoder.py`: Emformer 기반 인코더 (StreamSpeech의 Conformer 대체)
- `echostream_model.py`: 메인 모델 (StreamSpeech의 `streamspeech_model.py`와 유사)
- `decoders/`: 모든 디코더 (StreamSpeech와 동일)

### `scripts/`
학습, 평가, 전처리 스크립트들.

### `docs/`
모든 문서 파일을 한 곳에 모았습니다.

## 📝 StreamSpeech와의 차이점

1. **인코더**: Conformer → Emformer
2. **구조**: 나머지는 StreamSpeech와 동일
3. **문서**: `docs/` 디렉토리로 통합

---

**마지막 업데이트**: 2025-01-XX


