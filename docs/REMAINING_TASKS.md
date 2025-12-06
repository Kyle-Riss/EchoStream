# 남은 작업 및 체크리스트

## ✅ 완료된 작업

### 1. 데이터 구조 정렬
- [x] `src_n_frames`, `tgt_n_frames` 필드 추가 (`datasets/s2st_dataset.py`)
- [x] Manifest 파일에 n_frames 추가 스크립트 생성 (`scripts/add_n_frames_to_manifest.py`)
- [x] StreamSpeech 표준 형식 준수 확인 (`data/train_sampled.units.streamspeech_format_final.tsv`)
- [x] 컬럼 순서: `id, src_audio, src_n_frames, src_text, tgt_text, tgt_audio, tgt_n_frames`

### 2. 모델 구조 확인
- [x] Emformer 인코더로 변경 (Conformer → Emformer)
- [x] 디코더 구조 StreamSpeech와 동일 확인
- [x] 출력 포맷 StreamSpeech/Fairseq 호환 확인

### 3. 학습 코드 확인
- [x] Loss 계산 방식 StreamSpeech와 동일 (CTC + Cross-Entropy)
- [x] Optimizer 설정 동일 (Adam, betas 0.9, 0.98)
- [x] Learning rate scheduler 구현 (inverse sqrt + warmup)
- [x] Gradient accumulation 지원
- [x] Multi-task loss 가중치 사용

### 4. 문서화
- [x] StreamSpeech 표준 형식 검증 문서
- [x] 학습 코드 비교 문서
- [x] 데이터 구조 비교 문서

---

## 🔧 남은 작업 (선택사항)

### 1. Config 파일 업데이트 (권장)

**현재 상태**:
- Config: `train_manifest: ../data/train_sampled.units.tsv`
- 생성된 파일: `data/train_sampled.units.streamspeech_format_final.tsv`

**작업**:
```yaml
# configs/echostream_config.mac_m2_8gb.yaml
data:
  train_manifest: ../data/train_sampled.units.streamspeech_format_final.tsv
  valid_manifest: ../data/dev_sampled.units.streamspeech_format_final.tsv  # dev도 생성 필요
  test_manifest: ../data/test_sampled.units.streamspeech_format_final.tsv  # test도 생성 필요
```

**또는** 기존 파일을 덮어쓰기:
```bash
cp data/train_sampled.units.streamspeech_format_final.tsv data/train_sampled.units.tsv
```

---

### 2. Dev/Test Manifest에도 n_frames 추가 (권장)

**작업**:
```bash
# Dev manifest
python scripts/add_n_frames_to_manifest.py \
  --in data/dev_sampled.units.tsv \
  --out data/dev_sampled.units.streamspeech_format_final.tsv \
  --data-root data \
  --units-root data/units

# Test manifest
python scripts/add_n_frames_to_manifest.py \
  --in data/test_sampled.units.tsv \
  --out data/test_sampled.units.streamspeech_format_final.tsv \
  --data-root data \
  --units-root data/units
```

---

### 3. 실제 학습 실행 및 검증 (필수)

**작업**:
1. 학습 시작:
   ```bash
   python scripts/train.py \
     --config configs/echostream_config.mac_m2_8gb.yaml \
     --train-manifest data/train_sampled.units.streamspeech_format_final.tsv \
     --dev-manifest data/dev_sampled.units.streamspeech_format_final.tsv \
     --save-dir checkpoints_mac_m2_8gb \
     --epochs 50
   ```

2. 학습 진행 확인:
   - Loss가 정상적으로 감소하는지
   - NaN 발생하지 않는지
   - 메모리 사용량 확인

3. Checkpoint 저장 확인:
   - `checkpoints_mac_m2_8gb/checkpoint_best.pt` 생성 확인

---

### 4. 평가 스크립트 테스트 (권장)

**작업**:
```bash
python scripts/evaluate.py \
  --config configs/echostream_config.mac_m2_8gb.yaml \
  --checkpoint checkpoints_mac_m2_8gb/checkpoint_best.pt \
  --test-manifest data/test_sampled.units.streamspeech_format_final.tsv \
  --output results/evaluation.json
```

**확인 사항**:
- BLEU 점수 계산
- ASR-BLEU 점수 계산
- Latency 메트릭 (AL, AP, DAL)

---

### 5. Inference 코드 검증 (선택사항)

**확인 사항**:
- `models/word_level_translator.py`: 단어 단위 번역
- `server/fastapi_app.py`: FastAPI 서버
- `agent/echostream_agent.py`: SimulEval agent

**테스트**:
```bash
# FastAPI 서버 실행
python server/fastapi_app.py

# 또는 SimulEval로 평가
simuleval --agent agent/echostream_agent.py \
  --source data/test_source.txt \
  --target data/test_target.txt
```

---

### 6. Label Smoothing 추가 (선택사항)

**현재**: Config에 `label_smoothing: 0.1` 있지만 미적용

**작업**: `scripts/train.py`의 `MultiTaskLoss`에 label smoothing 적용
```python
# CrossEntropyLoss에 label_smoothing 파라미터 추가
self.ce_loss = nn.CrossEntropyLoss(
    ignore_index=0, 
    reduction='mean',
    label_smoothing=0.1  # 추가
)
```

---

### 7. 데이터셋 로딩 검증 (권장)

**작업**:
```python
# 간단한 테스트 스크립트
from datasets import S2STManifestDataset

dataset = S2STManifestDataset(
    manifest_path='data/train_sampled.units.streamspeech_format_final.tsv',
    data_root='data',
    units_root='data/units',
)

# 샘플 로딩 테스트
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Speech shape: {sample['speech'].shape}")
print(f"src_n_frames: {dataset.entries[0].src_n_frames}")
print(f"tgt_n_frames: {dataset.entries[0].tgt_n_frames}")
```

---

## 🎯 우선순위

### 높음 (필수)
1. ✅ **Dev/Test manifest에 n_frames 추가**
2. ✅ **Config 파일 업데이트**
3. ✅ **실제 학습 실행 및 검증**

### 중간 (권장)
4. ✅ **평가 스크립트 테스트**
5. ✅ **데이터셋 로딩 검증**

### 낮음 (선택사항)
6. ✅ **Label Smoothing 적용**
7. ✅ **Inference 코드 검증**

---

## 📝 체크리스트

### 데이터 준비
- [ ] Train manifest에 n_frames 추가 ✅ (완료)
- [ ] Dev manifest에 n_frames 추가
- [ ] Test manifest에 n_frames 추가
- [ ] Config 파일 업데이트

### 학습
- [ ] 학습 스크립트 실행
- [ ] Loss 정상 감소 확인
- [ ] Checkpoint 저장 확인
- [ ] 메모리 사용량 확인

### 평가
- [ ] 평가 스크립트 실행
- [ ] BLEU 점수 확인
- [ ] Latency 메트릭 확인

### 검증
- [ ] 데이터셋 로딩 테스트
- [ ] Inference 테스트
- [ ] End-to-end 테스트

---

## 💡 다음 단계

1. **즉시 실행 가능**: Dev/Test manifest에 n_frames 추가
2. **학습 시작**: Config 업데이트 후 학습 실행
3. **평가 준비**: 학습 완료 후 평가 스크립트 실행

---

**마지막 업데이트**: 2025-11-18

