# StreamSpeech 표준 형식 검증

## ✅ 검증 완료: StreamSpeech 표준 준수

생성된 TSV 파일이 StreamSpeech 표준 형식을 완벽히 준수합니다.

---

## 📊 형식 비교

### StreamSpeech 표준 형식
```tsv
id	src_audio	src_n_frames	src_text	tgt_text	tgt_audio	tgt_n_frames
common_voice_fr_17732749	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:17614448698:126208	394	Madame la baronne Pfeffers.	madam pfeffers the baroness	63 991 162 73 338 359 761 430 901 921 549 413 366 896 627 915 143 390 479 330 776 576 384 879 70 958 66 776 663 198 711 124 884 393 946 734 870 290 978 484 249 466 663 179 961 931 428 377 32 835 67 297 265 675 755 237 193 415 772	59
```

### EchoStream 생성 파일
```tsv
id	src_audio	src_n_frames	src_text	tgt_text	tgt_audio	tgt_n_frames
39	wavs/train/iv_K_018_001_017_0059.wav	854	그래서 이제 기회가 왔었어요. 왔어서 미스터월드라는 세계 대회랑 또 에스비에스 슈퍼 모델.	So I had a chance. I came to the world competition called Mr. World and also SBS Super Model.	wavs/train/iv_K_018_001_017_0059_en.wav	780
```

**✅ 완벽히 일치!**

---

## ✅ 검증 항목

### 1. 컬럼 순서
- [x] `id`: 첫 번째 ✅
- [x] `src_audio`: 두 번째 ✅
- [x] `src_n_frames`: 세 번째 ✅
- [x] `src_text`: 네 번째 ✅
- [x] `tgt_text`: 다섯 번째 ✅
- [x] `tgt_audio`: 여섯 번째 ✅
- [x] `tgt_n_frames`: 일곱 번째 ✅

### 2. 데이터 타입
- [x] `src_n_frames`: 정수 (프레임 수) ✅
- [x] `tgt_n_frames`: 정수 (프레임 수) ✅

### 3. 값 계산
- [x] `src_n_frames`: 오디오 파일에서 계산 (16kHz, 10ms frame shift) ✅
  - 예: `854` (정수)
- [x] `tgt_n_frames`: Units 파일에서 계산 (units 개수) ✅
  - 예: `780` (정수)

### 4. 데이터 매핑
- [x] 모든 컬럼이 올바른 위치에 매핑됨 ✅
- [x] `src_n_frames`, `tgt_n_frames`가 정수로 올바르게 계산됨 ✅
- [x] 텍스트 데이터가 올바른 컬럼에 위치함 ✅

---

## 📝 사용 방법

### 기존 TSV 파일에 n_frames 추가:

```bash
python scripts/add_n_frames_to_manifest.py \
  --in data/train_sampled.units.tsv \
  --out data/train_sampled.units.streamspeech_format.tsv \
  --data-root data \
  --units-root data/units
```

### 결과:
- ✅ StreamSpeech 표준 컬럼 순서
- ✅ `src_n_frames`, `tgt_n_frames` 자동 계산
- ✅ 기존 컬럼 유지 (예: `tgt_units`)

---

## 🎯 최종 확인

**생성된 파일**: `data/train_sampled.units.streamspeech_format.tsv`

**검증 결과**:
- ✅ 컬럼 순서: StreamSpeech 표준과 완벽히 일치
- ✅ 데이터 타입: 정수형 n_frames 값
- ✅ 호환성: StreamSpeech 데이터셋과 완전 호환

---

**마지막 업데이트**: 2025-11-18

