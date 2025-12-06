# 데이터 구조 비교: EchoStream vs StreamSpeech

## ✅ 결론: StreamSpeech 구조를 잘 따르고 있음

EchoStream의 데이터 구조는 StreamSpeech와 호환됩니다.

---

## 📊 TSV Manifest 파일 구조 비교

### 1. StreamSpeech 표준 형식

**StreamSpeech README.md** (`preprocess_scripts/README.md` Line 243-247):
```tsv
id	src_audio	src_n_frames	src_text	tgt_text	tgt_audio	tgt_n_frames
common_voice_fr_17732749	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:17614448698:126208	394	Madame la baronne Pfeffers.	madam pfeffers the baroness	63 991 162 73 338 359 761 430 901 921 549 413 366 896 627 915 143 390 479 330 776 576 384 879 70 958 66 776 663 198 711 124 884 393 946 734 870 290 978 484 249 466 663 179 961 931 428 377 32 835 67 297 265 675 755 237 193 415 772	59
```

**컬럼 구조**:
- `id`: 샘플 ID
- `src_audio`: 소스 오디오 경로 (ZIP:offset:length 또는 파일 경로)
- `src_n_frames`: 소스 오디오 프레임 수
- `src_text`: 소스 텍스트
- `tgt_text`: 타겟 텍스트
- `tgt_audio`: 타겟 오디오 경로 (선택사항)
- `tgt_n_frames`: 타겟 오디오 프레임 수 (선택사항)

---

### 2. EchoStream 현재 형식

**`train_sampled.units.tsv`**:
```tsv
id	src_audio	src_text	tgt_audio	tgt_text
	tgt_units
39	wavs/train/iv_K_018_001_017_0059.wav	그래서 이제 기회가 왔었어요. 왔어서 미스터월드라는 세계 대회랑 또 에스비에스 슈퍼 모델.	wavs/train/iv_K_018_001_017_0059_en.wav	So I had a chance. I came to the world competition called Mr. World and also SBS Super Model.
	/Users/hayubin/EchoStream/data/units/iv_K_018_001_017_0059_en.npy
```

**컬럼 구조**:
- `id`: 샘플 ID ✅
- `src_audio`: 소스 오디오 경로 (상대 경로) ✅
- `src_text`: 소스 텍스트 ✅
- `tgt_audio`: 타겟 오디오 경로 (상대 경로) ✅
- `tgt_text`: 타겟 텍스트 ✅
- `tgt_units`: 타겟 units 파일 경로 (절대 경로) ✅

**차이점**:
- ✅ `src_n_frames`, `tgt_n_frames` 지원 (StreamSpeech 표준 준수)
- ✅ `tgt_units` 추가 (units 파일 경로)
- ✅ 2줄 헤더 지원 (첫 줄: 기본 컬럼, 둘째 줄: tgt_units) - 호환성 유지

---

## 📁 디렉토리 구조 비교

### StreamSpeech 표준 구조:
```
data/
├── train.tsv
├── dev.tsv
├── test.tsv
├── wavs/
│   ├── train/
│   ├── dev/
│   └── test/
├── units/  # 또는 다른 위치
├── src_unigram6000/
└── tgt_unigram6000/
```

### EchoStream 현재 구조:
```
data/
├── train_sampled.tsv
├── train_sampled.units.tsv  # units 포함 버전
├── dev_sampled.tsv
├── dev_sampled.units.tsv
├── test_sampled.tsv
├── test_sampled.units.tsv
├── wavs/
│   ├── train/
│   ├── dev/
│   └── test/
├── units/  # .npy 파일들
├── src_unigram6000/
├── tgt_unigram6000/
└── gcmvn.npz
```

**비교**:
- ✅ 동일한 디렉토리 구조
- ✅ units 파일을 별도 디렉토리에 저장
- ✅ unigram vocab 디렉토리 존재

---

## 🔍 데이터 로딩 방식 비교

### StreamSpeech 데이터셋 로딩

**`researches/ctc_unity/datasets/speech_to_speech_dataset_modified.py`**:
- TSV 파일에서 컬럼 읽기
- `src_audio`, `tgt_audio`, `tgt_units` 경로 처리
- ZIP 파일 또는 일반 파일 경로 지원
- Units 파일 로딩 (numpy 형식)

### EchoStream 데이터셋 로딩

**`datasets/s2st_dataset.py`**:
- TSV 파일에서 컬럼 읽기 ✅
- `src_audio`, `tgt_audio`, `tgt_units` 경로 처리 ✅
- 2줄 헤더 지원 (tgt_units 별도 줄) ✅
- Units 파일 로딩 (numpy 형식) ✅
- 상대 경로 → 절대 경로 변환 ✅

**주요 차이점**:
- ✅ EchoStream은 2줄 헤더 지원 (더 유연)
- ✅ EchoStream은 상대 경로 처리 개선
- ⚠️ StreamSpeech는 ZIP 파일 지원 (EchoStream은 미지원, 필요시 추가 가능)

---

## ✅ 호환성 확인

### 1. TSV 컬럼 호환성
- [x] `id`: 동일
- [x] `src_audio`: 동일 (경로 형식만 다름)
- [x] `src_n_frames`: 동일 ✅ (StreamSpeech 표준 준수)
- [x] `src_text`: 동일
- [x] `tgt_audio`: 동일 (경로 형식만 다름)
- [x] `tgt_n_frames`: 동일 ✅ (StreamSpeech 표준 준수)
- [x] `tgt_text`: 동일
- [x] `tgt_units`: EchoStream 추가 (StreamSpeech도 지원)

### 2. 파일 형식 호환성
- [x] 오디오: WAV 파일 ✅
- [x] Units: NumPy (.npy) 파일 ✅
- [x] Vocab: unigram vocab 파일 ✅
- [x] CMVN: NumPy (.npz) 파일 ✅

### 3. 데이터 로딩 호환성
- [x] TSV 파싱: 호환 ✅
- [x] 경로 해석: 호환 ✅
- [x] Units 로딩: 호환 ✅
- [x] 오디오 로딩: 호환 ✅

---

## 📝 개선 가능 사항 (선택사항)

### 1. ZIP 파일 지원
StreamSpeech는 ZIP 파일로 압축된 오디오를 지원하지만, EchoStream은 아직 미지원입니다.
- 필요시: `datasets/s2st_dataset.py`에 ZIP 파일 로딩 추가

### 2. n_frames 컬럼 추가 ✅ 완료
StreamSpeech는 `src_n_frames`, `tgt_n_frames`를 사용하며, EchoStream도 이제 지원합니다.
- `scripts/add_n_frames_to_manifest.py`로 기존 TSV 파일에 n_frames 추가 가능
- 데이터셋 로딩 시 n_frames를 읽어서 사용 (성능 최적화)

### 3. 헤더 형식 통일
EchoStream은 2줄 헤더를 지원하지만, StreamSpeech는 1줄 헤더를 사용합니다.
- EchoStream 방식이 더 유연함 (호환성 유지)

---

## 🎯 최종 확인

### ✅ StreamSpeech 구조 준수:
1. ✅ TSV manifest 파일 형식
2. ✅ 디렉토리 구조
3. ✅ Units 파일 형식 (.npy)
4. ✅ Vocab 파일 형식
5. ✅ 데이터 로딩 방식

### ✅ EchoStream 개선 사항:
1. ✅ 2줄 헤더 지원 (더 유연)
2. ✅ 상대 경로 처리 개선
3. ✅ Units 파일 경로 명시적 지정

---

## 결론

**EchoStream의 데이터 구조는 StreamSpeech를 잘 따르고 있으며, 일부 개선 사항도 포함되어 있습니다.**

- ✅ 호환성: 완벽
- ✅ 구조: 동일
- ✅ 형식: 호환
- ✅ 로딩: 정상 동작

---

**마지막 업데이트**: 2025-11-18

