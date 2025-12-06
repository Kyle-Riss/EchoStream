# Conformer vs Emformer: 핵심 차이점 분석

## 📋 개요

StreamSpeech는 **Chunk-based Conformer** 인코더를 사용하고, EchoStream은 **Emformer** 인코더를 사용합니다. 이 문서는 두 인코더의 핵심 차이점을 명확히 정리합니다.

---

## 🔍 1. 아키텍처 구조 비교

### StreamSpeech (Chunk-based Conformer)

```
Input [B, T, 80]
    ↓
Conv2D Subsampling (4x) → [T/4, B, 256]
    ↓
Positional Embedding (abs/rel_pos/rope)
    ↓
Linear Projection + Dropout
    ↓
Conformer Layers (16L):
    ├─ FFN1 (0.5 residual)
    ├─ Self-Attention (chunk mask)
    ├─ Depthwise Convolution Module ⭐
    ├─ FFN2 (0.5 residual)
    └─ Layer Norm
    ↓
Output [T/4, B, 256]
```

### EchoStream (Emformer)

```
Input [B, T, 80]
    ↓
Conv2D Subsampling (4x) → [T/4, B, 256]
    ↓
❌ Positional Encoding 없음!
    ↓
Emformer Layers (16L):
    ├─ Self-Attention (left context cache + memory bank)
    └─ FFN
    ↓
Output [T/4, B, 256]
```

---

## ⚠️ 2. 주요 차이점

### 2.1 Positional Encoding

| 항목 | StreamSpeech (Conformer) | EchoStream (Emformer) |
|------|---------------------------|----------------------|
| **Positional Encoding** | ✅ **있음** | ❌ **없음** |
| **타입** | `PositionalEmbedding` (abs) 또는 `RelPositionalEncoding` (rel_pos) | - |
| **적용 위치** | Subsampling 후, Linear projection 전 | - |
| **코드** | ```python<br>x += positions<br>``` | - |

**영향**: Positional encoding이 없으면 시퀀스 내 위치 정보를 모델이 학습하기 어려울 수 있습니다.

---

### 2.2 Convolution Module

| 항목 | StreamSpeech (Conformer) | EchoStream (Emformer) |
|------|---------------------------|----------------------|
| **Depthwise Conv** | ✅ **있음** | ❌ **없음** |
| **Kernel Size** | 31 (default) | - |
| **위치** | Self-Attention 후, FFN2 전 | - |
| **역할** | 로컬 패턴 캡처 (시간적 의존성) | - |

**영향**: Conformer의 핵심 구성요소인 depthwise convolution이 없어 로컬 시간적 패턴 학습이 약할 수 있습니다.

---

### 2.3 Layer 구조

#### StreamSpeech Conformer Layer (Macaron-style)

```python
# 1. FFN1 (0.5 residual)
residual = x
x = self.ffn1(x)
x = x * 0.5 + residual  # ⭐ 0.5 scaling

# 2. Self-Attention
residual = x
x = self.self_attn_layer_norm(x)
x = self.self_attn(x, ...)
x = residual + x

# 3. Depthwise Convolution ⭐
residual = x
x = x.transpose(0, 1)  # TBC → BTC
x = self.conv_module(x)
x = x.transpose(0, 1)  # BTC → TBC
x = residual + x

# 4. FFN2 (0.5 residual)
residual = x
x = self.ffn2(x)
x = x * 0.5 + residual  # ⭐ 0.5 scaling

# 5. Final Layer Norm
x = self.final_layer_norm(x)
```

#### EchoStream Emformer Layer (Transformer-style)

```python
# 1. Self-Attention (with left context cache + memory bank)
residual = query
query = self.self_attn_layer_norm(query)
attn_output = self.self_attn(query, key, value)
query = residual + self.dropout(attn_output)

# 2. FFN
residual = query
query = self.final_layer_norm(query)
query = self.fc1(query)
query = self.activation_fn(query)
query = self.fc2(query)
output = residual + self.dropout(query)
```

**차이점**:
- Conformer: **Macaron-style** (FFN1 → Attn → Conv → FFN2, 0.5 residual scaling)
- Emformer: **Transformer-style** (Attn → FFN, 1.0 residual scaling)
- Conformer: **Depthwise convolution** 포함
- Emformer: **Left context cache + Memory bank** 포함

---

### 2.4 Attention Mechanism

#### StreamSpeech (Chunk-based Attention)

```python
# Chunk mask 생성
chunk_size = self.chunk_size  # e.g., 8
idx = torch.arange(0, dim) // chunk_size + 1
chunk_mask = torch.where(
    idx <= tmp, 
    float("-inf"),  # Mask future chunks
    0.0
)
```

- **Chunk 단위로 attention 제한**
- 각 chunk는 이전 chunk들만 볼 수 있음
- O(T²) 복잡도 (chunk 내부는 full attention)

#### EchoStream (Left Context Cache + Memory Bank)

```python
# Left context K, V (cached from previous segments)
if left_context_key is not None:
    keys.append(left_context_key)  # 재사용!

# Memory bank (from lower layer)
if memory_bank is not None:
    keys.append(memory_bank)

# Attention: Q=[C, R] × K=[M, L, C, R]
attn_output = self.self_attn(query, key, value)
```

- **Left context K, V 캐싱** (재계산 없음)
- **Memory bank** (하위 레이어에서 상위 레이어로)
- O(1) 복잡도 (segment 길이와 무관)

---

### 2.5 Linear Projection

#### StreamSpeech

```python
# Subsampling 후
x = self.embed_scale * x  # ⭐ Scaling
if pos_enc_type == "abs":
    x += positions  # ⭐ Positional encoding
x = self.linear(x)  # ⭐ Linear projection
x = self.dropout(x)
```

#### EchoStream

```python
# Subsampling 후
# ❌ Positional encoding 없음
# ❌ Linear projection 없음
# 바로 Emformer로 전달
```

---

## 🎯 3. 성능에 미치는 영향

### 3.1 Positional Encoding 부재

**문제점**:
- 시퀀스 내 위치 정보 부족
- 모델이 상대적/절대적 위치를 학습하기 어려움

**해결 방안**:
- Emformer에 positional encoding 추가 고려
- 또는 Emformer의 segment 구조가 위치 정보를 암묵적으로 제공하는지 확인

### 3.2 Depthwise Convolution 부재

**문제점**:
- 로컬 시간적 패턴 학습 능력 저하
- Conformer의 핵심 강점 중 하나 손실

**해결 방안**:
- Emformer에 convolution module 추가 고려
- 또는 Emformer의 attention이 충분히 로컬 패턴을 캡처하는지 확인

### 3.3 Layer 구조 차이

**문제점**:
- Macaron-style (0.5 residual) vs Transformer-style (1.0 residual)
- 학습 동역학이 다를 수 있음

**해결 방안**:
- 학습률/스케줄러 조정 필요할 수 있음

---

## 📊 4. 출력 형식 비교

### 둘 다 StreamSpeech/Fairseq 형식 사용 ✅

```python
{
    'encoder_out': [x],  # List of [T, B, D]
    'encoder_padding_mask': [mask] or [],  # List of [B, T]
    'encoder_embedding': [],
    'encoder_states': [],
    'src_tokens': [],
    'src_lengths': [],
}
```

**결론**: 출력 형식은 동일하므로 디코더 호환성 문제는 없습니다.

---

## 🔧 5. 권장 수정 사항

### 우선순위 1: Positional Encoding 추가

```python
# EchoStreamSpeechEncoder.__init__()
self.embed_positions = PositionalEmbedding(
    max_positions=6000,
    embedding_dim=encoder_embed_dim,
    padding_idx=1,
)

# EchoStreamSpeechEncoder.forward()
x, input_lengths = self.subsample(src_tokens, src_lengths)
x = self.embed_scale * x  # Add scaling
x += self.embed_positions(x)  # Add positional encoding
x = self.linear(x)  # Add linear projection
x = self.dropout(x)
```

### 우선순위 2: Depthwise Convolution 고려

- Emformer의 효율성을 유지하면서 convolution 추가는 복잡할 수 있음
- 먼저 positional encoding 추가 후 성능 확인

### 우선순위 3: Layer 구조 조정

- Macaron-style로 변경은 큰 수정이 필요
- 우선 positional encoding 추가 후 평가

---

## 📝 6. 체크리스트

- [ ] Positional encoding 추가
- [ ] Linear projection 추가 (Conformer와 동일하게)
- [ ] Embed scale 추가
- [ ] Depthwise convolution 추가 검토
- [ ] 학습 후 성능 비교

---

## 💡 결론

**가장 큰 차이점**:
1. ❌ **Positional Encoding 부재** (가장 중요!)
2. ❌ **Depthwise Convolution 부재**
3. ❌ **Linear Projection 부재**
4. ⚠️ **Layer 구조 차이** (Macaron vs Transformer)

**즉시 수정 권장**: Positional Encoding 추가

---

**마지막 업데이트**: 2025-01-XX




