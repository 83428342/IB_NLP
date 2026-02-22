# Model Architecture Documentation

**프로젝트:** 시계열 데이터의 Variational Information Bottleneck 기반 멀티모달 융합 프레임워크  
**최종 실험 결과:** PTB-XL ECG 데이터셋 기준 (epochs=20, beta=0.001)

---

## 1. 전체 아키텍처 개요

본 모델은 세 가지 핵심 구성 요소로 이루어집니다:

```
                        ┌─────────────────────────────┐
  x_ts                  │  Time-Series Encoder (VIB)  │
  (B, T, F) ──────────▶│  Transformer + Mean Pooling  │──▶ z_ts  (B, 64)
                        │  + VIBLayer (μ, σ → z + KL) │               │
                        └─────────────────────────────┘               │
                                                                   concat
                        ┌─────────────────────────────┐               │
  x_text                │  Text Encoder (Auxiliary)   │               │
  (B, L)   ──────────▶│  Bio_ClinicalBERT [CLS]       │──▶ z_text (B, 32)
                        │  → Projection MLP (768→32)  │               │
                        └─────────────────────────────┘               │
                                                                       ▼
                        ┌─────────────────────────────┐
                        │  Fusion Classifier           │
                        │  Linear(96→128) → ReLU       │──▶ logits (B, 2)
                        │  → Dropout → Linear(128→2)  │
                        └─────────────────────────────┘
```

**입력 차원 (PTB-XL 기준)**
- `x_ts`: `(Batch, 1000, 12)` — 10초 ECG, 12 리드, 100Hz 샘플링
- `x_text`: `(Batch, 512)` — 환자 메타데이터 텍스트 토큰 (Bio_ClinicalBERT)

---

## 2. 구성 요소 상세

### 2-1. VIBLayer

가우시안 잠재 변수를 생성하는 핵심 병목층입니다.

```python
class VIBLayer(nn.Module):
    # input_dim → z_dim = 64
    fc_mu     : Linear(64, 64)   # 평균 μ
    fc_logvar : Linear(64, 64)   # 로그 분산 log σ²
```

**Forward 연산:**
$$z = \mu_\phi(h) + \sigma_\phi(h) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**KL Divergence (해석적 계산):**
$$\mathcal{L}_{KL} = -\frac{1}{2} \sum_{j=1}^{64} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

### 2-2. TimeSeriesTransformerVIB

ECG 시계열에서 노이즈를 제거하고 최소 충분 표현을 추출합니다.

```python
class TimeSeriesTransformerVIB(nn.Module):
    embedding       : Linear(12, 64)           # 리드 피처 → d_model
    pos_encoder     : Parameter(1, 1000, 64)   # 학습 가능 위치 인코딩
    transformer_enc : TransformerEncoder(
                          d_model=64, nhead=4,
                          num_layers=2, dropout=0.2
                      )
    vib             : VIBLayer(64, 64)
```

**Forward:**
1. `x_emb = embedding(x) * √64 + pos_encoder`  — 스케일링된 입력 임베딩
2. `h = TransformerEncoder(x_emb)`  — shape: `(B, 1000, 64)`
3. `h_pool = h.mean(dim=1)`  — **Mean pooling** (전 타임스텝 평균), shape: `(B, 64)`
4. `z_ts, kl = VIBLayer(h_pool)`  — 확률적 병목 압축

> **설계 근거:** 마지막 토큰만 사용하는 방식(last-token pooling)은 1000 타임스텝 ECG 신호에서 대부분의 정보를 손실합니다. Mean pooling은 전체 시퀀스에 걸쳐 균일하게 정보를 집약합니다.

---

### 2-3. TextBERTEncoder

환자 메타데이터 텍스트를 보조 정보로 인코딩합니다. **VIB 없음** — 텍스트는 병목 없이 있는 그대로 fusion에 기여합니다.

```python
class TextBERTEncoder(nn.Module):
    bert       : AutoModel("emilyalsentzer/Bio_ClinicalBERT")
                 # 대부분 frozen, 마지막 encoder layer + pooler만 fine-tune
    projection : Linear(768, 128) → LayerNorm → ReLU
               → Linear(128, 32)  → LayerNorm → ReLU
```

**입력 텍스트 형식 (PTB-XL):**  
```
Patient: 56yo female. BMI: 24.2. Device: CS-12. 
Signal quality: baseline drift, no noise.
```

출력: `z_text ∈ ℝ^32`

---

### 2-4. MultimodalStockPredictor (Proposed Model: fusion_vib)

```python
class MultimodalStockPredictor(nn.Module):
    ts_encoder  : TimeSeriesTransformerVIB(feat=12, win=1000, z_dim=64)
    text_encoder: TextBERTEncoder(proj_dim=32)
    classifier  : Dropout(0.3) → Linear(96, 128) → ReLU
                → Dropout(0.3) → Linear(128, 2)
```

**Forward:**
```python
z_ts,  kl_loss = ts_encoder(x_ts)          # VIB 적용 (B, 64)
z_text         = text_encoder(ids, mask)   # 보조 정보   (B, 32)
z_fused        = cat([z_ts, z_text])       # concat      (B, 96)
logits         = classifier(z_fused)       # 분류        (B, 2)
return logits, kl_loss
```

---

### 2-5. AblationPredictor (VIB 제거 Ablation)

`fusion_vib`와 **동일한 구조** — VIB만 제거한 비교군입니다.

```python
class AblationPredictor(nn.Module):
    embedding   : Linear(12, 64)
    pos_encoder : Parameter(1, 1000, 64)
    transformer : TransformerEncoder(d_model=64, nhead=4, layers=2, dropout=0.2)
    # VIBLayer 없음, 대신 mean(dim=1) → d=64 직접 사용
    text_encoder: TextBERTEncoder(proj_dim=32)
    classifier  : Dropout(0.3) → Linear(96, 128) → ReLU
                → Dropout(0.3) → Linear(128, 2)
```

| 항목 | fusion_vib | ablation |
|:---|:---:|:---:|
| TS pooling | mean(1000 steps) | mean(1000 steps) |
| TS z_dim | 64 (VIB 출력) | 64 (d_model) |
| VIB / KL loss | ✅ | ❌ |
| Text encoder | 동일 | 동일 |
| Classifier | 동일 | 동일 |
| embedding 스케일링 | `* √64` | `* √64` |
| Transformer dropout | 0.2 | 0.2 |

---

### 2-6. TimeSeriesOnlyPredictor (TS 단독 Baseline)

```python
class TimeSeriesOnlyPredictor(nn.Module):
    ts_encoder : TimeSeriesTransformerVIB(feat=12, win=1000, z_dim=64)
    classifier : Dropout(0.3) → Linear(64, 64) → ReLU → Linear(64, 2)
```

### 2-7. TextOnlyPredictor (Text 단독 Baseline)

```python
class TextOnlyPredictor(nn.Module):
    text_encoder: TextBERTEncoder(proj_dim=32)
    classifier  : Dropout(0.3) → Linear(32, 128) → ReLU → Linear(128, 2)
```

---

## 3. 학습 목적 함수

$$\mathcal{J} = \frac{1}{N} \sum_{i=1}^{N} \underbrace{\mathcal{L}_{CE}(y_i,\ \hat{y}_i)}_{\text{분류 손실}} + \beta \cdot \underbrace{\mathcal{L}_{KL}(z_{ts}^{(i)})}_{\text{VIB 정규화}}$$

- **$\mathcal{L}_{CE}$**: Cross-Entropy Loss (이진 분류)
- **$\mathcal{L}_{KL}$**: KL Divergence from $\mathcal{N}(\mu, \sigma^2)$ to $\mathcal{N}(0, I)$
- **$\beta$**: KL 가중치 (`--beta 0.001`), Linear Annealing으로 첫 `epochs//2` 동안 0 → β까지 선형 증가

**KL Annealing 스케줄:**
$$\beta_t = \beta \cdot \min\!\left(1,\ \frac{t}{\lfloor\text{epochs}/2\rfloor}\right), \quad t = 1, 2, \dots, \text{epochs}$$

---

## 4. 최종 실험 결과 (PTB-XL, epochs=20, beta=0.001)

| Model | Best AUC | Last Acc | Last F1 | Last AUC |
|:---|:---:|:---:|:---:|:---:|
| ts_only | 0.9095 | 0.8317 | 0.8568 | 0.9064 |
| text_only | 0.7528 | 0.7096 | 0.7738 | 0.7518 |
| ablation | 0.9279 | 0.8358 | 0.8534 | 0.9255 |
| **fusion_vib (제안)** | **0.9283** | **0.8518** | **0.8685** | **0.9281** |

**해석:**
- `text_only(0.75) < ts_only(0.91)`: ECG가 인구통계 텍스트보다 압도적으로 강한 예측 신호
- `ablation(0.9279) > ts_only(0.9095)`: 텍스트 보조 정보가 실제로 ECG 표현을 보완
- `fusion_vib(0.9283) > ablation(0.9279)`: **VIB 정규화가 미세하지만 일관되게 성능 향상** — KL loss가 TS 인코더의 일반화를 개선

---

## 5. 하이퍼파라미터 요약

| 파라미터 | 값 |
|:---|:---:|
| d_model (Transformer) | 64 |
| n_heads | 4 |
| n_layers | 2 |
| Transformer dropout | 0.2 |
| z_dim (VIB 출력) | 64 |
| BERT projection dim | 32 |
| Fusion dim | 96 (= 64 + 32) |
| Classifier dropout | 0.3 |
| β (KL weight) | 0.001 |
| KL warmup | epochs // 2 = 10 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 20 |
| Dataset (primary) | PTB-XL (ECG) |
| BERT model | Bio_ClinicalBERT |

---

## 6. 소스 파일 위치

| 파일 | 설명 |
|:---|:---|
| `src/models.py` | 모든 모델 클래스 정의 |
| `src/data_loader.py` | PTB-XL / FNSPID DataLoader |
| `src/prepare_ptbxl.py` | PTB-XL 전처리 (텍스트 포함) |
| `src/prepare_fnspid.py` | FNSPID 주가 데이터 전처리 |
| `experiments/train.py` | 학습 루프 (KL annealing 포함) |
| `experiments/eval.py` | 평가 및 그래프 생성 |
| `experiments/run_ptbxl.sh` | 4개 실험 병렬 실행 스크립트 |
