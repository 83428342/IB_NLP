# 📚 데이터셋 설명서

본 프로젝트는 **두 개의 이종(Heterogeneous) 도메인 데이터셋**을 사용하여 VIB 기반 멀티모달 융합 프레임워크의 도메인 불변 일반화 능력을 검증합니다.

---

## 도메인 1: PTB-XL (의료 — ECG 분류)

### 1-1. 개요

**PTB-XL**은 PhysioNet에서 제공하는 공개 대규모 ECG 데이터셋입니다.

| 항목 | 값 |
|:---|:---|
| 총 레코드 수 | 21,837명의 환자, 21,837개 ECG |
| 샘플링 주파수 | 100 Hz |
| 시계열 길이 | 10초 → **1,000 타임스텝** |
| 리드(채널) 수 | **12 리드** (표준 12-lead ECG) |
| 출처 | PhysioNet / `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3` |

- **전처리 스크립트:** `src/prepare_ptbxl.py`

### 1-2. 이진 분류 레이블

`scp_codes`의 likelihood 값을 기반으로 심전도 이상 여부를 이진화합니다.

- `Label = 1` (비정상): NORM 코드 외에 100% confidence 진단이 포함된 경우
- `Label = 0` (정상): NORM(정상 리듬) 100% confidence만 존재

> 클래스 비율: 정상 약 44%, 비정상 약 56% (적절한 균형)

### 1-3. 보조 텍스트 (자연어 임베딩 입력)

**데이터 누수(Data Leakage) 방지**를 위해 임상 소견서(`report` 컬럼, 진단 내용 직접 포함)를 사용하지 않습니다. 대신 환자의 **인구통계 정보 및 ECG 신호 품질 플래그**로 자연어 문장을 구성합니다.

```
Patient: 56yo female. BMI: 24.2. Device: CS-12.
Signal quality: baseline drift, no noise.
```

| 구성 요소 | 원본 컬럼 |
|:---|:---|
| 나이, 성별 | `age`, `sex` |
| BMI | `height`, `weight` (계산값) |
| 기록 장비 | `device` |
| 신호 품질 | `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`, `extra_beats` |

> **설계 의도:** 텍스트는 ECG 신호에 담기지 않는 **임상 컨텍스트(환자 특성, 측정 환경)**를 보조 정보로 제공합니다.

### 1-4. 데이터 분할

- Train / Validation / Test = 70% / 15% / 15% (stratified)
- 총 학습 배치: 약 15,000개 × 70% ÷ 32(batch) = **약 328 steps/epoch** (실제 545 steps)

### 1-5. 전처리 파이프라인 요약

```
PhysioNet ZIP 다운로드
    ↓
wfdb 라이브러리로 .dat/.hea 파싱 → (N, 1000, 12) numpy 배열
    ↓
scp_codes → is_abnormal() → 이진 레이블
    ↓
환자 메타데이터 → build_auxiliary_text() → 자연어 문장
    ↓
PTBXL_MULTI_merged.csv 저장 (ecg_path, label, article)
```

---

## 도메인 2: FNSPID Multi-Stock (금융 — 주가 방향 예측)

### 2-1. 개요

미국 NASDAQ 상위 23개 기술주에 대한 **일별 주가 시계열 + 금융 뉴스** 멀티모달 데이터셋입니다.

| 항목 | 값 |
|:---|:---|
| 종목 수 | 23개 (AAPL, MSFT, NVDA, TSLA 등) |
| 전체 행 수 | 약 16,680일치 (수직 스태킹) |
| 시계열 피처 | Open, High, Low, Close, Volume, Returns (6개) |
| 윈도우 크기 | **5일 슬라이딩 윈도우** |
| 출처 | HuggingFace `benstaf/FNSPID-nasdaq-100-post2019-1newsperrow` + Stooq Finance |

- **전처리 스크립트:** `src/prepare_fnspid.py`

### 2-2. 이진 분류 레이블

```
Label = 1: 내일 종가 ≥ 오늘 종가 (상승 또는 유지)
Label = 0: 내일 종가 < 오늘 종가 (하락)
```

> 클래스 비율: 상승(1) 약 52.3%, 하락(0) 약 47.7%

### 2-3. 보조 텍스트

해당 종목·날짜의 금융 뉴스. 하루 여러 기사가 있는 경우 `[SEP]`로 연결합니다.

### 2-4. Cross-Ticker Bleeding 방지

23개 종목 데이터가 하나의 CSV에 수직 병합되어 있으므로, `groupby('Stock_symbol')` 기반의 **유효 인덱스 매핑**으로 종목 경계가 교차하는 슬라이딩 윈도우를 제거합니다.

---

## DataLoader 설정 (`src/data_loader.py`)

`--dataset` 인수에 따라 자동으로 전환됩니다:

```python
get_dataloaders(data_dir, dataset_name='ptbxl', batch_size=32, window_size=1000)
get_dataloaders(data_dir, dataset_name='fnspid', batch_size=32, window_size=5)
```

| 설정 | PTB-XL | FNSPID |
|:---|:---:|:---:|
| `window_size` | 1000 | 5 |
| `feature_dim` | 12 | 6 |
| `BERT model` | Bio_ClinicalBERT | bert-base-uncased |
| 텍스트 출처 | 환자 메타데이터 | 금융 뉴스 |
