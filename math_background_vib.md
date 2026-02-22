# 시계열 예측을 위한 멀티모달 Information Bottleneck (IB) 수학적 기반 및 심화 이론

본 문서는 논문 작성 시 방법론(Methodology) 및 이론적 배경(Theoretical Background) 챕터에 활용할 수 있도록, Information Bottleneck(IB)의 정보이론적 기반부터 Variational Inference를 통한 최적화 가능 형태(VIB) 유도, 그리고 멀티모달 융합에 대한 심화 수학적 정리를 포함합니다.

---

## 1. Information Theory의 기초 개념

IB를 이해하기 위한 정보이론의 핵심 개념들은 다음과 같이 정의됩니다.

### 1-1. Entropy (엔트로피)
이산 확률 변수 $X$에 대하여, 엔트로피 $H(X)$는 불확실성(Uncertainty)의 척도입니다. (연속 확률 변수의 경우 Differential Entropy $h(X)$)
$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

### 1-2. Kullback-Leibler Divergence (KLD)
두 확률 분포 $P$와 $Q$ 간의 정보적 거리(Information distance, strictly pseudo-distance)를 측정합니다.
$$D_{KL}(P \parallel Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$
**Theorem 1 (Gibbs' Inequality):** 임의의 $P, Q$에 대해 $D_{KL}(P \parallel Q) \ge 0$ 이며, $P=Q$일 때만 $0$입니다.

### 1-3. Mutual Information (상호정보량)
두 확률 변수 $X, Y$가 공유하는 정보의 양으로, 하나의 변수를 앎으로써 감소하는 다른 변수의 불확실성을 의미합니다.
$$I(X; Y) = H(X) - H(X|Y) = D_{KL}(P_{X,Y} \parallel P_X \otimes P_Y) = \mathbb{E}_{p(x,y)}\left[ \log \frac{p(x,y)}{p(x)p(y)} \right]$$

---

## 2. The Information Bottleneck (IB) Principle

Tishby et al. (1999)에 의해 제안된 IB 원리는 입력 $X$가 출력 타겟 $Y$에 대해 갖는 "연관 정보(Relevant Information)"를 포함하는 부호화(Encoding) 변수 $Z$를 찾는 최적화 문제입니다. 

### 2-1. Markov Chain 가설
임베딩 $Z$는 오직 $X$에 의해서만 생성되며, $Y$의 정보는 $X$를 거쳐서만 $Z$로 전달된다고 가정합니다. 따라서 다음의 마르코프 연쇄(Markov Chain)를 만족합니다.
$$Y \leftrightarrow X \leftrightarrow Z$$
**Data Processing Inequality (DPI):** 위와 같은 마르코프 체인에서 $I(Y; X) \ge I(Y; Z)$ 가 항상 성립합니다. 즉 $X$를 어떻게 압축하든 원본 데이터가 가진 $Y$에 대한 정보를 초과할 수는 없습니다.

### 2-2. IB Lagrangian (목적 함수)
IB 모델의 목적은 $Z$를 $X$의 압축된 통계량으로 만들면서($I(X; Z)$ 최소화), $Y$에 대한 서술력은 최대한 유지하는($I(Z; Y)$ 최대화) 것입니다.
$$\min_{p(z|x)} \mathcal{L}_{IB} = I(X; Z) - \beta I(Z; Y)$$
여기서 $\beta \in [0, \infty]$는 Lagrange multiplier로 압축률과 정확도의 Trade-off를 제어합니다.

---

## 3. Variational Information Bottleneck (VIB) 유도 (Alemi et al., 2016)

$I(X; Z)$와 $I(Z; Y)$의 직접적인 계산은 고차원 데이터(시계열, 이미지 등)에서 주변부 분포(Marginal Distribution) $p(z) = \int p(z|x)p(x)dx$를 적분해야 하므로 Intractable(계산 불가)합니다. 따라서 Variational Inference(변분 추론)를 도입해 최적화 가능한 상/하한선(Bounds)을 도출합니다.

### 3-1. 예측 정보량 $I(Z;Y)$의 하한선 (Variational Lower Bound)
사후 확률(Posterior) $p(y|z)$를 계산 불가능하므로, 뉴럴 네트워크 기반의 매개변수화된 근사 예측기(Variational Classifier) $q_{\theta}(y|z)$를 도입합니다.
$$
\begin{align*}
I(Z; Y) &= \iint p(z, y) \log \frac{p(y|z)}{p(y)} dz dy \\
&= \iint p(z, y) \log p(y|z) dz dy - H(Y) \\
&= \iint p(z, y) \log \frac{p(y|z) q_{\theta}(y|z)}{q_{\theta}(y|z)} dz dy - H(Y) \\
&= \iint p(z, y) \log q_{\theta}(y|z) dz dy + \underbrace{\mathbb{E}_{p_z}[D_{KL}(p(y|z) \parallel q_{\theta}(y|z))]}_{\ge 0} - H(Y) \\
&\ge \iint p(z, y) \log q_{\theta}(y|z) dz dy - H(Y)
\end{align*}
$$
$H(Y)$는 주어진 데이터의 상수이므로 생략 가능하며, 이를 전개하면 다음과 같은 형태가 됩니다. (Empirical risk / Cross Entropy)
$$I(Z; Y) \ge \int p(x) \int p(z|x) \int p(y|x) \log q_{\theta}(y|z) dy dz dx \approx \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}_{z \sim p_{\phi}(z|x_i)} [\log q_{\theta}(y_i|z)]$$

### 3-2. 압축 정보량 $I(X;Z)$의 상한선 (Variational Upper Bound)
마찬가지로 주변부 확률 $p(z)$를 근사하는 사전 분포(Prior) $r(z) \sim \mathcal{N}(0, I)$를 도입합니다.
$$
\begin{align*}
I(X; Z) &= \iint p(x, z) \log \frac{p(z|x)}{p(z)} dz dx \\
&= \iint p(x, z) \log \frac{p(z|x) r(z)}{p(z) r(z)} dz dx \\
&= \iint p(x, z) \log \frac{p(z|x)}{r(z)} dz dx - \underbrace{D_{KL}(p(z) \parallel r(z))}_{\ge 0} \\
&\le \iint p(x, z) \log \frac{p(z|x)}{r(z)} dz dx \\
&= \mathbb{E}_{x \sim p(x)} \left[ D_{KL}(p_{\phi}(z|x) \parallel r(z)) \right]
\end{align*}
$$

### 3-3. 최종 VIB 목적 함수 (ELBO)
위 두 바운드를 IB Lagrangian $\mathcal{L}_{IB}$에 대입하여 $\min \mathcal{L}_{IB}$를 최소화 가능한 손실함수로 수식화하면 최종 실증적 VIB Loss(Evidence Lower Bound 대응)가 유도됩니다. (관례상 부호를 반전시켜 딥러닝에서 최소화할 수 있는 Loss로 씁니다.)

$$\mathcal{J}_{VIB} = \frac{1}{N} \sum_{i=1}^{N} \left[ - \mathbb{E}_{z \sim p_{\phi}(z|x_i)} [\log q_{\theta}(y_i|z)] + \beta D_{KL}(p_{\phi}(z|x_i) \parallel r(z)) \right]$$

---

## 4. 구조화된 다변량 정규 분포 하에서의 KL-Divergence

본 모델에서는 시계열 데이터를 처리하는 Transformer를 $\phi$ 로 사용하여, 잠재 변수 $z$가 대각 공분산 다변량 가우시안 형식을 띤다고 가정합니다.
$$p_{\phi}(z|x) = \mathcal{N}(z ; \mu_{\phi}(x), \operatorname{diag}(\sigma_{\phi}^2(x)))$$
사전 분포를 $r(z) = \mathcal{N}(0, I)$ 로 둘 때, 두 가우시안 분포 간의 수식은 解析的(Analytical)으로 구해집니다.
$$D_{KL}\left( \mathcal{N}(\mu, \sigma^2 I) \parallel \mathcal{N}(0, I) \right) = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
*해석:* $\sigma$가 $1$에 수렴하고 $\mu$가 $0$에 수렴할수록 페널티가 0이 되며, 특정 시점의 데이터 특징(Feature)이 강한 정규성을 갖게 되어 노이즈(오버피팅)가 강하게 억제됩니다.

### 4-1. Reparameterization Trick 하에서의 Gradient 추정
$\mathbb{E}_{z \sim p_{\phi}}[\cdot]$ 항에 대하여 오차역전파법(Backpropagation)을 적용하기 위해, 확률 분포를 통한 미분이 가능하도록 무작위 노이즈 $\epsilon$에 대한 선형 변환으로 모델링 식을 분리합니다.
$$z = g(\mu, \sigma, \epsilon) = \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
이로써 연산 그래프(Computation Graph)의 미분 연결성이 성립합니다.
$$\nabla_{\phi} \mathbb{E}_{z}[f(z)] = \mathbb{E}_{\epsilon}[\nabla_{\phi} f(g(\phi, \epsilon))]$$

---

## 5. Multimodal Fusion with Auxiliary Text (제안 아키텍처 수식화)

본 연구에서 제안하는 VIB-Transformer 시계열 인코더와 BERT 텍스트 인코더 간의 정보 결합 과정입니다.

### 5-1. Time-Series Encoding ($x_{ts}$)

PTB-XL 기준, ECG 피처 행렬 $x_{ts} \in \mathbb{R}^{T \times F}$ ($T=1000$, $F=12$)가 Transformer 인코더를 거칩니다.

$$\tilde{x}_{ts} = \sqrt{d_{model}} \cdot W_{emb}\, x_{ts} + PE$$

여기서 $PE \in \mathbb{R}^{1 \times T \times d_{model}}$는 학습 가능한 위치 인코딩입니다.

$$H = \operatorname{TransformerEncoder}(\tilde{x}_{ts}) \in \mathbb{R}^{T \times d_{model}}$$

**Mean Pooling** (전체 타임스텝 평균 — 세 모델 모두 동일):
$$h_{pool} = \frac{1}{T}\sum_{t=1}^{T} H_t \in \mathbb{R}^{d_{model}}$$

VIB로 확률적 병목을 거칩니다 ($d_{model} = z_{dim} = 64$):
$$\mu_{ts} = W_{\mu}\, h_{pool} + b_{\mu}, \quad \log\sigma^2_{ts} = W_{\sigma}\, h_{pool} + b_{\sigma}$$
$$Z_{ts} \sim \mathcal{N}(\mu_{ts},\, \operatorname{diag}(\sigma^2_{ts})) \qquad Z_{ts} \in \mathbb{R}^{64}$$

### 5-2. Text Encoding via Bio_ClinicalBERT ($x_{text}$)

환자의 인구통계 정보와 ECG 신호 품질 플래그로 구성된 자연어 문장을 인코딩합니다.

$$Z_{text} = \operatorname{Proj}\!\left(\operatorname{Bio\_ClinicalBERT}(x_{text})_{[\text{CLS}]}\right) \in \mathbb{R}^{32}$$

여기서 $\operatorname{Proj} : \mathbb{R}^{768} \to \mathbb{R}^{128} \to \mathbb{R}^{32}$는 계층적 투영(Projection) MLP입니다. 텍스트 인코더에는 **Information Bottleneck을 적용하지 않습니다** — 텍스트는 보조(side) 정보로, 병목 없이 원본 표현을 전달합니다.

### 5-3. Joint Fusion Classifier ($q_{\theta}$)

$$Z_{fused} = \operatorname{Concat}(Z_{ts},\, Z_{text}) \in \mathbb{R}^{96}$$
$$\hat{y} = q_{\theta}(y \mid Z_{fused}) = \operatorname{Softmax}\!\left(\operatorname{MLP}(Z_{fused})\right)$$

### 5-4. 최적화 목적 함수

$$\mathcal{J} = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_{CE}(y_i,\, \hat{y}_i) + \beta_t \cdot D_{KL}\!\left(\mathcal{N}(\mu_{ts}^{(i)},\, \sigma_{ts}^{(i)2}) \;\Big\|\; \mathcal{N}(0, I)\right)$$

이를 IB Lagrangian으로 해석하면:
$$\max_{Z_{ts},\, \theta}\; I(Z_{ts}, Z_{text};\, Y) - \beta_t\, I(X_{ts};\, Z_{ts})$$

**설계 원리:** ECG 신호(노이즈 중심)는 VIB로 핵심 패턴만 추출하고, 인구통계·신호품질 텍스트(이벤트 중심)는 있는 그대로 판단 근거에 반영합니다.

### 5-5. KL Annealing 스케줄

학습 초기 KL 페널티를 점진적으로 증가시켜 모델이 먼저 예측 능력을 확보하도록 합니다.

$$\beta_t = \beta \cdot \min\!\left(1,\; \frac{t}{\lfloor\text{epochs}/2\rfloor}\right), \qquad t = 1, 2, \dots, \text{epochs}$$

- `epochs=20`, `beta=0.001` 기준: epoch 10까지 선형 증가 → 이후 $\beta = 0.001$ 고정
- epoch 1부터 $\beta_1 = 0.0001 > 0$ 이므로 첫 에폭부터 VIB 학습이 시작됩니다.

### 5-6. 최종 실험 결과 (PTB-XL ECG 이진 분류)

| 모델 | Best AUC | Last Acc | Last F1 | Last AUC |
|:---|:---:|:---:|:---:|:---:|
| `ts_only` | 0.9095 | 0.8317 | 0.8568 | 0.9064 |
| `text_only` | 0.7528 | 0.7096 | 0.7738 | 0.7518 |
| `ablation` (fusion, no VIB) | 0.9279 | 0.8358 | 0.8534 | 0.9255 |
| **`fusion_vib` (제안)** | **0.9283** | **0.8518** | **0.8685** | **0.9281** |

ablation 대비 fusion_vib가 모든 지표에서 향상되어, VIB 정규화가 멀티모달 융합 표현의 일반화 능력을 개선함을 보입니다.

