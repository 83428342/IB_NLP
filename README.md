# Conditional Information Bottleneck for Asymmetric Multimodal Time-Series Prediction

This repository contains the PyTorch implementation of the multimodal architecture proposed in the paper: **"Conditional Information Bottleneck for Asymmetric Multimodal Time-Series Prediction"**.

Our model fuses continuous time-series data (e.g., 12-lead ECG signals, daily financial metrics) with auxiliary discrete text data (e.g., patient demographics & signal quality metadata, financial news headlines). To prevent the high-variance time-series data from overwhelming the text signal, we apply a **Variational Information Bottleneck (VIB)** regularization strictly to the time-series encoder.

---

## Key Features

- **Domain-Agnostic Design:** Evaluated on two highly distinct domains:
  1. **Medical:** PTB-XL (12-lead ECG + Patient Metadata) $\to$ Cardiac rhythm abnormality detection.
  2. **Financial:** FNSPID (NASDAQ-100 Stock Prices + Financial News) $\to$ Next-day stock movement prediction.
- **Asymmetric Multimodal Fusion:** 
  - Time-Series signals are modeled via a Transformer Encoder and compressed into a minimal sufficient representation ($Z_{ts}$) using the Reparameterization Trick and KL-Divergence annealing.
  - Text signals are embedded via frozen Pre-Trained Language Models (e.g., `emilyalsentzer/Bio_ClinicalBERT`) without an information bottleneck ($Z_{text}$), ensuring critical auxiliary context is preserved.
- **Automated Experiment Pipeline:** Built-in scripts to cleanly download data, preprocess sequences, train four ablation models in parallel, and generate publication-ready plots.

---

## Directory Structure

```text
.
├── data/                    # Automatically populated by download scripts
├── experiments/             
│   ├── run_ptbxl.sh         # Parallel execution script for all 4 PTB-XL experiments
│   ├── train.py             # Main training loop (KL Annealing + Cross Entropy)
│   ├── eval.py              # Evaluates models and generates ROC/Loss figures
│   └── show_results.py      # Quick console summary of all metrics
└── src/                     
    ├── prepare_ptbxl.py     # Downloads and merges PTB-XL ECGs and clinical data
    ├── prepare_fnspid.py    # Downloads and merges FNSPID stock and news data
    ├── data_loader.py       # PyTorch DataLoader (Sliding windows + Tokenization)
    └── models.py            # Model definitions (VIB, Transformer, BERT, Fusion)
```

## Installation & Setup

We recommend using Anaconda to isolate the environment.

```bash
conda create -n IB_LLM python=3.10
conda activate IB_LLM
pip install torch torchvision torchaudio
pip install transformers wfdb scikit-learn pandas numpy matplotlib seaborn tqdm yfinance
```

##  How to Run the Experiments (PTB-XL)

### 1. Data Preparation
Run the preparation script to download the PhysioNet dataset and construct the multimodal CSV.
*(Note: This downloads ~2GB of ECG waveform data.)*
```bash
python src/prepare_ptbxl.py
```

### 2. Training (Parallel Execution)
Navigate to the `experiments` directory and run the shell script. This will launch all four experimental conditions (`ts_only`, `text_only`, `ablation`, `fusion_vib`) concurrently across different GPUs.

**(Ensure you edit `run_ptbxl.sh` to match your available `CUDA_VISIBLE_DEVICES` before running.)**

```bash
cd experiments
bash run_ptbxl.sh
```

### 3. Evaluation & Results
Once training is complete, invoke the evaluation scripts to print metrics and generate plots.

```bash
python show_results.py
python eval.py --dataset ptbxl
```
This generates `roc_comparison.png`, `loss_curves.png`, and a confusion matrix in the current directory.

---

## Experimental Results (PTB-XL)

By applying a controlled Information Bottleneck to the time-series encoder (`beta=0.001`), the proposed `fusion_vib` model achieves the highest AUC, outperforming the direct concatenation baseline (`ablation`).

| Model Configuration | Best AUC | Final Accuracy | Final F1-Score | Final AUC |
|:---|:---:|:---:|:---:|:---:|
| `ts_only` (ECG only) | 0.9095 | 0.8317 | 0.8568 | 0.9064 |
| `text_only` (Meta only) | 0.7528 | 0.7096 | 0.7738 | 0.7518 |
| `ablation` (Fusion w/o VIB) | 0.9279 | 0.8358 | 0.8534 | 0.9255 |
| **`fusion_vib` (Proposed)** | **0.9283** | **0.8518** | **0.8685** | **0.9281** |

See `model.md` and `math_background_vib.md` for the exact mathematical formulation of the loss function and KL-annealing schedule.
