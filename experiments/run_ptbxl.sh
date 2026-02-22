#!/bin/bash
# =======================================================
# PTB-XL 4개 실험 병렬 실행 스크립트
#   GPU 0 → ts_only
#   GPU 1 → text_only
#   GPU 3 → ablation
#   GPU 5 → fusion_vib
# 실행 방법: bash run_ptbxl.sh
# =======================================================

# ----- conda 환경 설정 -----
CONDA_SH="/home/ku_ai1/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="IB_LLM"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$ENV_NAME"
    echo "[OK] conda 환경 활성화: $ENV_NAME"
else
    echo "[WARN] conda.sh not found at $CONDA_SH"
    echo "       IB_LLM 환경이 이미 활성화되어 있다고 가정합니다."
fi

# ----- 작업 디렉토리 이동 -----
cd /home/ku_ai1/sulee/졸업논문/experiments

# ----- 하이퍼파라미터 -----
DATASET="ptbxl"
EPOCHS=20
BATCH_SIZE=32
LR=1e-4
BETA=0.001

# ----- 로그 디렉토리 생성 -----
mkdir -p logs

echo ""
echo "======================================================="
echo " PTB-XL 병렬 실험 시작 [epochs=$EPOCHS, beta=$BETA]"
echo "   GPU 0 → ts_only"
echo "   GPU 1 → text_only"
echo "   GPU 3 → ablation"
echo "   GPU 5 → fusion_vib"
echo "======================================================="
echo ""

# ----- 4개 실험 동시 실행 -----
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset $DATASET --exp ts_only \
    --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --beta $BETA \
    > logs/ts_only.log 2>&1 &
PID0=$!
echo "[GPU 0] ts_only    started (PID $PID0) → logs/ts_only.log"

CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset $DATASET --exp text_only \
    --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --beta $BETA \
    > logs/text_only.log 2>&1 &
PID1=$!
echo "[GPU 1] text_only  started (PID $PID1) → logs/text_only.log"

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset $DATASET --exp ablation \
    --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --beta $BETA \
    > logs/ablation.log 2>&1 &
PID3=$!
echo "[GPU 3] ablation   started (PID $PID3) → logs/ablation.log"

CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset $DATASET --exp fusion_vib \
    --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --beta $BETA \
    > logs/fusion_vib.log 2>&1 &
PID5=$!
echo "[GPU 5] fusion_vib started (PID $PID5) → logs/fusion_vib.log"

echo ""
echo "모니터링: tail -f logs/ts_only.log logs/text_only.log logs/ablation.log logs/fusion_vib.log"
echo ""

# ----- 모든 실험 완료 대기 -----
wait $PID0 && echo "[GPU 0] ts_only    DONE" || echo "[GPU 0] ts_only    FAILED"
wait $PID1 && echo "[GPU 1] text_only  DONE" || echo "[GPU 1] text_only  FAILED"
wait $PID3 && echo "[GPU 3] ablation   DONE" || echo "[GPU 3] ablation   FAILED"
wait $PID5 && echo "[GPU 5] fusion_vib DONE" || echo "[GPU 5] fusion_vib FAILED"

echo ""
echo "======================================================="
echo " 전체 학습 완료! 결과 요약:"
echo "======================================================="
python show_results.py

echo ""
echo "평가 및 그래프 생성..."
python eval.py --dataset $DATASET

echo "완료."
