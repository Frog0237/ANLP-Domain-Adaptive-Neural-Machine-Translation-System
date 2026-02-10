#!/bin/bash
# End-to-end NMT pipeline: Train → Evaluate → Diagnose → Report
# Usage: bash scripts/run_pipeline.sh
# 说明：该脚本按训练→评估→诊断→搜索顺序执行。

set -e

echo "============================================"
echo "  NMT Pipeline - Full Run"
echo "============================================"

# Step 1: Train from scratch
echo ""
echo "[1/4] Training model..."
python -m src.train --config configs/train_config.yaml

# Optional: Fine-tune pretrained model (set RUN_FINETUNE=1)
if [ "${RUN_FINETUNE}" = "1" ]; then
  echo ""
  echo "[1b] Fine-tuning pretrained model..."
  python -m src.train --config configs/finetune_config.yaml
fi

# Optional: Train Marian with ALiBi (set RUN_ALIBI=1)
if [ "${RUN_ALIBI}" = "1" ]; then
  echo ""
  echo "[1c] Training Marian with ALiBi..."
  python -m src.train --config configs/alibi_config.yaml
fi

# Optional: Train relative-position T5 (set RUN_RELATIVE=1)
if [ "${RUN_RELATIVE}" = "1" ]; then
  echo ""
  echo "[1d] Training relative-position T5 model..."
  python -m src.train --config configs/relative_t5_config.yaml
fi

# Step 2: Evaluate
echo ""
echo "[2/4] Evaluating models..."
python -m src.evaluate --config configs/eval_config.yaml

# Step 3: Diagnose
echo ""
echo "[3/4] Running attention coverage diagnosis..."
python -m src.diagnose --model-path outputs/my-de-en-nmt --split iid --output-dir outputs/diagnosis
python -m src.diagnose --model-path outputs/my-de-en-nmt --split ood --output-dir outputs/diagnosis

# Step 4: Decoding search
echo ""
echo "[4/4] Running decoding strategy search..."
python -m src.decode_search --config configs/decode_search.yaml

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in outputs/"
echo "============================================"

