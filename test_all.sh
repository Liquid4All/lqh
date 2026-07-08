#!/usr/bin/env bash
set -e

source .venv/bin/activate

# python -m pytest tests/function/test_cloud_finetune_agent_e2e.py -q
# python -m pytest tests/e2e/test_full_pipeline.py -q
# python -m pytest tests/e2e/test_full_pipeline_tools.py -q
# python -m pytest tests/unit/test_compute_resolve.py -q
# python -m pytest tests/unit/test_training.py -q -k "TrainingToolValidation or training_status or start_training"
# python -m pytest tests --collect-only -q


 uv run python -m tests.benchmarks.base_vs_instruct.run \
    --run-name bvi-256-sft \
    --tasks messy_extraction,style_rewrite \
    --models 350M-Instruct,350M-Base,1.2B-Instruct,1.2B-Base \
    --train-size 10000 \
    --eval-size 400 \
    --grid-size small \
    --dpo-train-size 0
