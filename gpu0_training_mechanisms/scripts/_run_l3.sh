#!/bin/bash
source /root/.secrets/llm_api_keys.env
cd /root/experiments/dreaming-in-code-coop
PYTHONPATH=src:$PYTHONPATH python scripts/generate_llm_judgments.py 2>&1 | tee /root/experiments/dicode_runs/aggregation/llm_pilot/l3_run.log
echo "EXIT=$?" >> /root/experiments/dicode_runs/aggregation/llm_pilot/l3_run.log
