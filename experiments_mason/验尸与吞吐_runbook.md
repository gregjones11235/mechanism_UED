# 一次开机 Runbook:深层失败验尸取数 + 吞吐税日志切片

> 目标:开一次 pod,~20 分钟计费,取回两份数据后 stop。之后全部分析在本地。

## 0. 前置(Windows 侧,开机前)
桌面 clone 打最新 patch(eval 明细补丁 + 本 runbook + throughput_tax.py)并 push。

## 1. 开机 + 环境(~5 min)
source /workspace/venv/bin/activate
which ollama || curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve >> /workspace/ollama_server.log 2>&1 &
sleep 5
curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5-coder:14b","prompt":"hi","stream":false,"keep_alive":-1}' > /dev/null
curl -s http://localhost:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"hi","keep_alive":-1}' > /dev/null
cd /workspace/mechanism_UED && git pull
cd dicode_src && uv run pytest src/dicode/skill_preflight/tests/ -q   # 62 passed

## 2. 验尸取数:终点单点明细 eval(~10 min)
tmux new-session -d -s evald "cd /workspace/mechanism_UED/dicode_src && source /workspace/venv/bin/activate && \
export GENERATION_SERVER_URL=http://localhost:11434/v1 EMBEDDING_SERVER_URL=http://localhost:11434/v1 && \
python experiments/training/eval_checkpoints.py \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  use_wandb=false \
  '+eval.ckpt_root=/workspace/mechanism_UED/dicode_src/outputs/2026-07-12_043418_840474/rl_checkpoints' \
  '+eval.steps=[8500]' '+eval.tag=AUTOPSY' '+eval.details=true' seed=0 \
  > /workspace/eval_autopsy.log 2>&1"
# 完成判定:tail 见两行 [saved](汇总 + _details.json)。sanity:汇总 mean_return 应 =43.47±0.01
#(与 eval_C2LITE2E9SUP 的 8500 逐位对上,证明 detail 路径未污染主计算)。

## 3. 吞吐税日志切片(~2 min,纯 grep)
for L in run_C2lite_2e9.log run_t04_3e8.log run_C2lite_v2_3e8.log; do
  grep -aE "Starting Session|\[Preflight\] kept" /workspace/$L > /workspace/slice_$L.txt 2>/dev/null
done
ls -lh /workspace/slice_*.txt   # 应各几十 KB

## 4. 下载三样 → stop pod
- outputs/<新目录>/eval_AUTOPSY_seed0.json + *_details.json
- /workspace/slice_*.txt(三份)
JupyterLab 直接下载。然后 STOP POD。

## 5. 本地分析(零 GPU)
- 验尸:_details.json 交给 Claude(死因 × max_floor 二维表 + episode 时长假说裁决);
- 吞吐:python experiments_mason/throughput_tax.py slice_run_C2lite_2e9.log.txt \
        experiments_mason/eval/eval_C2LITE2E9_seed0.json
  (主跑切片跨两段,session 号有 resume 归零 —— 脚本按 session×13.1M 换算,resumed 段
   切片单独跑一次并在解读时手动加 865M 偏移;t04 切片直接跑。)
