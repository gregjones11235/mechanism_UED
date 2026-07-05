#!/bin/bash
#SBATCH -J dicode-v6-s0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o dicode-v6-s0-%j.out
#SBATCH -e dicode-v6-s0-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu

# v6 = v5y coop 机制 (modeler + 2 coop proposers 顺序补位) 原样保留, 只改 PROVIDER 省钱
# (v6模型省钱策略.md). Independent identity: NEW wandb run (dicode-v6-s0) + NEW output dir (v6_s0)
# + NEW code dir (~/dicode_v6). NEVER touches v5 (jobs 3640867 v5yA / 3641416 v5yB still running).
#
# Provider 切换:
#   proposers[0] DeepSeek-V4  : DeepInfra -> DeepSeek 官方   (out -66%, key ~/.deepseek_key)
#   proposers[1] Qwen3.5-397B : DeepInfra -> 阿里百炼美国区   (out -78%, 人民币, key ~/.dashscope_key)
#   task/env/critic Qwen3-235B: DeepInfra -> 阿里百炼美国区   (env_generator = biggest out sink)
#   modeler GLM-5.2           : 留 DeepInfra                 (用户决定, GLM 官方折算反贵 ~30%)
#   embedding Qwen3-Emb-0.6B  : 留 DeepInfra                 (切换会改向量维度/分布, 破坏 conditioning)

echo "dicode-v6 START $(date)"
echo "host: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  JOBID: $SLURM_JOB_ID"

source /oscar/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate dicode || { echo FATAL conda; exit 1; }

# CRITICAL (editable-pth-overrides-dir-copy-use-pythonpath): the conda env's editable .pth points at
# ~/dicode_auction (v4). Prepend v6 so THIS job loads v6, leaving every other run's env untouched.
export PYTHONPATH="$HOME/dicode_v6/src:$HOME/dicode_v6:$PYTHONPATH"

# --- API keys: each provider reads ITS OWN key (llm.py v6 clean split, §5). ---
export DEEPINFRA_API_KEY=$(cat ~/.deepinfra_key)   # GLM modeler + embedding stay here
export DEEPSEEK_API_KEY=$(cat ~/.deepseek_key)     # DeepSeek 官方 proposer
export DASHSCOPE_API_KEY=$(cat ~/.dashscope_key)   # 阿里美国区(dashscope-us) (Qwen3.5 proposer + Qwen3-235B roles)
echo "DEEPINFRA key len: ${#DEEPINFRA_API_KEY}  DEEPSEEK key len: ${#DEEPSEEK_API_KEY}  DASHSCOPE key len: ${#DASHSCOPE_API_KEY}"
[ ${#DEEPSEEK_API_KEY} -gt 0 ] || { echo "FATAL deepseek_key_empty (create ~/.deepseek_key)"; exit 1; }
[ ${#DASHSCOPE_API_KEY} -gt 0 ] || { echo "FATAL dashscope_key_empty (create ~/.dashscope_key)"; exit 1; }

# --- Provider self-checks: a 200 here ALSO hard-verifies the model id is correct (§6 待办 1).
#     If a model id is wrong the endpoint returns 4xx -> FATAL here, not silently mid-run. ---
DI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $DEEPINFRA_API_KEY" -H "Content-Type: application/json" -d '{"model":"zai-org/GLM-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' https://api.deepinfra.com/v1/openai/chat/completions)
echo "DeepInfra(GLM) self-check: $DI_STATUS"; [ "$DI_STATUS" = "200" ] || { echo "FATAL deepinfra_not_200 ($DI_STATUS)"; exit 1; }

# DeepSeek 官方: model id 'deepseek-v4-pro' (✅Oscar实测2026-07-04); thinking:disabled 关思考
# (reasoning_effort:none 会 400 — v4-pro 官方默认 thinking ON, 与 DeepInfra 相反).
DS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $DEEPSEEK_API_KEY" -H "Content-Type: application/json" -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"thinking":{"type":"disabled"}}' https://api.deepseek.com/v1/chat/completions)
echo "DeepSeek官方 self-check: $DS_STATUS"; [ "$DS_STATUS" = "200" ] || { echo "FATAL deepseek_not_200 ($DS_STATUS) — check model id 'deepseek-v4-pro' + key"; exit 1; }

# 阿里百炼 美国(弗吉尼亚) dashscope-us self-check (proposer Qwen3.5-397B + Qwen3-235B roles).
# ✅ model id 实测确认 (Oscar 2026-07-04, dashscope-us /models). key 是美国区域绑定 (北京/新加坡端点会401)。
for M in "qwen3.5-397b-a17b" "qwen3-235b-a22b-thinking-2507"; do
  AS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions)
  echo "阿里dashscope-us self-check [$M]: $AS_STATUS"; [ "$AS_STATUS" = "200" ] || { echo "FATAL dashscope_not_200 ($AS_STATUS) for model '$M' — key 区域是否美国? id 是否在 /models?"; exit 1; }
done

export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)
export WANDB_RUN_ID="dicode-v6-s0"
FIXED_OUT=/oscar/scratch/$USER/dicode_outputs/v6_s0
mkdir -p "$FIXED_OUT"
echo "output dir (resumable, NEW): $FIXED_OUT   wandb run (NEW): $WANDB_RUN_ID"

cd ~/dicode_v6

# --- sanity: fail fast if the sync did not land / loaded the wrong code path ---
python -c "import jax; print('devices:', jax.devices()); assert jax.default_backend()=='gpu'" || { echo FATAL no_gpu; exit 1; }
python -c "import dicode, os; assert os.path.realpath(dicode.__file__).startswith(os.path.realpath(os.path.expanduser('~/dicode_v6'))), dicode.__file__; print('dicode loaded from v6:', os.path.dirname(dicode.__file__))" || { echo FATAL dicode_not_v6; exit 1; }
python -c "import auction; from auction.craftax_achievements import NUM_ACHIEVEMENTS; print('auction import OK', NUM_ACHIEVEMENTS)" || { echo FATAL auction_import; exit 1; }
python -c "from dicode.dreaming.gen_manager import TaskGenerator; assert hasattr(TaskGenerator,'evolve_mastered_coop') and hasattr(TaskGenerator,'_ensure_modeler')" || { echo FATAL coop_method_missing; exit 1; }
# v6-specific: new provider branches present in llm.py
python -c "from dicode.dreaming.llm import LLM; import inspect; s=inspect.getsource(LLM._create_client); assert 'deepseek' in s and 'dashscope' in s, 'v6 provider branch missing'; print('v6 providers OK')" || { echo FATAL v6_provider_missing; exit 1; }
python -c "import yaml; c=yaml.safe_load(open('conf/gen_manager/auction_c_v6.yaml')); assert c['auction_modeler'] is True and c['auction'] is False; assert c['proposers'][0]['provider']=='deepseek' and c['proposers'][1]['provider']=='dashscope'; assert c['modeler']['provider']=='deepinfra'; print('v6 config OK')" || { echo FATAL v6_config_bad; exit 1; }
echo "v6 sanity: v6 code path, coop+modeler present, new providers wired, v6 config OK"

python -m pytest auction/tests/ -q 2>&1 | tail -3 || { echo FATAL auction_tests_fail; exit 1; }

# --- LLM role overrides: proposers come from auction_c_v6.yaml; the 3 NON-proposer generation roles
#     move to 阿里美国区(dashscope-us) (dashscope). embedding STAYS on DeepInfra (dim/dist would break conditioning). ---
OV_LLM="gen_manager/llm@gen_manager.task_generator=dashscope gen_manager/llm@gen_manager.env_generator=dashscope gen_manager/llm@gen_manager.interestingness_critic=dashscope gen_manager/llm@gen_manager.embedding_model=deepinfra_embed"

echo "=== v6: v5y coop 机制 + 省钱 provider (DeepSeek官方 + 阿里美国区(dashscope-us); GLM/embedding 留 DeepInfra) ==="
python -u experiments/training/run_dicode.py \
  hydra.run.dir="$FIXED_OUT" \
  gen_manager=auction_c_v6 \
  $OV_LLM \
  use_wandb=true \
  wandb_project=DiCode-v6 \
  wandb_entity=gregjones11235-brown-university \
  training.load_checkpoint=true \
  seed=0

echo "dicode-v6 END $(date)"
echo "outputs in: $FIXED_OUT"
