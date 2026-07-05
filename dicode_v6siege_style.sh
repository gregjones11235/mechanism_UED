#!/bin/bash
#SBATCH -J dicode-v6siege-style-s0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o dicode-v6siege-style-s0-%j.out
#SBATCH -e dicode-v6siege-style-s0-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiayu_zhu@brown.edu

# v6-SIEGE = the v6 METHOD (Focused Milestone-Chain Curriculum, v6_design.md §3) on top of the v5y
# coop mechanism + v6 cost-saving providers. THIS is the method run; dicode_v6.sh is the pure
# provider-switch control (v5y mechanism unchanged) — keep the two independent so "provider switch"
# and "siege mechanism" stay separable variables.
#
# Independent identity (never collides with the control run or v5): NEW wandb run (dicode-v6siege-style-s0)
# + NEW output dir (v6siege_style_s0) + SAME code dir (~/dicode_v6, siege gated by config.siege). The siege
# state (siege_notebook.json / student_profile_history.json) lives in the output dir, so it survives
# resume and is isolated from the control run's dir.
#
# What siege: true adds over auction_c_v6 (all purely additive; siege off == byte-for-byte v5y coop):
#   - SiegeNotebook: persistent cross-session siege journal (focus / prereq chain / verified chains /
#     protected set) with B-layer hard constraints (SR-computed mastery, §3.2 scope, anti-thrash,
#     EARLY-TRAINING maturity gate, code-driven conquest).
#   - modeler.diagnose_siege: v5 diagnosis + a proposed notebook update (focus + prereq_tree),
#     depth inferred from SR + mechanics, never a tier table.
#   - §3.4 Completed gate: proposer SIEGE_DIRECTIVE + code backstop (completed_gate.py) pulls any
#     unmastered chain link out of Completed back into Relevant.
#   - §3.6 rehearsal: append (NOT replace) rehearsal levels for FORGETTING protected skills, siege
#     batch kept in full, capped by rehearsal_total_cap (24).
#
# Provider 切换 (identical to dicode_v6.sh):
#   proposers[0] DeepSeek-V4  : DeepSeek 官方   (key ~/.deepseek_key)
#   proposers[1] Qwen3.5-397B : 阿里百炼美国区   (key ~/.dashscope_key)
#   task/env/critic Qwen3-235B: 阿里百炼美国区
#   modeler GLM-5.2           : 留 DeepInfra
#   embedding Qwen3-Emb-0.6B  : 留 DeepInfra

echo "dicode-v6siege START $(date)"
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

# --- Provider self-checks (a 200 also hard-verifies the model id). ---
DI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $DEEPINFRA_API_KEY" -H "Content-Type: application/json" -d '{"model":"zai-org/GLM-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' https://api.deepinfra.com/v1/openai/chat/completions)
echo "DeepInfra(GLM) self-check: $DI_STATUS"; [ "$DI_STATUS" = "200" ] || { echo "FATAL deepinfra_not_200 ($DI_STATUS)"; exit 1; }

DS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $DEEPSEEK_API_KEY" -H "Content-Type: application/json" -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"thinking":{"type":"disabled"}}' https://api.deepseek.com/v1/chat/completions)
echo "DeepSeek官方 self-check: $DS_STATUS"; [ "$DS_STATUS" = "200" ] || { echo "FATAL deepseek_not_200 ($DS_STATUS) — check model id 'deepseek-v4-pro' + key"; exit 1; }

for M in "qwen3.5-397b-a17b" "qwen3-235b-a22b-thinking-2507"; do
  AS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions)
  echo "阿里dashscope-us self-check [$M]: $AS_STATUS"; [ "$AS_STATUS" = "200" ] || { echo "FATAL dashscope_not_200 ($AS_STATUS) for model '$M' — key 区域是否美国? id 是否在 /models?"; exit 1; }
done

export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)
export WANDB_RUN_ID="dicode-v6siege-style-s0"
FIXED_OUT=/oscar/scratch/$USER/dicode_outputs/v6siege_style_s0
mkdir -p "$FIXED_OUT"
echo "output dir (resumable, NEW): $FIXED_OUT   wandb run (NEW): $WANDB_RUN_ID"

cd ~/dicode_v6

# --- sanity: fail fast if the sync did not land / loaded the wrong code path ---
python -c "import jax; print('devices:', jax.devices()); assert jax.default_backend()=='gpu'" || { echo FATAL no_gpu; exit 1; }
python -c "import dicode, os; assert os.path.realpath(dicode.__file__).startswith(os.path.realpath(os.path.expanduser('~/dicode_v6'))), dicode.__file__; print('dicode loaded from v6:', os.path.dirname(dicode.__file__))" || { echo FATAL dicode_not_v6; exit 1; }
python -c "import auction; from auction.craftax_achievements import NUM_ACHIEVEMENTS; print('auction import OK', NUM_ACHIEVEMENTS)" || { echo FATAL auction_import; exit 1; }
python -c "from dicode.dreaming.gen_manager import TaskGenerator; assert hasattr(TaskGenerator,'evolve_mastered_coop') and hasattr(TaskGenerator,'_ensure_modeler')" || { echo FATAL coop_method_missing; exit 1; }
python -c "from dicode.dreaming.llm import LLM; import inspect; s=inspect.getsource(LLM._create_client); assert 'deepseek' in s and 'dashscope' in s, 'v6 provider branch missing'; print('v6 providers OK')" || { echo FATAL v6_provider_missing; exit 1; }

# --- v6-SIEGE-specific sanity: siege modules present + config wired ---
python -c "from auction.siege_notebook import SiegeNotebook; from auction.completed_gate import enforce_completed_gate; from auction.cooccurrence_log import CooccurrenceLog; from auction.modeler import Modeler; assert hasattr(Modeler,'diagnose_siege'); from dicode.selection import append_rehearsal_tasks; print('siege modules OK')" || { echo FATAL siege_modules_missing; exit 1; }
python -c "import yaml; c=yaml.safe_load(open('conf/gen_manager/auction_c_v6siege.yaml', encoding='utf-8')); assert c['siege'] is True and c['auction_modeler'] is True and c['auction'] is False; assert c['proposers'][0]['provider']=='deepseek' and c['proposers'][1]['provider']=='dashscope'; print('v6siege config OK')" || { echo FATAL v6siege_config_bad; exit 1; }
echo "v6siege sanity: siege code path + config OK"

python -m pytest auction/tests/ -q 2>&1 | tail -3 || { echo FATAL auction_tests_fail; exit 1; }

# --- LLM role overrides (identical to dicode_v6.sh): 3 non-proposer gen roles -> dashscope-us;
#     embedding STAYS on DeepInfra (dim/dist would break conditioning). ---
OV_LLM="gen_manager/llm@gen_manager.task_generator=dashscope gen_manager/llm@gen_manager.env_generator=dashscope gen_manager/llm@gen_manager.interestingness_critic=dashscope gen_manager/llm@gen_manager.embedding_model=deepinfra_embed"

echo "=== v6-SIEGE: 攻坚课程 (siege) + v5y coop + 省钱 provider ==="
python -u experiments/training/run_dicode.py \
  hydra.run.dir="$FIXED_OUT" \
  gen_manager=auction_c_v6siege \
  $OV_LLM \
  use_wandb=true \
  wandb_project=DiCode-v6siege-style \
  wandb_entity=gregjones11235-brown-university \
  training.load_checkpoint=true \
  seed=0

echo "dicode-v6siege END $(date)"
echo "outputs in: $FIXED_OUT"
