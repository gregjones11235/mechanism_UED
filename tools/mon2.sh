#!/usr/bin/env bash
# mon2.sh - read-only status board for the two-card / two-arm setup.
# Replaces mon.sh (single arm, single card, LOG= hand-edited every run).
# Strictly read-only: nothing here writes, kills or touches training state.

# ------------------------------------------------------------------ config
ARMS=(longStack longSlots9)

declare -A DIR=(  [longStack]=/root/outputs/longStack   [longSlots9]=/root/outputs/longSlots9 )
declare -A LOG=(  [longStack]=/root/run_longStack.log   [longSlots9]=/root/run_longSlots9.log )
declare -A GPU=(  [longStack]=0                         [longSlots9]=1 )
declare -A PORT=( [longStack]=11434                     [longSlots9]=11435 )
declare -A OLOG=( [longStack]=/workspace/ollama_0.log   [longSlots9]=/workspace/ollama_1.log )
declare -A WBID=( [longStack]=8tlg55ky                  [longSlots9]=nz304941 )

WANDB_ROOT=/root/wandb
VOLUME=/workspace
STALL_MIN=40        # checkpoint staleness alarm (a session is ~20-25 min, orbax lags ~20)
TG_MIN=10           # t/s floor; healthy has been 33-66
UTIL_SAMPLES=3      # consecutive 0% samples before calling a card idle

# ----------------------------------------------------------------- helpers
ALARMS=()
add(){ ALARMS+=("$1"); }
now=$(date +%s)

age_min(){  # age in minutes of a path, "" if missing
  [ -e "$1" ] || { echo ""; return; }
  local t; t=$(stat -c %Y "$1" 2>/dev/null) || { echo ""; return; }
  echo $(( (now - t) / 60 ))
}
newest_age_min(){  # age of the newest file under a dir
  [ -d "$1" ] || { echo ""; return; }
  local t; t=$(find "$1" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  [ -n "$t" ] || { echo ""; return; }
  echo $(( (now - t) / 60 ))
}
last_ckpt(){ ls -1 "$1/rl_checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1; }

# --------------------------------------------------- sample GPU utilisation
declare -A UTIL_HIST MEM_NOW
for _i in $(seq 1 "$UTIL_SAMPLES"); do
  while IFS=, read -r idx mem util; do
    idx=${idx// /}; mem=${mem// /}; util=${util// /}
    UTIL_HIST[$idx]="${UTIL_HIST[$idx]} $util"
    MEM_NOW[$idx]=$mem
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null)
  [ "$_i" -lt "$UTIL_SAMPLES" ] && sleep 2
done

echo "=============================================================="
echo " mon2  $(date -u '+%F %H:%M:%S') UTC   ( Toronto = UTC-4 )"
echo "=============================================================="

# ------------------------------------------------------------- per-arm view
for a in "${ARMS[@]}"; do
  d=${DIR[$a]}; l=${LOG[$a]}; g=${GPU[$a]}; ol=${OLOG[$a]}
  echo
  echo "--- $a   (card $g, ollama :${PORT[$a]}, wandb ${WBID[$a]})"

  # liveness
  pid=$(pgrep -f "run_dicode.py.*hydra.run.dir=$d" 2>/dev/null | tail -1)
  if [ -n "$pid" ]; then echo "  pid          : $pid alive"
  else echo "  pid          : NOT RUNNING"; add "$a: no run_dicode process"; fi

  # progress
  ses=$(grep -aoE '\-\-\- Starting Session [0-9]+' "$l" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  sdur=$(grep -aoE 'Session finished in [0-9.]+' "$l" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
  ck=$(last_ckpt "$d"); ckage=$(age_min "$d/rl_checkpoints/$ck")
  logage=$(age_min "$l")
  echo "  session      : ${ses:--}      last took ${sdur:--}s      log written ${logage:--} min ago"
  echo "  last ckpt    : ${ck:--}      written ${ckage:--} min ago"
  if [ -n "$ckage" ] && [ "$ckage" -gt "$STALL_MIN" ]; then
    add "$a: newest checkpoint is ${ckage}min old (>$STALL_MIN) - stalled, or orbax stuck in fsync"
  fi

  # wandb freshness  (process alive != training healthy)
  wb=$(find "$WANDB_ROOT" -maxdepth 3 -type d -name "run-*${WBID[$a]}*" 2>/dev/null | head -1)
  wbage=$(newest_age_min "$wb")
  echo "  wandb local  : ${wbage:--} min since last write   ${wb:-(not found)}"
  if [ -z "$wb" ]; then add "$a: no local wandb run dir for ${WBID[$a]}"
  elif [ -n "$wbage" ] && [ "$wbage" -gt "$STALL_MIN" ]; then
    add "$a: no wandb metric written for ${wbage}min - check the ppo_tr.py:265 log callback"
  fi

  # generation funnel
  echo "  gate         : $(grep -a 'ScaffoldGate' "$l" 2>/dev/null | tail -1 | sed 's/^ *WORKER: *//' | tr -s ' ')"
  echo "  preflight    : $(grep -a '\[Preflight\] kept' "$l" 2>/dev/null | tail -1 | tr -s ' ')"
  echo "  frontier     : $(grep -a 'SkillGraph' "$l" 2>/dev/null | tail -1 | sed 's/.*targets: //; s/ (one-step.*//')"

  # errors
  tb=$(grep -ac 'Traceback' "$l" 2>/dev/null)
  rt=$(grep -ac 'Retrying request' "$l" 2>/dev/null)
  gen=$(grep -a -A4 'Traceback' "$l" 2>/dev/null | grep -ac 'check_compilation\|_validate_on_cpu_impl')
  real=$(grep -a -A6 'Traceback' "$l" 2>/dev/null | grep -ac 'run_dicode\.py\|ppo_tr\.py\|dicode/setup\.py')
  echo "  errors       : Traceback=${tb:-0} (gen-code=${gen:-0}, trainer=${real:-0})  Retrying=${rt:-0}"
  [ "${real:-0}" -gt 0 ] && add "$a: ${real} traceback frame(s) inside the TRAINER (run_dicode/ppo_tr/setup) - not generated-code compile noise"

  # ollama throughput for this arm
  tg=$(grep -ao 'tg = *[0-9.]* t/s' "$ol" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
  echo "  ollama tg    : ${tg:--} t/s"
  if [ -n "$tg" ] && awk -v v="$tg" -v m="$TG_MIN" 'BEGIN{exit !(v<m)}'; then
    add "$a: ollama tg ${tg} t/s is below ${TG_MIN} (healthy 33-66)"
  fi

  # card utilisation
  echo "  card $g       : ${MEM_NOW[$g]:--} MiB used, util samples:${UTIL_HIST[$g]:- -}"
  zero=1; for u in ${UTIL_HIST[$g]}; do [ "$u" != "0" ] && zero=0; done
  [ -n "${UTIL_HIST[$g]}" ] && [ "$zero" = "1" ] && \
    add "$a: card $g at 0% across $UTIL_SAMPLES samples - arm may be blocked"
done

# --------------------------------------------------------------- global view
echo
echo "--- gpus"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null | sed 's/^/  /'

echo
echo "--- ollama"
for a in "${ARMS[@]}"; do
  p=${PORT[$a]}
  out=$(OLLAMA_HOST=127.0.0.1:$p timeout 10 ollama ps 2>&1)
  echo "  :$p"
  echo "$out" | sed 's/^/    /'
  body=$(echo "$out" | tail -n +2 | grep -v '^[[:space:]]*$')
  if [ -z "$body" ]; then
    add "ollama :$p has no model loaded"
  elif echo "$body" | grep -q 'CPU'; then
    add "ollama :$p PROCESSOR shows CPU - layers offloaded off the GPU"
  elif ! echo "$body" | grep -q '100% GPU'; then
    add "ollama :$p PROCESSOR is not 100% GPU"
  fi
done

echo
echo "--- volume"
timeout 5 ls "$VOLUME" >/dev/null 2>&1; rc=$?
echo "  timeout 5 ls $VOLUME -> rc=$rc"
[ "$rc" = "124" ] && add "MooseFS mount $VOLUME is hung (rc=124) - fsync ~1000x slower than local; kill -9 will NOT reach D-state tasks"

# -------------------------------------------------------------------- alarms
echo
echo "=============================================================="
if [ ${#ALARMS[@]} -eq 0 ]; then
  echo " OK  no alarms"
else
  echo " ALARMS (${#ALARMS[@]})"
  for m in "${ALARMS[@]}"; do echo "  !! $m"; done
fi
echo "=============================================================="
