#!/bin/bash
# GPUStack custom-backend launcher. GPUStack renders the backend's run command as
#
#   bash /app/boost/gpustack.sh --model {{model_path}} --port {{port}} \
#        --served-model-name {{model_name}} TP={{gpu_count}} [KEY=VALUE ...] [vllm flags ...]
#
# and appends the deployment's backend parameters. Rules:
#   - KEY=VALUE tokens are HyperQwen knobs (SPEC, CTX, VISION, MODE, MAX_LEN, KV_MEM,
#     DFLASH_MAX_LEN, DFLASH_TOKENS, INT8_ACT, PREFILL_ATTN, GPU_UTIL, MAX_SEQS, ...) and
#     are applied only when the variable is not already set: the deployment's env wins
#     over the backend's defaults in the run command.
#   - anything starting with "-" is passed to vLLM through EXTRA_ARGS (the launcher expands
#     it last, so a repeated flag such as --served-model-name overrides the launcher's own).
#   - TP=<n> (from {{gpu_count}}) adds --tensor-parallel-size <n> unless the parameters
#     already carry one.
# HyperQwen's prepare step is skipped: GPUStack downloads the (already prepared)
# checkpoint and mounts it; only the DFlash2 drafter is fetched if the image lacks it.
set -e
cd /app
export PATH=/app/venv/bin:$PATH
export HOME=${HOME:-/cache}
mkdir -p "$HOME" 2>/dev/null || true

MODEL_ARG=""; PORT_ARG=""; NAMES=(); EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model)   MODEL_ARG=$2; shift 2 ;;
    --model=*) MODEL_ARG=${1#*=}; shift ;;
    --port)    PORT_ARG=$2; shift 2 ;;
    --port=*)  PORT_ARG=${1#*=}; shift ;;
    --served-model-name)
      shift; while [ $# -gt 0 ] && [ "${1#-}" = "$1" ] && [ "${1#*=}" = "$1" ]; do NAMES+=("$1"); shift; done ;;
    --served-model-name=*) NAMES+=("${1#*=}"); shift ;;
    [A-Z_]*=*)
      k=${1%%=*}
      if [ -z "${!k+x}" ]; then export "$1"; fi
      shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
export MODEL=${MODEL_ARG:-${MODEL:?gpustack.sh: --model <path> is required}}
export PORT=${PORT_ARG:-${PORT:-18020}}
export HOST=${HOST:-0.0.0.0}
[ -f "$MODEL/config.json" ] || { echo "gpustack.sh: no config.json under MODEL=$MODEL" >&2; exit 1; }

# GPUStack hands a replica CUDA_VISIBLE_DEVICES with HOST indexes while the container may
# only see the assigned cards (a replica on host GPU 1 alone sees one device, index 0, and
# CUDA_VISIBLE_DEVICES=1 finds nothing). Count what is really visible and remap.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  n=$(env -u CUDA_VISIBLE_DEVICES venv/bin/python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
  want=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  max=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | sort -n | tail -1)
  if [ "$n" -gt 0 ] && [ "${max:-0}" -ge "$n" ]; then
    echo "[gpustack] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES names host indexes but this container sees $n device(s): using $(seq -s, 0 $((n-1)))"
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
    [ "$want" -gt "$n" ] && echo "[gpustack] WARNING: $want cards were assigned, $n are visible"
  fi
fi

# Tensor parallel from GPUStack's gpu_count unless the parameters say otherwise.
TPN=1
for ((i=0; i<${#EXTRA[@]}; i++)); do
  case "${EXTRA[$i]}" in
    --tensor-parallel-size|-tp) TPN=${EXTRA[$((i+1))]} ;;
    --tensor-parallel-size=*|-tp=*) TPN=${EXTRA[$i]#*=} ;;
  esac
done
if [ "$TPN" = 1 ] && [ "${TP:-1}" -gt 1 ]; then
  TPN=$TP; EXTRA+=(--tensor-parallel-size "$TP")
fi

# Profile defaults: DFlash2 (the fastest single-user profile), 64k, vision on. On one card
# the tower streams from host RAM and the DFlash2 pool is pinned a gigabyte under
# HyperQwen's default because these checkpoints carry ~0.6 GiB more weight (int8 heads)
# than the base -fast variant the pin was sized on; the -fast siblings get 0.3 GiB back.
export MODE=${MODE:-single} SPEC=${SPEC:-dflash2} CTX=${CTX:-fast} VISION=${VISION:-1} PREFIX_CACHE=${PREFIX_CACHE:-1}
if [ "$TPN" -gt 1 ]; then
  export VISION_OFFLOAD=${VISION_OFFLOAD:-0}
else
  export VISION_OFFLOAD=${VISION_OFFLOAD:-1}
  if [ "$SPEC" = dflash2 ] && [ "$CTX" = fast ] && [ -z "${KV_MEM+x}" ]; then
    lm_bits=$(venv/bin/python -c "import json,sys; c=json.load(open(sys.argv[1]))['quantization_config']['config_groups']; g=[v for v in c.values() if v.get('targets')==['re:.*lm_head\$']]; print(g[0]['weights']['num_bits'] if g else 16)" "$MODEL/config.json" 2>/dev/null || echo 16)
    if [ "$lm_bits" = 4 ]; then export KV_MEM=4600000000; else export KV_MEM=4300000000; fi
    if [ -z "${DFLASH_MAX_LEN:-}" ]; then
      if [ "${DFLASH_TOKENS:-7}" -gt 7 ]; then
        [ "$lm_bits" = 4 ] && export DFLASH_MAX_LEN=40960 || export DFLASH_MAX_LEN=36864
      else
        export DFLASH_MAX_LEN=49152
      fi
    fi
  fi
fi

# The DFlash2 drafter: baked into the image at /app/models; fetched once if it is not.
if [ "$SPEC" = dflash2 ] && [ -z "${DRAFT:-}" ] && [ ! -f models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors ]; then
  echo "[gpustack] fetching the DFlash2 drafter (syvai/Qwen3.8-27B-DFlash2-W4A16, ~1.2 GB)"
  venv/bin/python prepare/fetch_dflash2.py
fi

[ ${#NAMES[@]} -gt 0 ] && EXTRA=(--served-model-name "${NAMES[@]}" qwen3.8-27b "${EXTRA[@]}")
export EXTRA_ARGS="${EXTRA[*]}"
echo "[gpustack] MODEL=$MODEL PORT=$PORT MODE=$MODE SPEC=$SPEC CTX=$CTX VISION=$VISION VISION_OFFLOAD=$VISION_OFFLOAD TP=$TPN KV_MEM=${KV_MEM-unset} DFLASH_MAX_LEN=${DFLASH_MAX_LEN:-} MAX_LEN=${MAX_LEN:-} INT8_ACT=${INT8_ACT-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} EXTRA_ARGS=$EXTRA_ARGS"
if [ "$MODE" = batch ]; then exec bash batch/start_qwen.sh; else exec bash single-user/start_qwen.sh; fi
