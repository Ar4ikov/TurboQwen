#!/bin/bash
# GPUStack custom-backend launcher. GPUStack renders the backend's run command as
#
#   bash /app/boost/gpustack.sh --model {{model_path}} --port {{port}} \
#        --served-model-name {{model_name}} TP={{gpu_count}} [KEY=VALUE ...]
#
# and appends the deployment's backend parameters, which are plain vllm serve flags.
#
# HyperQwen's launcher (single-user/start_qwen.sh) decides a few of those flags itself
# from its knobs -- the attention backend and KV dtype (CTX), the speculative config
# (SPEC), the pinned KV pool (KV_MEM), the context (MAX_LEN), the seat count (MAX_SEQS).
# So the vllm flags that overlap are translated into the knobs here, and the launcher
# emits them consistently; everything else is passed through EXTRA_ARGS, which the
# launcher expands last, so a repeated flag overrides its own. What is translated:
#
#   --max-model-len N              -> MAX_LEN / DFLASH_MAX_LEN
#   --kv-cache-memory B            -> KV_MEM (per GPU, bytes; also --kv-cache-memory-bytes)
#   --kv-cache-dtype bfloat16|auto -> CTX=fast   (FLASH_ATTN, bf16 KV)
#                    int8_per_token_head -> CTX=long with SPEC=dflash2 (TRITON_ATTN int8 KV,
#                                   the split-KV verify kernel of the patch series)
#                    fp8            -> CTX=long with SPEC=mtp (FlashInfer fp8 KV; DFlash2's
#                                   fp8 verify kernel needs sm89+, so on Ampere fp8 means MTP)
#                    kvarn_k4v2_g128 -> CTX=huge (KVarN)
#                    int4_per_token_head -> CTX=long + the flag itself (TRITON_ATTN int4 KV with
#                                   DFlash2: HyperQwen's experimental 256k route,
#                                   single-user/alternative.sh)
#                    turboquant_*   -> CTX=fast with KV_DTYPE=<dtype> (the launcher then emits
#                                   the dtype without an explicit attention backend, so vLLM's
#                                   TURBOQUANT backend serves the quantized layers and
#                                   FlashAttention the drafter's sliding-window layers)
#   --max-num-seqs N               -> MAX_SEQS
#   --gpu-memory-utilization X     -> GPU_UTIL
#   --[no-]enable-prefix-caching   -> PREFIX_CACHE
#   --tensor-parallel-size N       -> kept, and TP={{gpu_count}} adds it when absent
#
# With the tower on, a prompt may carry up to IMAGES_PER_PROMPT images (default 10; HyperQwen's
# own default is 1); an explicit --limit-mm-per-prompt in the parameters wins.
#
# KEY=VALUE tokens are HyperQwen knobs (SPEC=dflash2|mtp|off, VISION=1, MODE=batch,
# INT8_ACT=int8, PREFILL_ATTN=int8, DFLASH_TOKENS, REASONING_EFFORT, ...); a token in the
# run command applies only when the variable is not already set, so the deployment's env
# wins over the backend's defaults. HyperQwen's prepare step is skipped: GPUStack
# downloads the (already prepared) checkpoint; the DFlash2 drafter ships in the image.
#
# GPUSTACK_DRY_RUN=1 prints the resolved knobs and vllm flags (the "[gpustack] ..." line)
# and exits before the launcher; boost/test_gpustack_sh.sh runs the translation table
# through it at image build time.
set -e
cd "${APP_DIR:-/app}"
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

# vllm flags the launcher also decides: translate into its knobs (see the header).
PASS=(); i=0
while [ $i -lt ${#EXTRA[@]} ]; do
  a=${EXTRA[$i]}; v=""; adv=1; key=""
  case "$a" in
    --*=*) key=${a%%=*}; v=${a#*=} ;;
    --*)   key=$a; nxt=${EXTRA[$((i+1))]:-}
           if [ -n "$nxt" ] && [ "${nxt#-}" = "$nxt" ]; then v=$nxt; adv=2; fi ;;
  esac
  case "$key" in
    --max-model-len)        export MAX_LEN=$v DFLASH_MAX_LEN=$v ;;
    --kv-cache-memory|--kv-cache-memory-bytes) export KV_MEM=$v ;;
    --max-num-seqs)         export MAX_SEQS=$v ;;
    --gpu-memory-utilization) export GPU_UTIL=$v ;;
    --enable-prefix-caching)    export PREFIX_CACHE=1 ;;
    --no-enable-prefix-caching) export PREFIX_CACHE=0 ;;
    --kv-cache-dtype)
      case "$v" in
        auto|bfloat16|bf16) export CTX=fast ;;
        int8_per_token_head|int8) export CTX=long; export SPEC=${SPEC:-dflash2} ;;
        fp8|fp8_e4m3|fp8_e5m2)
          export CTX=long
          if [ "${SPEC:-dflash2}" = dflash2 ]; then
            echo "[gpustack] --kv-cache-dtype $v: DFlash2's fp8 verify kernel needs sm89+, serving fp8 KV with SPEC=mtp (FlashInfer)"
            export SPEC=mtp
          fi ;;
        kvarn*) export CTX=huge ;;
        int4_per_token_head)
          # HyperQwen's experimental 256k route (single-user/alternative.sh): the int4
          # per-token-head cache on TRITON_ATTN with the DFlash2 drafter
          # (patches/int4-kv-per-token-head.patch, spec-decode-int4-kv-mq3d.patch). CTX=long
          # sets the backend and the int8 dtype; the flag, passed through last, overrides
          # the dtype. About 20% slower decode than int8 on short prompts, 2x the pool.
          echo "[gpustack] --kv-cache-dtype int4_per_token_head: HyperQwen's experimental route (TRITON_ATTN + DFlash2, docs/long-context.md); int8_per_token_head is the verified one"
          export CTX=long; export SPEC=${SPEC:-dflash2}; export VLLM_INT4_MQ_3D=${INT4_MQ_3D:-1}
          PASS+=("$a"); [ $adv = 2 ] && PASS+=("$v") ;;
        turboquant*)
          # TurboQuant (vLLM's own backend: Hadamard rotation + Lloyd-Max keys, uniform
          # values). The launcher's CTX=fast profile with KV_DTYPE set: no explicit
          # attention backend, so vLLM picks TURBOQUANT for the quantized layers and
          # FlashAttention for the drafter's sliding-window layers (kept in bf16 through
          # --kv-cache-dtype-skip-layers sliding_window; TurboQuant has no window mask).
          # The image carries patches/turboquant-spec-as-decode.patch, without which the
          # DFlash2/MTP verify block runs through the backend's prefill path.
          echo "[gpustack] --kv-cache-dtype $v: TurboQuant cache, CTX=fast profile with KV_DTYPE=$v (see README, KV cache types)"
          export CTX=fast; export SPEC=${SPEC:-dflash2}; export KV_DTYPE=$v ;;
        *) PASS+=("$a"); [ $adv = 2 ] && PASS+=("$v") ;;
      esac ;;
    *) PASS+=("$a"); [ $adv = 2 ] && PASS+=("$v") ;;
  esac
  i=$((i + adv))
done
EXTRA=("${PASS[@]}")

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
# the tower streams from host RAM and, unless --kv-cache-memory was given, the DFlash2 pool
# is pinned a gigabyte under HyperQwen's default because these checkpoints carry ~0.6 GiB
# more weight (int8 heads) than the base -fast variant the pin was sized on.
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
if [ "$SPEC" = dflash2 ] && [ "${GPUSTACK_DRY_RUN:-0}" != 1 ] && [ -z "${DRAFT:-}" ] && [ ! -f models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors ]; then
  echo "[gpustack] fetching the DFlash2 drafter (syvai/Qwen3.8-27B-DFlash2-W4A16, ~1.2 GB)"
  venv/bin/python prepare/fetch_dflash2.py
fi

# REASONING_EFFORT=medium|low|xhigh: the server-side default for the chat template's
# reasoning_effort (the same thing as passing --default-chat-template-kwargs in the
# parameters). In Qwen3.8's template xhigh and low add a system instruction; medium is the
# bare prompt; unset means xhigh. Per request, the OpenAI-style reasoning_effort field or
# chat_template_kwargs override it. No spaces in the JSON: EXTRA_ARGS is word-split.
if [ -n "${REASONING_EFFORT:-}" ]; then
  EXTRA+=(--default-chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}")
fi
# Images per prompt. HyperQwen's VISION=1 emits --limit-mm-per-prompt with a count of 1;
# EXTRA_ARGS is expanded after it, so this one wins. The per-image pixel cap (2048 image
# tokens) stays HyperQwen's. vLLM's encoder budget is max(--max-num-batched-tokens, the
# largest single image), not a multiple of the count, so a higher count should not shrink
# the KV pool; not yet measured on a booted server.
if [ "$VISION" = 1 ] && [[ " ${EXTRA[*]} " != *" --limit-mm-per-prompt"* ]]; then
  EXTRA+=(--limit-mm-per-prompt "{\"image\":{\"count\":${IMAGES_PER_PROMPT:-10}}}")
fi
# The launcher only emits --kv-cache-memory on its DFlash2 branch (MTP sizes the pool from
# GPU_UTIL); a pin asked for here is a pin on every branch, so it is re-emitted last.
if [ -n "${KV_MEM:-}" ]; then
  EXTRA+=(--kv-cache-memory "$KV_MEM")
fi
[ ${#NAMES[@]} -gt 0 ] && EXTRA=(--served-model-name "${NAMES[@]}" qwen3.8-27b "${EXTRA[@]}")
export EXTRA_ARGS="${EXTRA[*]}"
echo "[gpustack] MODEL=$MODEL PORT=$PORT MODE=$MODE SPEC=$SPEC CTX=$CTX KV_DTYPE=${KV_DTYPE:-} VISION=$VISION VISION_OFFLOAD=$VISION_OFFLOAD TP=$TPN KV_MEM=${KV_MEM-unset} MAX_LEN=${MAX_LEN:-} DFLASH_MAX_LEN=${DFLASH_MAX_LEN:-} MAX_SEQS=${MAX_SEQS:-} GPU_UTIL=${GPU_UTIL:-} INT8_ACT=${INT8_ACT-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} EXTRA_ARGS=$EXTRA_ARGS"
if [ "${GPUSTACK_DRY_RUN:-0}" = 1 ]; then exit 0; fi
if [ "$MODE" = batch ]; then exec bash batch/start_qwen.sh; else exec bash single-user/start_qwen.sh; fi
