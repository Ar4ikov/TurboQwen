#!/bin/bash
# Dry-run checks of boost/gpustack.sh's flag translation. GPUSTACK_DRY_RUN=1 makes the
# wrapper print its "[gpustack] ..." line (the resolved HyperQwen knobs and the vllm
# flags it passes on) and exit before the launcher, so this needs no GPU and no model:
# it runs in the image at build time (Dockerfile) and on any checkout with
# APP_DIR=<checkout> bash boost/test_gpustack_sh.sh.
set -u
APP=${APP_DIR:-/app}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/model"
echo '{"architectures":["Qwen3_5ForConditionalGeneration"]}' > "$TMP/model/config.json"
fail=0; n=0

run() {  # run [VAR=value ...] -- <run-command tokens and backend parameters>
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env -i PATH="$PATH" HOME="$TMP" GPUSTACK_DRY_RUN=1 APP_DIR="$APP" "${envs[@]}" \
    bash "$APP/boost/gpustack.sh" --model "$TMP/model" --port 1 --served-model-name m "$@" 2>&1 \
    | grep '^\[gpustack\] MODEL='
}
check() {  # check <name> "<line>" <needle>... ; a needle starting with ! must be absent
  local name=$1 line=$2 p; shift 2; n=$((n + 1))
  [ -n "$line" ] || { echo "FAIL $name: no [gpustack] line"; fail=1; return; }
  for p in "$@"; do
    if [ "${p#!}" != "$p" ]; then
      grep -qF -- "${p#!}" <<<"$line" && { echo "FAIL $name: found '${p#!}'"; echo "   $line"; fail=1; }
    else
      grep -qF -- "$p" <<<"$line" || { echo "FAIL $name: missing '$p'"; echo "   $line"; fail=1; }
    fi
  done
}

# the two-card deployment: every overlapping flag becomes a knob, nothing is emitted twice
L=$(run -- TP=2 SPEC=dflash2 CTX=fast VISION=1 PREFIX_CACHE=1 --tensor-parallel-size=2 \
      --max-model-len=262144 --kv-cache-memory=7600000000 --kv-cache-dtype=int8_per_token_head \
      --max-num-seqs=8 --max-num-batched-tokens=2048 --enable-prefix-caching --reasoning-parser=qwen3)
check int8-tp2 "$L" "SPEC=dflash2" "CTX=long" "TP=2" "KV_MEM=7600000000" "MAX_LEN=262144" \
      "DFLASH_MAX_LEN=262144" "MAX_SEQS=8" "VISION_OFFLOAD=0" "--kv-cache-memory 7600000000" \
      "--max-num-batched-tokens=2048" "--reasoning-parser=qwen3" \
      "!--kv-cache-dtype" "!--max-model-len" "!--max-num-seqs" "!--enable-prefix-caching"

# one card, the defaults: DFlash2 + bf16 pin sized for the int8-head checkpoint (no lm_head group)
L=$(run -- TP=1 SPEC=dflash2 CTX=fast VISION=1)
check single-default "$L" "SPEC=dflash2" "CTX=fast" "TP=1" "VISION_OFFLOAD=1" \
      "KV_MEM=4300000000" "DFLASH_MAX_LEN=49152" "--kv-cache-memory 4300000000" \
      "EXTRA_ARGS=--served-model-name m qwen3.8-27b"

# space-separated flags translate like --flag=value
L=$(run -- TP=1 --kv-cache-memory 123 --max-model-len 4096 --max-num-seqs 3 --kv-cache-dtype bfloat16)
check space-form "$L" "KV_MEM=123" "MAX_LEN=4096" "DFLASH_MAX_LEN=4096" "MAX_SEQS=3" "CTX=fast" \
      "--kv-cache-memory 123" "!--kv-cache-dtype"

# fp8 on Ampere means MTP (FlashInfer)
L=$(run -- TP=1 --kv-cache-dtype fp8)
check fp8 "$L" "CTX=long" "SPEC=mtp" "!--kv-cache-dtype"

# TurboQuant is redirected to int8 unless explicitly allowed
L=$(run -- TP=2 --kv-cache-dtype=turboquant_4bit_nc --max-model-len=262144)
check turboquant "$L" "CTX=long" "SPEC=dflash2" "MAX_LEN=262144" "!turboquant"
L=$(run ALLOW_TURBOQUANT=1 -- TP=2 --kv-cache-dtype=turboquant_k3v4_nc)
check turboquant-allowed "$L" "CTX=fast" "--kv-cache-dtype=turboquant_k3v4_nc"

# int4 per-token-head: the experimental route, flag kept so it overrides CTX=long's int8
L=$(run -- TP=1 --kv-cache-dtype int4_per_token_head)
check int4 "$L" "CTX=long" "SPEC=dflash2" "--kv-cache-dtype int4_per_token_head"

# KVarN
L=$(run -- TP=1 --kv-cache-dtype=kvarn_k4v2_g128)
check kvarn "$L" "CTX=huge" "!--kv-cache-dtype"

# a deployment's env beats the run command's KEY=VALUE tokens
L=$(run SPEC=mtp CTX=long -- TP=1 SPEC=dflash2 CTX=fast)
check env-wins "$L" "SPEC=mtp" "CTX=long"

# REASONING_EFFORT becomes the template default
L=$(run -- TP=1 REASONING_EFFORT=low)
check reasoning "$L" '--default-chat-template-kwargs {"reasoning_effort":"low"}'

# --no-enable-prefix-caching and --gpu-memory-utilization
L=$(run -- TP=1 --no-enable-prefix-caching --gpu-memory-utilization 0.9 --kv-cache-memory=)
check prefix-off "$L" "GPU_UTIL=0.9" "!--gpu-memory-utilization" "!--no-enable-prefix-caching" "!--kv-cache-memory"

# batch mode with TP: the tower stays resident, TP is added from the GPU count
L=$(run -- TP=2 MODE=batch INT8_ACT=int8)
check batch-tp2 "$L" "MODE=batch" "INT8_ACT=int8" "TP=2" "--tensor-parallel-size 2"

if [ "$fail" = 0 ]; then echo "test_gpustack_sh: $n cases ok"; else echo "test_gpustack_sh: FAILED"; exit 1; fi
