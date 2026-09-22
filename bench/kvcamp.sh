#!/bin/bash
# KV cache type campaign on two RTX 3090 (TP=2), the GPUStack deployment's geometry:
# 262k context where the cache holds it, the KV pool pinned to 7.6e9 bytes per card,
# DFlash2 k=7, vision resident. Per variant: boot, pool size, C1 harness (8 real prompts
# x 1024 tokens, default sampling then greedy), a 120k needle probe (cold TTFT, decode,
# second turn over the cached prefix). Results in results.tsv, one line per variant.
#
#   HQ=<HyperQwen checkout with venv> MODEL=<checkpoint> bash bench/kvcamp.sh [variant ...]
#   SUFFIX=-tag appends a tag to the variant name in results.tsv (before/after comparisons).
set -u
HQ=${HQ:-/root/hq/HyperQwen}; cd "$HQ"
export PATH=/usr/local/cuda-13.0/bin:$PATH HF_HOME=/root/hq/.hf
OUT=${OUT:-$HQ/kvcamp}; mkdir -p $OUT
PORT=18020
MODEL=${MODEL:-$PWD/models/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM-fast}
BENCH="venv/bin/vllm bench serve --host 127.0.0.1 --port $PORT --model $MODEL --served-model-name qwen3.8-27b"
PIN=${PIN:-7600000000}
TSV=$OUT/results.tsv
[ -f $TSV ] || echo -e "variant\tstatus\tboot_s\tbackends\tpool_tokens\tconc_262k\tc1_default_e2e\tc1_default_decode\tttft_default_ms\tc1_greedy_e2e\tc1_greedy_decode\tttft_greedy_ms\tp120k_prompt\tp120k_ttft_s\tp120k_prefill_tok_s\tp120k_decode\tneedle\tturn2_ttft_s\tturn2_cached\tvram_mib" > $TSV

# name | env assignments | extra vllm args (appended last, override the launcher)
variant() {
  case "$1" in
    int8)          ENV="SPEC=dflash2 CTX=long MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN"; EXTRA="" ;;
    tq-4bit-nc)    ENV="SPEC=dflash2 CTX=fast MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN KV_DTYPE=turboquant_4bit_nc"; EXTRA="" ;;
    tq-k8v4)       ENV="SPEC=dflash2 CTX=fast MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN KV_DTYPE=turboquant_k8v4"; EXTRA="" ;;
    bf16)          ENV="SPEC=dflash2 CTX=fast MAX_LEN=131072 DFLASH_MAX_LEN=131072 KV_MEM=$PIN"; EXTRA="" ;;
    fp8-mtp)       ENV="SPEC=mtp CTX=long MAX_LEN=262144 KV_MEM=$PIN"; EXTRA="--kv-cache-memory $PIN" ;;
    int4)          ENV="SPEC=dflash2 CTX=long MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN VLLM_INT4_MQ_3D=1"; EXTRA="--kv-cache-dtype int4_per_token_head" ;;
    kvarn)         ENV="SPEC=dflash2 CTX=huge MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN"; EXTRA="" ;;
    tq-k3v4-nc)    ENV="SPEC=dflash2 CTX=fast MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN KV_DTYPE=turboquant_k3v4_nc"; EXTRA="" ;;
    tq-3bit-nc)    ENV="SPEC=dflash2 CTX=fast MAX_LEN=262144 DFLASH_MAX_LEN=262144 KV_MEM=$PIN KV_DTYPE=turboquant_3bit_nc"; EXTRA="" ;;
    tq-4bit-nc-nospec) ENV="SPEC=off CTX=fast MAX_LEN=262144 KV_MEM=$PIN KV_DTYPE=turboquant_4bit_nc"; EXTRA="--kv-cache-memory $PIN" ;;
    tq-4bit-nc-mtp) ENV="SPEC=mtp CTX=fast MAX_LEN=262144 KV_MEM=$PIN KV_DTYPE=turboquant_4bit_nc"; EXTRA="--kv-cache-memory $PIN" ;;
    *) echo "unknown variant $1"; return 1 ;;
  esac
}

gpu_free() {  # wait until both cards are below 1 GiB used
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$m" -lt 1024 ] && return 0
    sleep 5
  done
  return 1
}

stop_server() {
  [ -n "${SPID:-}" ] && kill -- -"$SPID" 2>/dev/null
  sleep 3
  pkill -f "[v]llm serve" 2>/dev/null; pkill -f "[s]tart_qwen.sh" 2>/dev/null
  gpu_free || echo "WARN: GPUs did not free up"
}

run_variant() {
  variant "$1" || return
  local name=$1${SUFFIX:-}; local log=$OUT/$name.log
  echo "=== $name $(date +%T) ENV=[$ENV] EXTRA=[$EXTRA]" | tee -a $OUT/campaign.log
  gpu_free || { echo "GPUs busy, abort"; return 1; }
  local t0=$(date +%s)
  setsid env CUDA_VISIBLE_DEVICES=0,1 MODEL=$MODEL VISION=1 VISION_OFFLOAD=0 PREFIX_CACHE=1 PORT=$PORT \
      EXTRA_ARGS="--tensor-parallel-size 2 $EXTRA" $ENV bash single-user/start_qwen.sh > $log 2>&1 &
  SPID=$!
  local healthy=0
  for i in $(seq 1 150); do
    curl -sf -o /dev/null http://127.0.0.1:$PORT/health && { healthy=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 10
  done
  local boot=$(( $(date +%s) - t0 ))
  if [ $healthy = 0 ]; then
    echo "--- $name FAILED to boot (${boot}s)" | tee -a $OUT/campaign.log
    grep -iE "error|Traceback|refus|OutOfMemory|not valid|not support|assert" $log | grep -v "min_frames\|max_frames" | tail -6 | cut -c1-240 | tee -a $OUT/campaign.log
    echo -e "$name\tBOOT_FAIL\t$boot" >> $TSV
    stop_server; return
  fi
  local backends=$(grep -oE "Using [A-Z_0-9]+ (attention )?backend" $log | sed 's/Using //; s/ attention backend//; s/ backend//' | sort | uniq -c | awk '{printf "%s:%s ", $2, $1}')
  local pool=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" $log | head -1 | grep -oE "[0-9,]+ tokens" | tr -d ' tokens,')
  local conc=$(grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" $log | head -1 | grep -oE "[0-9.]+x$")
  local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "--- healthy in ${boot}s backends=[$backends] pool=$pool conc=$conc vram=$vram" | tee -a $OUT/campaign.log
  # correctness smoke
  local ans=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Столица Дании? Ответь одним словом."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | venv/bin/python -c 'import json,sys; r=json.load(sys.stdin); print(repr(r["choices"][0]["message"]["content"]))' 2>&1)
  echo "--- smoke: $ans" | tee -a $OUT/campaign.log
  # warmup then C1 default / greedy
  $BENCH --dataset-name custom --dataset-path bench/prompts_real.jsonl --custom-output-len 256 --num-prompts 4 --max-concurrency 2 > /dev/null 2>&1
  local row=""
  for T in default 0; do
    local targ=""; [ "$T" = 0 ] && targ="--temperature 0"
    $BENCH --dataset-name custom --dataset-path bench/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency 1 $targ > $OUT/$name-c1-T$T.log 2>&1
    local tp=$(awk '/Mean TPOT/ {print $4}' $OUT/$name-c1-T$T.log); local e2e=$(awk '/Output token throughput/ {print $5}' $OUT/$name-c1-T$T.log); local ttft=$(awk '/Mean TTFT/ {print $4}' $OUT/$name-c1-T$T.log)
    local dec=$(python3 -c "print(f'{1000/${tp:-1e9}:.1f}')")
    echo "--- C1 T=$T e2e=$e2e decode=$dec ttft=$ttft" | tee -a $OUT/campaign.log
    row="$row\t${e2e:-na}\t${dec:-na}\t${ttft:-na}"
  done
  # 120k needle probe
  python3 ${PROBE:-$(dirname "$0")/../boost/long_ctx_probe.py} --base http://127.0.0.1:$PORT --model qwen3.8-27b --tokens 120000 --turn2 --timeout 3600 > $OUT/$name-p120k.log 2>&1
  cat $OUT/$name-p120k.log | tee -a $OUT/campaign.log
  local p=$(grep "^turn 1:" $OUT/$name-p120k.log)
  local pp=$(echo "$p" | grep -oE "prompt=[0-9]+" | cut -d= -f2); local pt=$(echo "$p" | grep -oE "TTFT=[0-9.]+" | cut -d= -f2); local pr=$(echo "$p" | grep -oE "prefill [0-9]+" | awk '{print $2}'); local pd=$(echo "$p" | grep -oE "decode=[0-9.]+" | cut -d= -f2); local nd=$(echo "$p" | grep -oE "needle=[A-Z]+" | cut -d= -f2)
  local q=$(grep "^turn 2:" $OUT/$name-p120k.log); local qt=$(echo "$q" | grep -oE "TTFT=[0-9.]+" | cut -d= -f2); local qc=$(echo "$q" | grep -oE "cached=[0-9]+" | cut -d= -f2)
  grep -iE "OutOfMemory|EngineDead|Traceback" $log | tail -2 | cut -c1-200 | tee -a $OUT/campaign.log
  echo -e "$name\tOK\t$boot\t$backends\t${pool:-na}\t${conc:-na}$row\t${pp:-na}\t${pt:-na}\t${pr:-na}\t${pd:-na}\t${nd:-na}\t${qt:-na}\t${qc:-na}\t$vram" >> $TSV
  stop_server
}

ALL="int8 tq-4bit-nc tq-k8v4 bf16 fp8-mtp int4 kvarn tq-k3v4-nc tq-3bit-nc tq-4bit-nc-nospec"
for v in ${@:-$ALL}; do run_variant $v; done
echo "CAMPAIGN DONE $(date +%T)" | tee -a $OUT/campaign.log
