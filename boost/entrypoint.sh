#!/bin/bash
# TurboQwen entrypoint: pick the checkpoint, keep the vision tower, then hand
# over to HyperQwen's own entrypoint (prepare -> verify -> serve).
#
#   CHECKPOINT=uncensored (default) | base | any HF repo id in the prepared layout
#   VISION=1 (default here; HyperQwen's own default is 0 = --language-model-only)
#   IMAGES_PER_PROMPT=10 (default here; HyperQwen's VISION=1 allows 1). An explicit
#     --limit-mm-per-prompt in EXTRA_ARGS wins.
#
# The prepared checkpoints on the Hub already carry the int8 lm_head/embed_tokens, the
# int8 MTP module and the 40k draft head, so prepare only downloads them (plus the
# DFlash2 drafter unless DFLASH2=0). FAST_VARIANT=0: HyperQwen's -fast variant is the
# base model's int4 heads and does not apply to these checkpoints.
set -e
cd /app
case "${CHECKPOINT:-uncensored}" in
  uncensored) HF_REPO=${HF_REPO:-Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM-HyperQwen} ;;
  base)       HF_REPO=${HF_REPO:-Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM-HyperQwen} ;;
  */*)        HF_REPO=${HF_REPO:-$CHECKPOINT} ;;
  *) echo "entrypoint: CHECKPOINT=$CHECKPOINT is not uncensored|base|<hf repo id>" >&2; exit 1 ;;
esac
export HF_REPO
export BASE_MODEL_DIR=${BASE_MODEL_DIR:-/app/models/$(basename "$HF_REPO")}
export MODEL=${MODEL:-$BASE_MODEL_DIR}
export VISION=${VISION:-1}
export FAST_VARIANT=${FAST_VARIANT:-0}
# EXTRA_ARGS is expanded after HyperQwen's own --limit-mm-per-prompt, so this one wins.
if [ "$VISION" = 1 ] && [[ " ${EXTRA_ARGS:-} " != *" --limit-mm-per-prompt"* ]]; then
  export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--limit-mm-per-prompt {\"image\":{\"count\":${IMAGES_PER_PROMPT:-10}}}"
fi
echo "[boost] checkpoint=$HF_REPO model=$MODEL vision=$VISION images/prompt=${IMAGES_PER_PROMPT:-10}"
exec bash docker/entrypoint.sh "$@"
