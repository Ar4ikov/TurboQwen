# TurboQwen (formerly vllm-hyprfastQwen, vllm-qwen-boost): HyperQwen (vLLM 0.29.0 + its patch
# series + KVarN) plus the marlin-int8-asym-zp patch, packaged for the Ar4ikov
# Qwen3.8-27B AWQ-W4A16-ASYM checkpoints with the vision tower on by default.
#
# Same recipe as HyperQwen's own Dockerfile: Python 3.12 venv at /app/venv, nvcc for
# FlashInfer's JIT, every patch in patches/series applied at --fuzz 0, KVarN installed,
# verify.sh --install at build time. The HyperQwen tree comes from the git submodule
# hyperqwen/ (Ar4ikov/HyperQwen, branch awq-asym = upstream PR #148 + the asym patch).
#
#   docker compose --profile single up -d      (see README.md)
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.12 -m venv venv && venv/bin/pip install --upgrade pip
COPY hyperqwen/docker/requirements.txt docker/requirements.txt
RUN venv/bin/pip install -r docker/requirements.txt

COPY hyperqwen/ /app/
# Apply order lives in patches/series: a few patches carry hunk context an earlier
# patch adds, so the glob order of the directory would be wrong.
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' patches/series | \
    while IFS= read -r name; do \
      case "$name" in \
        dflash2-backport.patch) echo "== skip $name (DFlash2 is native since vLLM 0.28.0)"; continue ;; \
      esac; \
      echo "== $name"; patch -p1 --fuzz 0 --no-backup-if-mismatch -d "$SP" < "patches/$name"; \
    done; \
    bash kvarn/install.sh; \
    bash verify.sh --install

# The W4A16 DFlash2 drafter (syvai, 1.2 GB) ships in the image, so a container with no
# models volume (GPUStack, a bare docker run) can start SPEC=dflash2 without a download.
# A ./models bind mount hides it; HyperQwen's prepare then fetches it into the mount once.
RUN HF_XET_HIGH_PERFORMANCE=1 venv/bin/python prepare/fetch_dflash2.py

COPY boost/ /app/boost/
RUN chmod +x /app/boost/*.sh
# The GPUStack wrapper's flag translation, dry-run (no GPU, no model).
RUN bash /app/boost/test_gpustack_sh.sh

# HOME is a volume: torch.compile cache, Triton, FlashInfer JIT, HF hub cache.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_XET_HIGH_PERFORMANCE=1 \
    VISION=1 FAST_VARIANT=0
VOLUME ["/cache", "/app/models"]
EXPOSE 18020
ENTRYPOINT ["bash", "boost/entrypoint.sh"]
CMD ["single"]
