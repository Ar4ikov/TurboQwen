# TurboQwen (formerly vllm-hyprfastQwen, vllm-qwen-boost): HyperQwen (vLLM 0.33.0 + its patch
# series + KVarN, with the marlin-int8-asym-zp patch that is now part of that series) plus
# the TurboQuant spec-as-decode patch, packaged for the Ar4ikov Qwen3.8-27B AWQ-W4A16-ASYM
# checkpoints with the vision tower on by default.
#
# Same recipe as HyperQwen's own Dockerfile, step for step: Python 3.12 venv at /app/venv,
# nvcc for FlashInfer's JIT, the pth-flash-attn and fp8-KV FA2 op libraries, the patched
# xgrammar, the LSA plugin, every patch in patches/series applied at --fuzz 0 by
# patches/apply.sh, KVarN installed, verify.sh --install and the CPU tests at build time.
# The HyperQwen tree comes from the git submodule hyperqwen/ (Ar4ikov/HyperQwen, branch
# awq-asym-0.33 = syv-ai/HyperQwen main + the KV_DTYPE hook + turboquant-spec-as-decode).
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

# The op libraries and plugins compile ahead of the full tree, as in HyperQwen's Dockerfile,
# so an edit elsewhere does not recompile them.
COPY hyperqwen/pthfa/ pthfa/
RUN PY=venv/bin/python bash pthfa/install.sh
COPY hyperqwen/fa2/ fa2/
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && FA2_CACHE=/tmp/fa2-cache bash fa2/install.sh && rm -rf /tmp/fa2-cache
COPY hyperqwen/xgrammar/ xgrammar/
# xgrammar builds from source in pip's isolated build environment, which resolves
# apache-tvm-ffi on its own: 0.1.16 (2026-11-11) compiles the bindings as C++20 while the
# rest of xgrammar at this pin is C++17, and the LTO link fails on -Werror=odr. Build
# against the tvm-ffi the venv already has (what vLLM's own pins resolved), which is also
# the one the bindings load at run time.
# pip 25.3+ reads build-environment constraints from PIP_BUILD_CONSTRAINT; PIP_CONSTRAINT
# is kept for an older pip.
RUN venv/bin/pip freeze | grep -i '^apache-tvm-ffi==' > /tmp/tvm-ffi.constraint \
    && cat /tmp/tvm-ffi.constraint \
    && PIP_BUILD_CONSTRAINT=/tmp/tvm-ffi.constraint PIP_CONSTRAINT=/tmp/tvm-ffi.constraint bash xgrammar/install.sh
COPY hyperqwen/lsa/ lsa/
RUN PY=venv/bin/python bash lsa/install.sh --no-map

COPY hyperqwen/ /app/
# patches/apply.sh reads the apply order from patches/series and applies it at --fuzz 0.
# The tests after verify.sh are HyperQwen's own build-time gates (CPU tensors, no model).
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    bash patches/apply.sh "$SP"; \
    bash kvarn/install.sh; \
    bash verify.sh --install; \
    venv/bin/python bench/test_draft_vocab_stats.py; \
    venv/bin/python bench/test_gdn_checkpoint_safety.py; \
    venv/bin/python bench/test_gdn_multi_checkpoint.py; \
    venv/bin/python bench/test_gdn_checkpoint_export.py; \
    venv/bin/python bench/test_offload_mamba_swa_boundaries.py; \
    venv/bin/python bench/test_dflash_context_capacity.py; \
    venv/bin/python bench/mq3d_scratch_pool_test.py; \
    venv/bin/python bench/test_quant_scale_views.py; \
    CUDA_VISIBLE_DEVICES= venv/bin/python pthfa/tests/gate_test.py; \
    venv/bin/python fa2/tests/gate_test.py; \
    venv/bin/python fa2/tests/check_mixed_prefill.py --device cpu; \
    CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_async_mamba_kv_sizing.py; \
    CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_kv_cache_seat_budget.py; \
    CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_continuous_stream_usage.py; \
    venv/bin/python bench/test_gdn_ba_lora.py; \
    CUDA_VISIBLE_DEVICES= venv/bin/python lsa/tests/gate_test.py

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
