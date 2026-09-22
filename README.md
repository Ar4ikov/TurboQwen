<div align="center">

<h1>vllm-qwen-boost</h1>

<p><b>Qwen3.8-27B AWQ-W4A16-ASYM on the GPUs people actually own.<br>
HyperQwen's speed stack on vLLM 0.29, the int8 Marlin path unlocked for zero-point weights, and the vision tower kept on.</b></p>

<p>
<a href="https://github.com/Ar4ikov/vllm-qwen-boost/actions/workflows/image.yml"><img alt="image" src="https://img.shields.io/github/actions/workflow/status/Ar4ikov/vllm-qwen-boost/image.yml?branch=main&label=image&labelColor=0B0D12"></a>
<a href="https://github.com/Ar4ikov/vllm-qwen-boost/actions/workflows/submodule-check.yml"><img alt="patch integrity" src="https://img.shields.io/github/actions/workflow/status/Ar4ikov/vllm-qwen-boost/submodule-check.yml?branch=main&label=patch%20integrity&labelColor=0B0D12"></a>
<a href="https://github.com/Ar4ikov/vllm-qwen-boost/pkgs/container/vllm-qwen-boost"><img alt="ghcr.io" src="https://img.shields.io/badge/ghcr.io-ar4ikov%2Fvllm--qwen--boost-2496ED?labelColor=0B0D12&logo=docker&logoColor=white"></a>
<a href="https://github.com/Ar4ikov/vllm-qwen-boost/releases/latest"><img alt="release" src="https://img.shields.io/github/v/release/Ar4ikov/vllm-qwen-boost?sort=semver&display_name=tag&label=release&labelColor=0B0D12&color=0E9E74"></a>
<a href="https://github.com/vllm-project/vllm/releases/tag/v0.29.0"><img alt="vLLM 0.29.0" src="https://img.shields.io/badge/vLLM-0.29.0-5C3EE8?labelColor=0B0D12"></a>
<a href="https://github.com/Ar4ikov/HyperQwen/tree/awq-asym"><img alt="HyperQwen awq-asym" src="https://img.shields.io/badge/HyperQwen-awq--asym-0E9E74?labelColor=0B0D12"></a>
<a href="https://huggingface.co/collections/Ar4ikov"><img alt="checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20checkpoints-prepared-FFD21E?labelColor=0B0D12"></a>
<a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-0E9E74?labelColor=0B0D12"></a>
</p>

</div>

[HyperQwen](https://github.com/syv-ai/HyperQwen) is a patch series against a pinned vLLM
plus a model-preparation pipeline that serves Qwen3.8-27B on a single 24 GB card at
100+ tok/s: requantized heads, a calibrated draft vocabulary, MTP and DFlash2 speculation,
int8 Marlin GEMMs, the KVarN KV cache. It was measured to death on one checkpoint — the
symmetric AutoRound export of the base model, vision tower dropped.

This repo is the same stack for **asymmetric AWQ exports** — the two
`Ar4ikov/Qwen3.8-27B*-AWQ-W4A16-ASYM` checkpoints, quantized with llm-compressor from bf16
with zero points, the vision tower, the MTP head and the SSM gates kept in bf16 — on
**vLLM 0.29.0**, with **the vision tower on by default**, as **one container image** with
the checkpoints already prepared on the Hub. Three things had to happen for that, and all
three are here: [what this repo adds](#what-this-repo-adds).

## Quick start

```bash
git clone https://github.com/Ar4ikov/vllm-qwen-boost && cd vllm-qwen-boost
cp .env.example .env                      # CHECKPOINT=uncensored|base, and the profile knobs

docker compose --profile single up -d     # one or a few people chatting  (RTX 3090: ~107 tok/s)
docker compose --profile batch  up -d     # API backend, many concurrent requests
docker compose --profile tp2    up -d     # two cards, tensor parallel, 262k context
```

First start pulls the image (~9.5 GB) and the prepared checkpoint (~16.9 GB, once, into
`./models`), then serves an OpenAI-compatible API on `:18020`. The server binds `0.0.0.0`
with no auth until you set a key: `echo "VLLM_API_KEY=$(openssl rand -hex 24)" >> .env`.

```bash
curl http://localhost:18020/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Images go in the usual `image_url` content part. `boost/image_smoke.py` draws one with
known content, sends it, and fails if the answer does not name what is in it:

```bash
docker compose exec single venv/bin/python boost/image_smoke.py
# ANSWER: The image contains a red square and a blue circle, with the text "HYPERQWEN 42" ...
```

## What you get

One RTX 3090 (350 W, sm86), vLLM 0.29.0, `VISION=1` in every row (the tower is served,
streamed from pinned host RAM per image), `bench/run_benchmarks.sh single` from
HyperQwen, second run after boot kept. C1 = one stream of real prompts, 1,024-token
answers; `tok/step` is what speculation accepts per forward; C8 = eight concurrent
streams; pool = KV cache in tokens.

| profile (`configs/`) | checkpoint | C1 T=default | C1 T=0 | tok/step | C8 T=default | TTFT | pool |
|---|---|---|---|---|---|---|---|
| **M** `single-mtp` | uncensored, int8 heads | **107.4 tok/s** | 117.2 | 2.71 / 2.85 | 441 tok/s | 151 ms | 70,933 |
| **D** `single-dflash2` | uncensored, int8 heads | **123.0 tok/s** | 135.7 | 3.15 / 3.41 | 481 tok/s | 157 ms | 49,662 |
| **P** `single-production` (DFlash2 k=15, int8 GEMMs, int8 prefill attention) | uncensored, int8 heads | **117.5 tok/s** | 134.0 | 3.12 / 3.49 | 4 slots: n/a | **96 ms** | 37,834 |
| **L** `single-long` (100k, fp8 KV) | uncensored, int8 heads | **84.4 tok/s** | 92.1 | 2.58 / 2.73 | 464 tok/s | 175 ms | 164,705 |
| **M** + fast variant (int4-GPTQ `lm_head`, `-HyperQwen-fast`) | uncensored | **112.7 tok/s** | 127.6 | 2.74 / 2.95 | 434 tok/s | 149 ms | 80,185 |
| **D** + fast variant | uncensored | **135.9 tok/s** | 139.3 | 3.29 / 3.29 | 401 tok/s | 150 ms | 53,233 |
| **P** + fast variant | uncensored | **116.7 tok/s** | 135.5 | 2.97 / 3.44 | 4 slots: n/a | **96 ms** | 42,113 |
| **M** | base, int8 heads | **111.4 tok/s** | 116.0 | 2.85 / 2.82 | 437 tok/s | 150 ms | 70,933 |
| **D** + fast variant | base | **130.7 tok/s** | 144.9 | 3.17 / 3.44 | 449 tok/s | 153 ms | 53,233 |
| **B** `batch` (64 concurrent 128 in / 512 out, int8 GEMMs, fp8 KV) | uncensored, int8 heads | 47.3 tok/s (no speculation) | | | **1,169 tok/s** decode, 1,072 e2e at 64 | 102 ms | 215,267 |
| **T** `tp2` (two 3090s, TP=2, 262k fp8 KV, tower resident; one card on a PCIe x4 link) | uncensored, fast | 87.1 tok/s | 97.7 | 2.47 / 2.70 | 424 tok/s | 159 ms | 794,351 |

For scale: HyperQwen's own reference rows on a native 3090 at 250 W and vLLM 0.29 are
115.1 tok/s for setting B (the base model's fast variant, vision off) and 134.0 for its
production line. Every number here is a first measurement on one box; the harness and
the exact `.env` for each row are in this repo, so they are cheap to check.

## What this repo adds

### 1. The int8 Marlin path on zero-point weights

vLLM already compiles the Marlin kernel for "AWQ-INT4 with INT8 activation" — int4
weights *with zero points* times int8 activations — and tests it. Two Python asserts
meant to keep 8-bit weights off that path only admitted the symmetric type, so an
asymmetric checkpoint died at load the moment HyperQwen's `INT8_ACT=int8` was set:

```
AssertionError: W8A8 is not supported by marlin kernel.
```

That is the batch-mode default and the prefill win in single-user mode, and it is the
failure a field report called "can't use int8 — it's asymmetric". The fix is
[`patches/marlin-int8-asym-zp.patch`](https://github.com/Ar4ikov/HyperQwen/blob/awq-asym/patches/marlin-int8-asym-zp.patch):
admit `uint4` next to `uint4b8` in both places, refuse 8-bit weights as before. Measured
on a random asymmetric int4 g128 weight at this model's shapes, the int8 path lands
0.9–1.05% from the float reference against 0.26% for the bf16 path — the per-token
activation noise a symmetric body pays too, nothing from the zero points. The kernel
test, the small-M throughput numbers and the reasoning about hand-written kernels are in
[docs/kernels.md](docs/kernels.md). The patch is proposed upstream in
[syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen/pulls) and lives on the
[`awq-asym`](https://github.com/Ar4ikov/HyperQwen/tree/awq-asym) branch of the fork this
image is built from, on top of the vLLM 0.29.0 port ([#148](https://github.com/syv-ai/HyperQwen/pull/148)).

### 2. The checkpoints, prepared

The published AWQ exports are not servable on 24 GB as they ship: two 2.5 GB bf16
embedding matrices and a bf16 MTP module. HyperQwen's `prepare/quant_heads_stream.py`
requantizes `lm_head`, `embed_tokens` and the MTP module to int8 (group 128, symmetric,
in place, streaming the 18.6 GB shard) and `build_draft_vocab.py` slices the 40,960-row
draft head the MTP drafter scores. That was done once and pushed, so the image only
downloads:

| on the Hub | body | heads | MTP | vision |
|---|---|---|---|---|
| [Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM-HyperQwen](https://huggingface.co/Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM-HyperQwen) | int4 asym g128 (unchanged) | int8 g128 + 40k draft head | int8 g128 | bf16 |
| [Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM-HyperQwen](https://huggingface.co/Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM-HyperQwen) | int4 asym g128 (unchanged) | int8 g128 + 40k draft head | int8 g128 | bf16 |

`CHECKPOINT=uncensored` (default) or `CHECKPOINT=base` in `.env` picks one; any other
repo id in the same layout works too. The original exports
([uncensored](https://huggingface.co/Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM),
[base](https://huggingface.co/Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM)) stay as they are for
plain vLLM and transformers.

### 3. The vision tower, on

HyperQwen drops the tower by default (`--language-model-only`, 0.86 GiB back). Here
`VISION=1` is the image default: the tower's weights sit in pinned host RAM and each
block is copied in for its own forward (HyperQwen's `vision-tower-cpu-offload` patch),
which costs ~36 ms per image and nothing at all when no image is sent. On two cards
(`tp2` profile) it stays resident. `boost/image_smoke.py` is the proof, and the
benchmark rows above were all taken with it on.

## Profiles

Each file in `configs/` is one row of the table; `.env` takes the same variables.

| profile | file | what it is for |
|---|---|---|
| M | `configs/single-mtp.env` | the default: Qwen's own MTP head, bf16 KV, 64k context. Start here. |
| D | `configs/single-dflash2.env` | the DFlash2 block drafter, 7 drafts per pass; 48k context on 24 GB with the tower on |
| P | `configs/single-production.env` | DFlash2 with a 15-token verify block, int8 GEMMs, int8 prefill attention: fastest prefill, fastest when the answer quotes the prompt |
| L | `configs/single-long.env` | 100k context, fp8 KV via FlashInfer, MTP |
| B | `configs/batch.env` | API backend, 64 concurrent, int8 GEMMs (the profile the asym patch unlocks) |
| T | `configs/tp2.env` | two cards, TP=2, 262k context, tower resident |

Every other HyperQwen knob (`MAX_LEN`, `KV_MEM`, `DFLASH_TOKENS`, `INT8_LAYERS`,
`PREFIX_CACHE`, `EXTRA_ARGS`, ...) passes through unchanged; the launchers' own comments
in `hyperqwen/single-user/start_qwen.sh` and `hyperqwen/batch/start_qwen.sh` are the
reference.

## GPUStack

The image doubles as a [GPUStack](https://github.com/gpustack/gpustack) custom backend
(v2.2+). [gpustack/backend.yaml](gpustack/backend.yaml) registers it (UI: Inference
Backends → Add from YAML, or `POST /v2/inference-backends/from-yaml`); GPUStack then
downloads the prepared checkpoint itself and starts the container on the cards you pick:

```
/app/boost/gpustack.sh --model {{model_path}} --port {{port}} --served-model-name {{model_name}} TP={{gpu_count}} SPEC=dflash2 CTX=fast VISION=1
```

[boost/gpustack.sh](boost/gpustack.sh) turns `KEY=VALUE` tokens into HyperQwen knobs (a
deployment's `env` overrides them), passes `--flags` from the backend parameters to
`vllm serve`, adds `--tensor-parallel-size` from the GPU count, pins the DFlash2 pool on
one card, and fixes GPUStack's host-index `CUDA_VISIBLE_DEVICES` when the container sees
fewer cards. Two ready deployments: [one 3090](gpustack/model-single-3090.json) (DFlash2,
48k, ~136 tok/s) and [two 3090s](gpustack/model-tp2-3090.json) (TP=2, 64k, ~325k-token
pool, tower resident). The DFlash2 drafter is baked into the image, so nothing but the
checkpoint is downloaded. Verified on a GPUStack 2.2.2 worker with two 3090s: the
deployment downloads the checkpoint, boots in ~5 minutes, answers images, and measures
125 tok/s at the default sampling / 145 greedy on one stream through its backend port.
One trap that is GPUStack's, not this image's: a worker pod that has lost NVML after a
`systemctl daemon-reload` starts model pods with `NVIDIA_VISIBLE_DEVICES=''` (vLLM then
dies with "Failed to infer device type"); `kubectl rollout restart` of the worker fixes it.

What the launcher ends up running for the single-card DFlash2 profile, for anyone who
wants the raw flags on a stock vLLM (the speculative decoding, draft head and int8 path
need the patched image; the rest is plain vLLM 0.29):

```
vllm serve <checkpoint> --served-model-name qwen3.8-27b --host 0.0.0.0 --port 18020 \
  --gpu-memory-utilization 0.93 --kv-cache-memory 4600000000 --max-model-len 49152 --max-num-seqs 8 \
  --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 --mamba-ssm-cache-dtype float16 \
  --mamba-cache-mode align --enable-prefix-caching --async-scheduling --max-num-batched-tokens 2048 \
  --limit-mm-per-prompt '{"image":{"count":1}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --speculative-config '{"method":"dflash","model":/app/models/Qwen3.8-27B-DFlash2-W4A16,"num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"],"cudagraph_mode":"PIECEWISE"}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prompt-tokens-details --sse-keep-alive-interval 30
# env: VLLM_SPEC_DECODE_ATTN=1 VLLM_DFLASH2_LOOKUP=1 VLLM_VISION_CPU_OFFLOAD_GB=1
#      VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
#      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Building it

The image is HyperQwen's own recipe (Python 3.12 venv, vLLM 0.29.0, every patch in
`patches/series` applied at `--fuzz 0`, KVarN, `verify.sh --install` at build time) from
the `hyperqwen/` submodule, plus `boost/`. It builds on a GPU-less runner in ~10 minutes.

| tag | when |
|---|---|
| `ghcr.io/ar4ikov/vllm-qwen-boost:latest` | every push to `main` |
| `ghcr.io/ar4ikov/vllm-qwen-boost:sha-<7>` | immutable, one per commit |
| `ghcr.io/ar4ikov/vllm-qwen-boost:1.2.3`, `1.2`, `1` | a `v1.2.3` tag, which also cuts a GitHub release |

`IMAGE_TAG=1.2.3 docker compose --profile single up -d` pins a release. A second
workflow applies the submodule's whole patch series to a pristine vLLM checkout at the
pinned version on every push and pull request, so a submodule bump cannot land a series
the image build would refuse.

```bash
git submodule update --init             # after cloning, or use --recurse-submodules
docker compose build                    # the same image, locally
```

Bare metal: follow `hyperqwen/docs/install.md` with `vllm==0.29.0`, then
`VISION=1 MODEL=$PWD/models/<checkpoint> bash single-user/start_qwen.sh`.

## Custom kernels

The int8 kernel this repo unlocks was not written here; it was compiled and unreachable.
Before writing a GEMV of our own, [docs/kernels.md](docs/kernels.md) measures what one
could win at decode: the big GEMMs already run at 81–83% of the 3090's bandwidth at M =
1..8, the small-N projections (6144 → 5120) at 42%, and the whole gap is about 6% of a
decode step. The speed levers that pay first are the int4-GPTQ `lm_head` (half the bytes
per verify step), the draft head, DFlash2, and the int8 path where it is compute-bound —
which is what the table above chases. The benchmark is the yardstick a custom kernel has
to beat.

## Reproducing the numbers

```bash
docker compose exec single bash bench/run_benchmarks.sh single   # discard: JIT warmup reads 30-50% low
docker compose exec single bash bench/run_benchmarks.sh single   # keep
docker compose exec single venv/bin/python boost/image_smoke.py
docker compose exec single venv/bin/python bench/test_marlin_int8_asym.py
```

## Upstream

- [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) — the stack; this repo is a
  packaging of its `awq-asym` fork branch and sends its changes back as pull requests.
- [cpuchip's vLLM 0.29.0 port](https://github.com/syv-ai/HyperQwen/pull/148) — the base
  this branch sits on.
- [KVarN](https://github.com/syv-ai/HyperQwen/tree/main/kvarn) (Huawei CSL, Apache-2.0),
  [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (incoai, requantized by
  syv-ai), [Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B) (Alibaba Qwen, Apache-2.0).

## License

Apache-2.0, like HyperQwen, vLLM and the model.
