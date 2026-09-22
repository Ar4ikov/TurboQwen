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
| **T** `tp2` (two 3090s, TP=2, 262k fp8 KV, MTP, tower resident; one card on a PCIe x4 link) | uncensored, fast | 87.1 tok/s | 97.7 | 2.47 / 2.70 | 424 tok/s | 159 ms | 794,351 |
| **T** two 3090s, DFlash2, 64k bf16 KV | uncensored, fast | 124.8 tok/s | 133.5 | 3.19 / 3.38 | 337 tok/s | 148 ms | 324,791 |
| **G** the GPUStack deployment: two 3090s, DFlash2, **262k**, int8 KV pinned to 7.08 GiB per card (1.55 requests of the full context) | uncensored, fast | **127.1 tok/s** | 141.6 | | | 151 ms | 406,694 |
| the same at 131k, 4.8 GiB per card | uncensored, fast | 125.0 tok/s | 135.0 | | | 147 ms | 252,143 |
| 131k, 4.8 GiB, fp8 KV + MTP instead | uncensored, fast | 91.1 tok/s | 91.4 | | | 156 ms | ~252k |

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

### 4. TurboQuant with speculative decoding

vLLM 0.29 ships four TurboQuant KV caches (`turboquant_k8v4`, `turboquant_4bit_nc`,
`turboquant_k3v4_nc`, `turboquant_3bit_nc`), and on this model with DFlash2 or MTP
every one of them answered garbage at 250 tok/s: the backend routes the speculative
verify block through its prefill path, and the captured CUDA graphs replay it with
stale shapes. Two patches in the series
([turboquant-spec-as-decode](https://github.com/Ar4ikov/HyperQwen/blob/awq-asym/patches/turboquant-spec-as-decode.patch),
[turboquant-skip-layers-by-name](https://github.com/Ar4ikov/HyperQwen/blob/awq-asym/patches/turboquant-skip-layers-by-name.patch))
make the verify block a decode there and keep the drafter's sliding-window layers in
bf16, so all four presets are correct and within 10% of int8 on short prompts. What
they cost at long context, and why int8 stays the default here, is measured in
[KV cache types](#kv-cache-types).

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

## KV cache types

Every KV cache vLLM 0.29 offers for this model, measured on the same box in the same
geometry as the GPUStack deployment: two RTX 3090 at TP=2, `--max-model-len 262144`
wherever the cache holds it, the pool pinned to 7.6e9 bytes per card, DFlash2 k=7 unless
noted, vision resident, prefix caching on, vLLM 0.29.0 with the patch series
([bench/kvcamp.sh](bench/kvcamp.sh) runs the whole table; one variant is one boot). C1 is
the usual eight real prompts × 1,024 tokens through `vllm bench serve`, e2e tok/s /
decode tok/s from the mean TPOT, default sampling and greedy. The 120k columns are
[boost/long_ctx_probe.py](boost/long_ctx_probe.py): generated prose with one needle
sentence at 50% depth, greedy, thinking off, cold TTFT with the prefill rate, the decode
rate of the (12-token) answer, whether the needle came back, and the TTFT of a second
question over the same document.

| `--kv-cache-dtype` | attention backend | pool, tokens | × 262k | C1 default | C1 greedy | TTFT | 120k cold TTFT (prefill) | 120k decode | needle | 2nd question |
|---|---|---|---|---|---|---|---|---|---|---|
| `bfloat16` (at 131k: 262k does not fit this pin) | FLASH_ATTN | 207,113 | 1.58× at 131k | 122.3 / 126.9 | 138.0 / 142.5 | 147 ms | 128 s (947 tok/s) | 72.3 | retrieved | 1.2 s |
| `int8_per_token_head` (**the deployment**) | TRITON_ATTN + HyperQwen's split-KV verify kernel | 406,694 | 1.55× | 124.0 / 128.2 | 135.1 / 139.3 | 147 ms | 246 s (490) | 52.4 | retrieved | 5.3 s |
| `fp8` (`SPEC=mtp`: DFlash2's fp8 verify kernel needs sm89+) | FLASHINFER | 414,995 | 1.58× | 92.7 / 94.0 | 95.6 / 97.8 | 157 ms | 133 s (906) | 111.1 | retrieved | 1.8 s |
| `int4_per_token_head` (HyperQwen's experimental route) | TRITON_ATTN | 731,019 | 2.79× | 111.0 / 114.7 | 118.1 / 122.2 | 158 ms | 243 s (497) | 71.6 | retrieved | 6.8 s |
| `kvarn_k4v2_g128` (`CTX=huge`) | KVARN, FLASH_ATTN for the drafter | 789,471 | 3.01× | 111.4 / 115.1 | 128.6 / 133.3 | 170 ms | 128 s (942) | 34.2 | retrieved | 2.2 s |
| `turboquant_4bit_nc`, **stock vLLM backend** | TURBOQUANT, FLASH_ATTN for the drafter | 474,867 | 1.81× | 204 / 211 (garbage) | 247 / 258 (garbage) | 171 ms | 123 s (981) | 275 (garbage) | **missing** | no cache hit |
| `turboquant_k8v4`, stock backend | TURBOQUANT + FLASH_ATTN | 413,327 | 1.58× | 245 / 255 (garbage) | 251 / 256 (garbage) | 170 ms | 123 s (980) | 275 (garbage) | **missing** | no cache hit |
| `turboquant_4bit_nc`, `SPEC=off`, stock backend | TURBOQUANT | 758,837 | 2.89× | 65.0 / 65.6 | 64.2 / 64.8 | 165 ms | | | | |

With the two TurboQuant patches the series now carries (below), the same four caches:

| `--kv-cache-dtype` | attention backend | pool, tokens | × 262k | C1 default | C1 greedy | TTFT | 120k cold TTFT (prefill) | 120k decode | needle | 2nd question |
|---|---|---|---|---|---|---|---|---|---|---|
| `turboquant_4bit_nc` | TURBOQUANT, FLASH_ATTN for the drafter | 474,867 | 1.81× | 120.1 / 122.5 | 126.5 / 130.5 | 177 ms | 129 s (938) | 11.7 | retrieved | no cache hit, 129 s |
| `turboquant_k8v4` | TURBOQUANT + FLASH_ATTN | 413,327 | 1.58× | 129.1 / 133.0 | 127.2 / 130.5 | 174 ms | 127 s (952) | 15.3 | retrieved | no cache hit |
| `turboquant_k3v4_nc` | TURBOQUANT + FLASH_ATTN | 528,482 | 2.02× | 114.1 / 117.8 | 119.6 / 124.1 | 173 ms | 128 s (943) | 10.5 | retrieved | no cache hit |
| `turboquant_3bit_nc` | TURBOQUANT + FLASH_ATTN | 612,368 | 2.34× | 113.7 / 117.8 | 121.1 / 125.0 | 177 ms | 128 s (946) | 9.6 | retrieved | no cache hit |
| `turboquant_4bit_nc`, `SPEC=mtp` (no drafter, no window layers) | TURBOQUANT | 545,259 | 2.08× | 115.3 / 118.5 | 125.5 / 129.4 | 180 ms | 131 s (923) | 21.9 | retrieved | 20.2 s, 106,496 tokens cached |

Reading the two tables:

- **On short prompts every correct cache is within about 10% of the others** (111–129
  tok/s at the default sampling); the exceptions are `fp8`, which loses DFlash2 on this
  card, and the two stock TurboQuant rows, whose 250 tok/s is garbage.
- **The pool at 262k is not what the bit counts promise.** On this hybrid model the pool
  is paid mostly by the DeltaNet state pages and the drafter's bf16 sliding-window
  layers, not by the 16 attention layers, so `turboquant_4bit_nc` holds 17% more than
  `int8_per_token_head`, `turboquant_k8v4` nothing more, and the 3-bit presets 30–50%
  more; `int4_per_token_head` (2.79×) and KVarN (3.01×) are the caches that actually
  fit two and three 262k requests.
- **Long context is where the caches differ.** Cold prefill is ~950 tok/s on
  FlashAttention-based routes (bf16, fp8, KVarN, TurboQuant) and ~490 on the Triton
  routes (int8, int4). Decode at 120k: bf16 72, int4 72, int8 52, KVarN 34,
  TurboQuant 10–15 tok/s. TurboQuant's decode kernel scans four positions per
  iteration, and the verify block reads the cache once per draft row, so it is the
  slowest long-context choice here by a wide margin. Its prefix cache also never hits
  under DFlash2 (the bf16 window layers promote the attention block to 8,192 tokens and
  the second question re-prefills), while with `SPEC=off` or `SPEC=mtp` it does (8.7 s
  and 20.2 s second turns).
- **Quality:** the needle at 120k came back on every correct row, including all four
  TurboQuant presets. vLLM's own perplexity deltas for the presets on dense models are
  +1.2% (`k8v4`), +2.7% (`4bit_nc`), +10.6% (`k3v4_nc`) and +20.6% (`3bit_nc`); no
  GSM8K battery was run here.
- **So:** `int8_per_token_head` stays the deployment's cache; `int4_per_token_head` is
  the one to take when 262k has to fit 2–3 requests; TurboQuant is now correct and
  supported, and on this model it is a worse trade than either.

**Why stock TurboQuant "works badly" with speculative decoding, and the fix.** vLLM's
TurboQuant backend declares CUDA-graph support for uniform batches but registers
`reorder_batch_threshold=1` without spec-as-decode, so every verify block of DFlash2 or
MTP (1+k query tokens per request) is routed through its *prefill* path: a per-request
Python loop over CPU-side metadata that launches the decode kernel with synthetic
per-token sequences (or dequantizes the whole cached context). The runner captures the
uniform verify batches as full CUDA graphs anyway, and a captured Python loop replays
the shapes it saw at capture time. The result is exactly what the two stock rows show:
"decode" at 250 tok/s because the drafter's proposals are accepted almost whole
(6.9 of 7 per step) while the target attends to the wrong keys, `quiquiqui…` as the
answer, the needle lost, and the prefix cache never hit. Without speculation
(`SPEC=off`) the same backend is correct, because plain decode is its captured path.
[turboquant-spec-as-decode.patch](https://github.com/Ar4ikov/HyperQwen/blob/awq-asym/patches/turboquant-spec-as-decode.patch)
makes the verify block a decode there: the metadata builder expands every request with
1+k query tokens into one decode row per token (seq_len = context so far + 1, the
request's block table repeated) in persistent device buffers, and the TurboQuant decode
kernel attends all rows in one launch; exact, because the block's K/V are stored before
the forward, and graph-safe, because the captured kernel reads buffers the builder
refills every step. A second small patch lets `--kv-cache-dtype-skip-layers
sliding_window` (which the option documents) survive the boundary-layer merge that
sorted the list as integers; the drafter's five sliding-window layers must stay bf16
because TurboQuant has no window mask.

## GPUStack

The image doubles as a [GPUStack](https://github.com/gpustack/gpustack) custom backend
(v2.2+). [gpustack/backend.yaml](gpustack/backend.yaml) registers it (UI: Inference
Backends → Add from YAML, or `POST /v2/inference-backends/from-yaml`); GPUStack then
downloads the prepared checkpoint itself and starts the container on the cards you pick:

```
/app/boost/gpustack.sh --model {{model_path}} --port {{port}} --served-model-name {{model_name}} TP={{gpu_count}} SPEC=dflash2 CTX=fast VISION=1
```

A deployment's **backend parameters are plain `vllm serve` flags**. Six of them overlap
with what HyperQwen's launcher decides itself, so [boost/gpustack.sh](boost/gpustack.sh)
translates those into its knobs (`--max-model-len`, `--kv-cache-memory` per GPU in bytes,
`--kv-cache-dtype`, `--max-num-seqs`, `--gpu-memory-utilization`,
`--[no-]enable-prefix-caching`) and passes everything else through
(`--default-chat-template-kwargs`, `--reasoning-parser`, `--tool-call-parser`, ...).
Speculation and vision stay env knobs (`SPEC=dflash2|mtp|off`, `VISION`, `VISION_OFFLOAD`).
It also adds `--tensor-parallel-size` from the GPU count and fixes GPUStack's host-index
`CUDA_VISIBLE_DEVICES` when the container sees fewer cards. The translation table is
checked at image build time by a dry run ([boost/test_gpustack_sh.sh](boost/test_gpustack_sh.sh),
`GPUSTACK_DRY_RUN=1`).

What each `--kv-cache-dtype` runs on this stack, with the measured pool, speed and
long-context behaviour of every one of them, is the [KV cache types](#kv-cache-types)
section above; the wrapper's mapping in one line: `bfloat16` → `CTX=fast`,
`int8_per_token_head` → `CTX=long` (DFlash2 kept), `fp8` → `CTX=long` with `SPEC=mtp`,
`int4_per_token_head` → `CTX=long` plus the flag, `kvarn_k4v2_g128` → `CTX=huge`,
`turboquant_*` → `CTX=fast` with `KV_DTYPE=<dtype>`.

Two ready deployments: [one 3090](gpustack/model-single-3090.json) (DFlash2, 48k, ~136
tok/s) and [two 3090s](gpustack/model-tp2-3090.json) (TP=2, DFlash2, **262,144 context**,
int8 KV pinned to 7.08 GiB per card = a 406,694-token pool, 1.55 requests of the full
context, tower resident, reasoning effort `medium` as the template default). The DFlash2
drafter is baked into the image, so nothing but the checkpoint is downloaded. Verified on a
GPUStack 2.2.2 worker with two 3090s: the deployment downloads the checkpoint, boots in ~5
minutes, answers images, and measures 127 tok/s at the default sampling / 142 greedy on one
stream through its backend port at 18.0 GB of VRAM per card.

Why int8 and not bf16 or fp8 for 262k: bf16 KV at TP=2 is 32 KB per token per card, so one
262k request alone is 8.6 GB per card beside 8.35 GB of weights, and 1.5 of them do not fit;
int8 halves that and, unlike fp8, keeps DFlash2 on sm86. What the deployment does with a
long document ([boost/long_ctx_probe.py](boost/long_ctx_probe.py): generated non-repeating
prose with one needle sentence at 50% depth, greedy, thinking off, one stream):

| prompt | cold TTFT (prefill rate) | decode | needle | second question over the cached prefix |
|---|---|---|---|---|
| 8 real prompts, 1,024-token answers (C1) | 151 ms | 131.1 tok/s | | |
| 120,757 tokens | 245 s (493 tok/s) | 51.6 tok/s | retrieved | TTFT 5.3 s, 119,232 tokens cached |
| 240,119 tokens | 856 s (280 tok/s) | 34.6 tok/s | retrieved | TTFT 11.5 s, 238,464 tokens cached |

The decode column at 120k/240k is a 12-token answer, so read it as the order of magnitude:
past 100k of context DFlash2 accepts fewer drafts on anything that is not reproducing the
prompt (HyperQwen measures 32-47 tok/s at 112k on one card), and the cold prefill is the
Triton int8 route's known cost, about 2x FlashAttention's at 112k and superlinear past
that. Prefix caching is what makes the mode usable: the second question over the same
document costs seconds, not minutes. No preemption and no engine error at either length.

[gpustack/register.py](gpustack/register.py) does the four API calls the UI would make for
you: the backend from the YAML, the model, the **model route** — without a route the
gateway answers `Model not found` for a model that is up and healthy on its port — and
the **metrics mapping**. GPUStack's worker scrapes an instance's `/metrics` only when its
backend name has a `runtime_mapping` entry in the metrics config (builtin: vLLM, SGLang,
MindIE), so a custom backend reads "No data" on the GPUStack Model dashboard while the pod
serves all 400+ `vllm:` metrics; `register.py` registers the vLLM mapping under
`hyperqwen-custom` through `POST /v2/metrics/config` (the server writes
`custom_metrics_config.yaml` into its data dir, the worker refreshes within 5 minutes).
That file replaces the builtin one, so re-run `register.py` after a GPUStack upgrade to
pick up new builtin mappings.
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
