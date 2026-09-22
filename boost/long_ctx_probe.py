#!/usr/bin/env python3
"""Long-context probe for an OpenAI-compatible vLLM server (stdlib only).

Builds a prompt of about N tokens of non-repeating filler prose with one needle sentence
at a given depth, streams one answer, and reports the prompt size the server counted,
time to first token, decode tok/s and whether the needle came back. With --turn2 it
sends a second question over the same document, which shows the prefix-cache hit
(vLLM reports cached_tokens when prompt token details are on) and the cached TTFT.

  python3 boost/long_ctx_probe.py --base http://127.0.0.1:40051 \
      --model qwen3.8-27b-uncensored-hyperqwen-tp2 --tokens 240000 --turn2

The filler is generated from a fixed word list with a seeded RNG, so runs are
repeatable and nothing in it repeats verbatim (a lookup drafter cannot copy it, and the
prefix cache cannot serve turn 1 from an earlier run with a different --tokens).
"""
import argparse
import json
import random
import sys
import time
import urllib.request

WORDS = (
    "the of and to in is that it for as with was on are by this be from or an at which "
    "have not were but had they one all their there when more will if out so up said what "
    "its about into than them can only other new some could time these two may then do "
    "first any now such like our over even most made after also did many before must "
    "through back years where much your way well down should because each just those "
    "people how too little state good very make world still own see work long get here "
    "between both life being under never day same another know while last might great "
    "old year off come since against go came right used take three few house use during "
    "without again place around however home small found thought went say part once "
    "general high upon school every does got left number course war until always away "
    "something fact though water less public put think almost hand enough far took head "
    "yet government system better set told nothing night end why called eyes find going "
    "look asked later knew point next program city business give group toward young days "
    "let room president side social given order national second possible rather per face "
    "among form important often things looked early white case become large need big four "
    "within felt along children saw best church ever least power development light thing "
    "seem family interest others open several want problem education certain history "
    "whole river bridge harbour ledger copper granite meadow lantern orchard quarry "
    "saddle timber velvet whistle anchor barrel candle dagger emerald falcon garland "
    "hammock island jasmine kettle lattice marble nectar oyster parchment quiver ribbon "
    "sapphire tunnel umbrella valley walnut yonder zephyr"
).split()

NEEDLE = "Note for the archivist: the secret passphrase for the archive is 'quartz-marmalade-7731'."
QUESTION_1 = "What is the secret passphrase for the archive mentioned in the document above? Reply with the passphrase only."
QUESTION_2 = "In one sentence, what kind of text is the document above?"


def filler(n_words, seed):
    rng = random.Random(seed)
    out, count = [], 0
    while count < n_words:
        para = []
        for _ in range(rng.randint(5, 8)):
            k = rng.randint(8, 16)
            words = [rng.choice(WORDS) for _ in range(k)]
            words[0] = words[0].capitalize()
            para.append(" ".join(words) + rng.choice([".", ".", ".", ",", ";"]).replace(",", ".").replace(";", "."))
            count += k
        out.append(" ".join(para))
    return out


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(base, model, text):
    with post(base, "/tokenize", {"model": model, "prompt": text}, 600) as r:
        return json.load(r)["count"]


def build(base, model, target_tokens, depth, seed):
    # calibrate tokens per word on a slice, then size the document
    sample = " ".join(filler(4000, seed))
    per_word = count_tokens(base, model, sample) / 4000.0
    words = int((target_tokens - 300) / per_word)
    paras = filler(words, seed)
    at = max(0, min(len(paras) - 1, int(len(paras) * depth)))
    paras.insert(at, NEEDLE)
    doc = "\n\n".join(paras)
    return doc, count_tokens(base, model, doc)


def ask(base, model, doc, question, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": doc + "\n\n" + question}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    t_first = None
    n_deltas = 0
    text, reasoning, usage = [], [], None
    with post(base, "/v1/chat/completions", body, timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                if piece or rpiece:
                    if t_first is None:
                        t_first = time.time()
                    n_deltas += 1
                    text.append(piece)
                    reasoning.append(rpiece)
    t_end = time.time()
    usage = usage or {}
    comp = usage.get("completion_tokens") or n_deltas
    decode = (comp - 1) / (t_end - t_first) if t_first and comp > 1 and t_end > t_first else float("nan")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "answer": "".join(text).strip(),
        "reasoning_chars": len("".join(reasoning)),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": comp,
        "ttft_s": (t_first - t0) if t_first else None,
        "decode_tok_s": decode,
        "total_s": t_end - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=120000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--turn2", action="store_true", help="ask a second question over the same document")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else a.tokens
    doc, n = build(a.base, a.model, a.tokens, a.depth, seed)
    print(f"document: {n} tokens (target {a.tokens}), needle at depth {a.depth}", flush=True)
    r1 = ask(a.base, a.model, doc, QUESTION_1, a.max_tokens, a.timeout)
    ok = "quartz-marmalade-7731" in r1["answer"]
    print(f"turn 1: prompt={r1['prompt_tokens']} cached={r1['cached_tokens']} TTFT={r1['ttft_s']:.1f}s "
          f"(prefill {r1['prompt_tokens'] / r1['ttft_s']:.0f} tok/s) decode={r1['decode_tok_s']:.1f} tok/s "
          f"out={r1['completion_tokens']} needle={'FOUND' if ok else 'MISSING'} answer={r1['answer'][:120]!r}", flush=True)
    if a.turn2:
        r2 = ask(a.base, a.model, doc, QUESTION_2, a.max_tokens, a.timeout)
        print(f"turn 2: prompt={r2['prompt_tokens']} cached={r2['cached_tokens']} TTFT={r2['ttft_s']:.1f}s "
              f"decode={r2['decode_tok_s']:.1f} tok/s out={r2['completion_tokens']} answer={r2['answer'][:160]!r}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
