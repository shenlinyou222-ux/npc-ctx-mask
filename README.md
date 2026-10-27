# Context-Mask: request-level KV block visibility for RAG

A small patch for **vLLM 0.30 + FlexAttention** that lets each request declare *which KV blocks it is allowed to read*. Built for world-model / role-play RAG, where every character must see a different subset of one shared corpus.

> Status: **engineering report / work in progress**. Numbers below are reproducible on a single 8 GB GPU.
> We are looking for an arXiv endorser and for feedback from engine maintainers — see [CALL_FOR_FEEDBACK.md](docs/CALL_FOR_FEEDBACK.md).

---

## What it does

Traditional RAG re-assembles the prompt per request: when the visible document set changes, the token sequence changes, and the whole prefix must be re-prefilled. We keep the sequence fixed and change **who can be read**:

```
agent side : "I am March 7th, story progress 2.4"     (intent only)
engine side: block store + block labels + per-request visible set
```

- **Masking is exact**: in the same context, a canary in a visible block is read 8/8; the same canary in a masked block is read 0/8 — identical to physically deleting it.
- **Quality is unchanged**: masking 24%–97% of blocks gives the same accuracy as no masking and as text deletion.
- **TTFT drops 16×**: token sequence unchanged → prefix cache hit (0.98 s → 0.06 s).
- **Independently, position distance has a budget**: retrieval degrades past ~32K (this model), dies at 160K; compressing the position axis recovers it completely.

## Results at a glance

| Experiment | Result |
|---|---|
| Canary (mask exactness) | masked 0/8 leaked vs visible 8/8 read |
| Mask vs delete vs no-mask (Baroque wiki, 51% masked) | 9/10 · 10/10 · 10/10 |
| Noise ablation (HotpotQA, 97% masked) | 22/24 · 22/24 |
| Version-gated dataset (ZZZ 2.0–3.2, 60 questions) | leak 58/60 → **0/60**, accuracy 58/60 unchanged |
| TTFT (7.5K context) | 0.98 s → **0.06 s** |
| Position gap scan | 32K: 6/6 · 80K: 3–4/6 · 160K: 0/6 |
| Position compression (160K → 40K effective) | 0/6 → **6/6** |

## Repo layout

```
patch/npc_ctx_mask.patch   # 5 files, ~210 added lines (vLLM 0.30)
scripts/                   # every experiment, self-contained
docs/paper_draft_v0.1.md   # full write-up (Chinese)
docs/engineering_notes_v1.md  # six rules + calibration protocol
docs/CALL_FOR_FEEDBACK.md  # what we want feedback on
data/                      # dataset description (pointer-based, CC BY-SA)
```

## Quickstart

```bash
# 1. apply patch (vLLM 0.30, FlexAttention)
patch -p1 -d $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/.. < patch/npc_ctx_mask.patch

# 2. start a server with the mask env var
export NPC_VIS_BLOCKS=$PWD/mask.json      # {"scope_name": [[blk_start, blk_end), ...]}
python -m vllm.entrypoints.openai.api_server \
  --model <your-model> --attention-backend FLEX_ATTENTION \
  --enable-prefix-caching --enforce-eager --max-model-len 8192

# 3. a request declares its scope; listed blocks are hidden
curl ... -d '{"model":"m","request_id":"scope=belobog|x1","messages":[...]}'
```

`mask.json` is a **deny list**: blocks listed there are hidden; everything else (including newly generated tokens) is visible.

## Block alignment

FlexAttention works on 16-token KV blocks. To make "one document = whole blocks", pad each document to a block boundary:

```
a = ceil(current_tokens / 16)
append document text + "\n"
pad with U+200C (zero-width non-joiner, exactly 1 token in Qwen) to a multiple of 16
b = current_tokens / 16
```

## Known limitations

- Position budget (~32K) is model-specific and was measured on one model only.
- Masking does **not** make the model say "I don't know" (refusal is a separate layer).
- Masking does **not** improve accuracy in our tests (random distractors only; misleading distractors untested).
- The patch is against vLLM 0.30 and will need rebasing.

## License

Code: MIT. Dataset annotations: CC BY-SA 4.0 (source attribution in `data/NOTICE.md`).
