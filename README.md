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
- **Masking irrelevant distractors costs nothing** (after fixing a 3-token block-offset bug that had made it look harmful): Baroque 9/10 = 9/10, multi-hop 6/7 = 6/7, neutral fillers 24/24 = 24/24. See [paper §5.3b](paper/paper_v1.md).
- **TTFT drops 18–20×**: token sequence unchanged → prefix cache hit (cold 1.27 s → warm 0.062 s at 7.5K). Changing *visibility* costs nothing (0.064 s); changing *text* costs 8× (0.515 s).
- **Independently, position distance has a budget**: retrieval degrades past ~32K (this model), dies at 160K; compressing the position axis recovers it completely.

## Results at a glance

| Experiment | Result |
|---|---|
| Canary (mask exactness) | masked 0/8 leaked vs visible 8/8 read (run `verify_mask_v3.py 8`; the JSON shipped is the earlier n=4 run) |
| Mask vs delete vs no-mask (Baroque wiki, 51% masked) | 9/10 · 10/10 · 10/10 |
| Filler ablation (97% masked, n=24) | real filler 22/24 → **15/24**; neutral filler 24/24 → 24/24 |
| Version-gated dataset (ZZZ 2.0–3.2, 60 questions) | leak **58/60 → 0/60**; with the answer's own version visible (gated) masked **59/60** vs full-context 58/60 |
| TTFT (7.5K context) | cold 1.27 s → warm **0.062 s** (18–20×); masked-warm 0.064 s; text-deleted-cold 0.515 s |
| Position gap scan | 32K: 6/6 · 80K: 3–4/6 · 160K: 0/6 |
| Position compression (160K → 40K effective) | 0/6 → **6/6** |

## Repo layout

```
patch/npc_ctx_mask.patch   # the vLLM patch actually used (5 files, +211)
patch/README.md            # what each hunk does + limits
paper/paper_v1.md          # arXiv draft (Chinese body, bilingual abstract)
scripts/                   # every experiment, self-contained
results/                   # raw JSON from each run
data/zzz/                  # the actual test dataset (CC BY-SA, with NOTICE)
docs/paradigm.md           # the paradigm: agents exchange KV blocks
docs/knowledge-as-blocks.md# knowledge blocks + reasoning weights
docs/reserved-slot.md      # slot hot-swap protocol and its four pitfalls
docs/CORRECTIONS.md        # a real bug we found and corrected
docs/start-position-negative.md  # negative result: start_position crashes
docs/block-alias-design.md # the missing engine primitive
docs/ROADMAP.md            # what is still missing (diversified samples)
docs/CALL_FOR_FEEDBACK.md  # what we want feedback on
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

## Verification (how to check the patch yourself)

The patch ships with a metadata header stating its base and expected result. Two checks:

```bash
# 1. does it apply to a pristine tree?
patch -p1 --dry-run < patch/npc_ctx_mask.patch     # expect: no hunks ignored

# 2. does the masked block actually become unreadable?
bash scripts/../run_canary.sh                      # or run the server yourself, then:
python scripts/canary_test.py 4
# expect: masked -> leak 0/4, visible-read 4/4
#         visible -> leak 4/4, visible-read 4/4
```

On our build: forward apply succeeds, the patched files are byte-identical to the build used for every number in this repo, and a fresh canary run on 2026-10-27 reproduced `0/4` vs `4/4`.

## For engine maintainers

We wrote up the design question separately — **should a request be able to mask blocks it may not read?** — in [docs/VLLM_RFC.md](docs/VLLM_RFC.md). The short version: we are not sure this belongs in the engine at all, and would rather hear that from the maintainers than assume it.


## Known limitations

- Position budget (~32K) is model-specific and was measured on one model only.
- Masking does **not** make the model say "I don't know" (refusal is a separate layer).
- Masking does **not** improve accuracy in our tests (random distractors only; misleading distractors untested).
- The patch is against vLLM 0.30 and will need rebasing.

## License

Code: MIT. Dataset annotations: CC BY-SA 4.0 (source attribution in `data/NOTICE.md`).
