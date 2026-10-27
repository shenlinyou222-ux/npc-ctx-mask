# Call for feedback (and an arXiv endorser)

We built a small vLLM patch and a set of measurements around one question:

> **Can a request tell the engine which KV blocks it is allowed to read — instead of the agent re-assembling a shorter prompt?**

We think this is the right place for "who may know what" in RAG (role-play / multi-tenant / story-progress gating), because it keeps the token sequence fixed, so the prefix cache keeps hitting.

## What we measured (single 8 GB GPU, Qwen3-1.7B, vLLM 0.30 + FlexAttention)

| Claim | Evidence |
|---|---|
| Masking is exact, not soft | canary in masked block: read 0/8; same canary visible: 8/8; identical to deletion |
| Quality unchanged | masking 24–97% of blocks → same accuracy as no-mask and as text deletion |
| 16× faster | TTFT 0.98 s → 0.06 s (7.5K context, prefix cache hit) |
| Position distance is a separate budget | 32K gap 6/6 · 80K 3–4/6 · 160K 0/6; position compression (160K→40K effective) restores 6/6 |

Repo: patch (5 files, ~210 lines), all experiment scripts, raw JSON results, and a small version-gated dataset (ZZZ 2.0–3.2, 62 questions, closed-book 0/60).

## Specific questions

1. **API**: would a request-level `visible_blocks` / `hidden_blocks` field fit vLLM's direction? We implemented it as a deny-list keyed by a `scope=` tag in `request_id` — deliberately minimal, but probably not the right surface.
2. **Overlap**: how does this relate to existing block-manager features (prefix cache salting, KV cache groups, `kv_transfer`)? Is there already a supported way to express "this request may not read these blocks"?
3. **Calibration**: we claim a model's usable *position span* can be measured in ~1 minute with ~300 tokens of KV (place evidence at position 0, push the question to gap = 8K/32K/80K/160K). Is that a sound proxy for effective context length, or are we missing an interaction?
4. **Block granularity**: FlexAttention forces 16-token blocks. Is block-level (not token-level) visibility a real limitation in practice?
5. **What would make this useful to you?** — a patch, an RFC, a benchmark, or nothing (i.e. "this belongs in the application layer").

## What we are not claiming

- Masking does **not** improve accuracy in our tests (only random distractors were tested).
- Masking does **not** make the model abstain (refusal is a separate layer).
- The position budget is measured on **one** model.

## Asks

- **Feedback** on the questions above — a GitHub issue reply is perfect.
- **An arXiv endorsement** for `cs.CL` (first-time submitter). If you think the report is worth posting, the endorsement code can be entered at https://arxiv.org/auth/endorse.

Contact: GitHub [@shenlinyou222-ux](https://github.com/shenlinyou222-ux) · repo: https://github.com/shenlinyou222-ux/npc-ctx-mask
