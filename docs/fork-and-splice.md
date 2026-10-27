# Fork & splice: can a branch's KV be merged back into the main context?

## The question

A "gather → branch → gather" architecture: the main agent forks into several sub-agents that share the prefix but run in parallel (think / act / predict). Each produces a message. The messages are merged back into the main context.

Merging can be done two ways:

- **text splice** — append the message tokens and re-prefill them (cheap for short messages);
- **KV splice** — take the K/V the branch already computed and place them in the main sequence.

KV splice is the interesting one, because it avoids any re-prefill. The obvious objection is that the branch's tokens never attended to the main context ("cross-attention is missing"). CacheBlend (arXiv 2405.16444) handles exactly that problem for RAG chunks and finds that naive splicing degrades quality — they recompute ~18% of tokens to fix it.

But there is a difference in kind:

- RAG chunks **should** see each other (they are pieces of one document);
- branch messages **should not** (they are messages, produced independently on purpose).

So the missing cross-attention may be exactly the part we do not want.

## Experiment: next-token distribution

Setup isolates the cross-attention variable: the two branches get **equal-length** questions, so the message tokens occupy **identical absolute positions** in both cases. Only "which question preceded it" differs.

| case | KL(baseline ‖ splice) | top-1 | top-5 overlap |
|---|---|---|---|
| short message, different question | 0.0026 | same | 5/5 |
| short message, different question | 0.0019 | same | 5/5 |
| short message, different question | 0.0053 | same | 5/5 |
| long message, **same** question | 0.0008 | same | 5/5 |
| long message, different question | 0.186 | same | 4/5 |
| long message, different question | 0.921 | same | 3/5 |

Reading: with short messages the splice is numerically identical to a full prefill. With long messages the divergence grows **only when the branch was answering a different question** — i.e. the difference is semantic, not positional or structural.

## Experiment: end-to-end

Does the main agent still answer correctly? Answer only exists in the branch message, so the message is genuinely required.

| arm | correct |
|---|---|
| no message | **0/10** |
| text splice (full prefill) | **10/10** |
| **KV splice** | **10/10** |

Messages are ~10 tokens; questions engineered to equal length; 10 random cases with random codes (not memorisable).

## Conclusion

For **messages** (as opposed to document chunks), KV splicing is equivalent to re-prefilling, both in next-token distribution (KL ≈ 0.003) and end-to-end accuracy (10/10 vs 10/10), while the message remains the sole source of the answer (0/10 without it).

CacheBlend's selective-recomputation is needed when chunks must see each other. It is not needed here.

## What is still missing (engineering, not semantics)

1. **Position alignment.** The experiment forced equal-length questions so the message lands at the same positions. In a real system the branch's positions differ from the main sequence, so either:
   - positions are **pre-allocated at generation time** (branch *i* writes into `[base_i, base_i+len)`), which vLLM does not currently expose; or
   - the K/V are **re-rotated** afterwards — mathematically exact, but our numerical check (relative error ~10%) has not passed yet.
2. **Parallelism ≠ speedup on one GPU.** Batching overlaps latency, but the arithmetic is shared. The gain is on the critical path (max instead of sum), not throughput.
3. **Branch independence must be real.** If a later branch needs an earlier branch's output, the fork is invalid and the sequence must be sequential.

## Reproduce

```bash
python scripts/kv_splice_test.py     # next-token KL
python scripts/kv_splice_e2e.py 10   # end-to-end, 3 arms
```
