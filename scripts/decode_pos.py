# -*- coding: utf-8 -*-
"""验证 decode 阶段的位置接续：从紧凑缓存生成，是否与直接前向完全一致。

对比：
  direct   直接对 (S0+A+B+Q) 前向，然后贪心生成 8 个 token
  pool     从长上下文池里取 S0+A+B，重紧凑成 W，再喂 Q，生成 8 个 token

两者生成的 token 序列应完全相同；同时打印缓存长度与位置，确认
解码 token 落在 W+len(Q) 之后。
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
FILL = 200
GEN = 8
FILLW = "The warehouse log noted routine checks and nothing unusual. "


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    d_head = model.config.head_dim
    unit = enc(FILLW)

    def fill(n):
        o = []
        while len(o) < n:
            o += unit
        return o[:n]

    S0 = "You read two records and answer with one letter.\n"
    A = "Record 1: Messi scored 7 goals.\n"
    B = "Record 2: Ronaldo scored 20 goals.\n"
    Q = "\nQuestion: who scored more goals?\n A) Messi\n B) Ronaldo\n C) same\n D) unknown\nAnswer:"

    s0, aA, aB, aQ = enc(S0), enc(A), enc(B), enc(Q)
    fid = fill(FILL)
    ids = s0 + aA + fid + aB + fid + aQ
    pa0, pa1 = len(s0), len(s0) + len(aA)
    pb0, pb1 = pa1 + len(fid), pa1 + len(fid) + len(aB)
    print("池上下文 %d token；A=[%d,%d) B=[%d,%d)" % (len(ids), pa0, pa1, pb0, pb1))

    @torch.no_grad()
    def greedy(prefix_ids, cache=None, n=GEN):
        toks = []
        if cache is None:
            o = model(input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True)
            cache, lg = o.past_key_values, o.logits[0, -1]
            pos = len(prefix_ids)
        else:
            pos = cache.layers[0].keys.shape[-2]
            o = model(input_ids=torch.tensor([prefix_ids], device="cuda"),
                      past_key_values=cache,
                      attention_mask=torch.ones((1, pos + len(prefix_ids)),
                                                device="cuda", dtype=torch.long))
            cache, lg = o.past_key_values, o.logits[0, -1]
            pos += len(prefix_ids)
        for _ in range(n):
            t = int(lg.argmax())
            toks.append(t)
            o = model(input_ids=torch.tensor([[t]], device="cuda"),
                      past_key_values=cache,
                      attention_mask=torch.ones((1, pos + 1), device="cuda", dtype=torch.long))
            cache, lg = o.past_key_values, o.logits[0, -1]
            pos += 1
        return toks, pos

    # direct
    d_toks, d_pos = greedy(s0 + aA + aB + aQ)
    print("direct 生成: %r  最终位置 %d" % (tok.decode(d_toks), d_pos))

    # pool
    with rc.capture_rope() as cap:
        o = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
    cache = o.past_key_values
    kpre = [k.float() for k in cap.pre]
    vals = [cache.layers[li].values.clone() for li in range(len(cache.layers))]
    del o, cache
    torch.cuda.empty_cache()

    sel = list(range(0, len(s0))) + list(range(pa0, pa1)) + list(range(pb0, pb1))
    W = len(sel)
    idx = torch.tensor(sel, device="cuda")
    cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
    comp = DynamicCache()
    for li in range(len(kpre)):
        k = rc.rotate_k(kpre[li][:, :, idx, :], cos.float(), sin.float()).to(torch.bfloat16)
        comp.update(k, vals[li][:, :, idx, :], li)

    p_toks, p_pos = greedy(aQ, cache=comp)
    print("pool   生成: %r  W=%d Q=%d 最终位置 %d" % (tok.decode(p_toks), W, len(aQ), p_pos))
    print("Q 的起点应为 W=%d；生成起点应为 W+|Q|=%d" % (W, W + len(aQ)))
    same = d_toks == p_toks
    print("\n生成序列一致: %s" % ("是 ✔" if same else "否 ✘"))
    print("最终位置一致: %s (%d vs %d)" % ("是 ✔" if d_pos == p_pos else "否 ✘", d_pos, p_pos))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
