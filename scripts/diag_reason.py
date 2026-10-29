# -*- coding: utf-8 -*-
"""单例深挖：比较类问题为什么池化后崩掉。

打印完整提示、每个选项的 logits，并做拆解：
  full            全部可见
  fresh           S0+A+B+Q 联合 prefill（理想）
  pool_both       S0+A+B 都从池里取，重紧凑
  pool_A_only     只取 S0+A
  pool_B_only     只取 S0+B
  hybrid          A 从池里取，**B 重新 prefill**（关键归因）
"""
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
FILL = 200
rnd = random.Random(20261105)
FILLW = "The warehouse log noted routine checks and nothing unusual. "


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    letters = [enc(" A")[0], enc(" B")[0], enc(" C")[0], enc(" D")[0]]
    d_head = model.config.head_dim
    unit = enc(FILLW)

    def fill(n):
        o = []
        while len(o) < n:
            o += unit
        return o[:n]

    def show(tag, lg):
        lg = lg.float()
        vals = [float(lg[i]) for i in letters]
        best = max(range(4), key=lambda i: vals[i])
        print("    %-14s A=%7.3f B=%7.3f C=%7.3f D=%7.3f  -> %s"
              % (tag, vals[0], vals[1], vals[2], vals[3], "ABCD"[best]))
        return best

    @torch.no_grad()
    def direct(seq):
        out = model(input_ids=torch.tensor([seq], device="cuda"))
        return out.logits[0, -1]

    @torch.no_grad()
    def build_pool(ids):
        with rc.capture_rope() as cap:
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        return ([k.float() for k in cap.pre],
                [cache.layers[li].values.clone() for li in range(len(cache.layers))])

    @torch.no_grad()
    def cache_from_pool(kpre, vals, sel_idx):
        W = len(sel_idx)
        idx = torch.tensor(sel_idx, device="cuda")
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
        c = DynamicCache()
        for li in range(len(kpre)):
            k = rc.rotate_k(kpre[li][:, :, idx, :], cos.float(), sin.float()).to(torch.bfloat16)
            c.update(k, vals[li][:, :, idx, :], li)
        return c, W

    @torch.no_grad()
    def ask_with_cache(cache, W, q_ids):
        o = model(input_ids=torch.tensor([q_ids], device="cuda"),
                  past_key_values=cache,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long))
        return o.logits[0, -1], o.past_key_values

    # 构造一个「A 说得少、B 说得多」的组合
    A = "Record 1: Messi scored 7 goals.\n"
    B = "Record 2: Ronaldo scored 20 goals.\n"
    S0 = "You read two records separated by logs, then answer with one letter.\n"
    Q = ("\nQuestion: did Messi score more goals than Ronaldo?\n"
         " A) yes\n B) no\n C) they scored the same\n D) cannot tell\nAnswer:")
    gold = 1  # 7 < 20 -> B) no

    s0, aA, aB, aQ = enc(S0), enc(A), enc(B), enc(Q)
    fid = fill(FILL)
    ids = s0 + aA + fid + aB + fid + aQ
    pa0, pa1 = len(s0), len(s0) + len(aA)
    pb0 = pa1 + len(fid)
    pb1 = pb0 + len(aB)

    print("上下文 %d token：S0(%d) + A(%d) + 填充(%d) + B(%d) + 填充(%d) + Q"
          % (len(ids), len(s0), len(aA), len(fid), len(aB), len(fid)))
    print("A 位于 [%d,%d)，B 位于 [%d,%d)" % (pa0, pa1, pb0, pb1))
    print("真值：Messi 7 < Ronaldo 20 → 答案 B) no\n")

    print("各臂选项 logits：")
    show("full", direct(ids))
    show("fresh", direct(s0 + aA + aB + aQ))

    kpre, vals = build_pool(ids)
    c_all, W = cache_from_pool(kpre, vals, list(range(0, len(s0))) + list(range(pa0, pa1)) + list(range(pb0, pb1)))
    lg, _ = ask_with_cache(c_all, W, aQ)
    show("pool_both", lg)
    del c_all

    c_a, Wa = cache_from_pool(kpre, vals, list(range(0, len(s0))) + list(range(pa0, pa1)))
    lg, _ = ask_with_cache(c_a, Wa, aQ)
    show("pool_A_only", lg)
    del c_a

    c_b, Wb = cache_from_pool(kpre, vals, list(range(0, len(s0))) + list(range(pb0, pb1)))
    lg, _ = ask_with_cache(c_b, Wb, aQ)
    show("pool_B_only", lg)
    del c_b

    # hybrid：A 从池里取，B 现场 prefill（B 这次看得到 A）
    c_h, Wh = cache_from_pool(kpre, vals, list(range(0, len(s0))) + list(range(pa0, pa1)))
    o = model(input_ids=torch.tensor([aB], device="cuda"), past_key_values=c_h,
              attention_mask=torch.ones((1, Wh + len(aB)), device="cuda", dtype=torch.long))
    lgh, _ = ask_with_cache(o.past_key_values, Wh + len(aB), aQ)
    show("hybrid", lgh)

    print("\n期望答案：B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
