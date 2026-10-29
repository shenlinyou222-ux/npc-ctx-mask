# -*- coding: utf-8 -*-
"""修正后的比较类测试。

上一版的缺陷：问“A 是否比 B 多”，只看到 B 的分数就能靠“不知道→否”蒙对。
本版改为问“谁进球更多”，选项是两个人名：
    只看一块时必须无法判断（用作废题检测）。

并且：
  - gold 严格 50/50 平衡，避免“no 偏置”刷分；
  - 加入 only_A / only_B 两臂，若它们显著高于随机，说明题目仍有捷径。
"""
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
FILL = 200
rnd = random.Random(20261105)
PLAYERS = ["Messi", "Ronaldo", "Pelé", "Zidane", "Maradona", "Cruyff"]
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

    def pick(lg):
        lg = lg.float()
        return max(range(4), key=lambda i: float(lg[letters[i]]))

    @torch.no_grad()
    def direct(seq):
        out = model(input_ids=torch.tensor([seq], device="cuda"))
        v = pick(out.logits[0, -1])
        del out
        torch.cuda.empty_cache()
        return v

    @torch.no_grad()
    def build_pool(ids):
        with rc.capture_rope() as cap:
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        kpre = [k.float() for k in cap.pre]
        vals = [cache.layers[li].values.clone() for li in range(len(cache.layers))]
        del out, cache
        torch.cuda.empty_cache()
        return kpre, vals

    @torch.no_grad()
    def ask_pool(kpre, vals, sel_idx, q_ids):
        W = len(sel_idx)
        idx = torch.tensor(sel_idx, device="cuda")
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
        c = DynamicCache()
        for li in range(len(kpre)):
            k = rc.rotate_k(kpre[li][:, :, idx, :], cos.float(), sin.float()).to(torch.bfloat16)
            c.update(k, vals[li][:, :, idx, :], li)
        o = model(input_ids=torch.tensor([q_ids], device="cuda"), past_key_values=c,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long))
        v = pick(o.logits[0, -1])
        del o, c
        torch.cuda.empty_cache()
        return v

    hit = {k: 0 for k in ("full", "fresh", "pool_both", "only_A", "only_B")}
    cost = {k: 0.0 for k in hit}
    used = 0

    for t in range(N):
        a, b = rnd.sample(PLAYERS, 2)
        # gold 严格 50/50
        if t % 2 == 0:
            ga, gb = rnd.randint(12, 20), rnd.randint(0, 7)
        else:
            ga, gb = rnd.randint(0, 7), rnd.randint(12, 20)
        gold = 0 if ga > gb else 1

        S0 = "You read two records and answer with one letter.\n"
        A = "Record 1: %s scored %d goals.\n" % (a, ga)
        B = "Record 2: %s scored %d goals.\n" % (b, gb)
        Q = ("\nQuestion: who scored more goals?\n"
             " A) %s\n B) %s\n C) they scored the same\n D) cannot tell\nAnswer:" % (a, b))

        s0, aA, aB, aQ = enc(S0), enc(A), enc(B), enc(Q)
        fid = fill(FILL)
        ids = s0 + aA + fid + aB + fid + aQ
        pa0, pa1 = len(s0), len(s0) + len(aA)
        pb0 = pa1 + len(fid)
        pb1 = pb0 + len(aB)
        pre = list(range(0, len(s0)))
        used += 1

        t0 = time.time(); vf = direct(ids); cost["full"] += time.time() - t0
        t0 = time.time(); vfr = direct(s0 + aA + aB + aQ); cost["fresh"] += time.time() - t0
        t0 = time.time(); kpre, vals = build_pool(ids)
        vb = ask_pool(kpre, vals, pre + list(range(pa0, pa1)) + list(range(pb0, pb1)), aQ)
        cost["pool_both"] += time.time() - t0
        t0 = time.time()
        va1 = ask_pool(kpre, vals, pre + list(range(pa0, pa1)), aQ); cost["only_A"] += time.time() - t0
        t0 = time.time()
        vb1 = ask_pool(kpre, vals, pre + list(range(pb0, pb1)), aQ); cost["only_B"] += time.time() - t0
        del kpre, vals
        torch.cuda.empty_cache()

        for k, v in (("full", vf), ("fresh", vfr), ("pool_both", vb),
                     ("only_A", va1), ("only_B", vb1)):
            hit[k] += int(v == gold)
        print("  %2d) %s=%d %s=%d gold=%s | full=%s fresh=%s pool=%s onlyA=%s onlyB=%s"
              % (t + 1, a, ga, b, gb, "A" if gold == 0 else "B",
                 "O" if vf == gold else "X", "O" if vfr == gold else "X",
                 "O" if vb == gold else "X", "O" if va1 == gold else "X",
                 "O" if vb1 == gold else "X"), flush=True)

    print("\nn=%d（四选一，随机 25%%；only_A/only_B 若明显高于 25%% 说明题目有捷径）" % used)
    for k in ("full", "fresh", "pool_both", "only_A", "only_B"):
        print("  %-10s %2d/%d  %.3fs/次" % (k, hit[k], used, cost[k] / used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
