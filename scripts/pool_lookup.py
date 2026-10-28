# -*- coding: utf-8 -*-
"""池化可用性：查找类问题能否只靠池里挑出的那一块回答。

上下文：[块A: X 的编码] + 填充 800 + [块B: Y 的编码] + 填充 800
问题：X 的编码是什么？（答案只在 A 里，不需要跨块）

三臂：
  full     全部可见
  pool_A   从池里只取块 A，重紧凑
  pool_B   从池里只取块 B（对照：问 X 却拿 Y 的块，应失败）
"""
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/npc-models-qwen3-1.7b"
if not __import__("os").path.exists(MODEL):
    MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
FILL = 800
rnd = random.Random(20261104)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine"]
FILLW = "The warehouse log noted routine checks and nothing unusual. "


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    letters = [enc(" A")[0], enc(" B")[0], enc(" C")[0], enc(" D")[0]]
    unit = enc(FILLW)
    d_head = model.config.head_dim

    def fill(n):
        o = []
        while len(o) < n:
            o += unit
        return o[:n]

    def pick(lg):
        lg = lg.float()
        return max(range(4), key=lambda i: float(lg[letters[i]]))

    @torch.no_grad()
    def run_plain(ids):
        out = model(input_ids=torch.tensor([ids], device="cuda"))
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
        cache = DynamicCache()
        for li in range(len(kpre)):
            k = rc.rotate_k(kpre[li][:, :, idx, :], cos.float(), sin.float()).to(torch.bfloat16)
            cache.update(k, vals[li][:, :, idx, :], li)
        o = model(input_ids=torch.tensor([q_ids], device="cuda"),
                  past_key_values=cache,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long))
        v = pick(o.logits[0, -1])
        del o, cache
        torch.cuda.empty_cache()
        return v

    hit = {"full": 0, "pool_A": 0, "pool_B": 0}
    cost = {k: 0.0 for k in hit}
    used = 0
    for t in range(N):
        a, b = rnd.sample(KIND, 2)
        ea = "%s %s" % (a, code())
        eb = "%s %s" % (b, code())
        ca, cb = code(), code()
        S0 = "You read records and answer with one letter.\n"
        A = "Record 1: the access code for the %s is %s.\n" % (ea, ca)
        B = "Record 2: the access code for the %s is %s.\n" % (eb, cb)
        opts = [ca, cb, code(), code()]
        rnd.shuffle(opts)
        gold = opts.index(ca)
        Q = ("\nQuestion: what is the access code for the %s?\n" % ea
             + "".join(" %s) %s\n" % ("ABCD"[i], opts[i]) for i in range(4))
             + "Answer:")

        s0, aA, aB, aQ = enc(S0), enc(A), enc(B), enc(Q)
        fid = fill(FILL)
        ids = s0 + aA + fid + aB + fid + aQ
        pa0, pa1 = len(s0), len(s0) + len(aA)
        pb0 = pa1 + len(fid)
        pb1 = pb0 + len(aB)
        used += 1

        t0 = time.time(); vf = run_plain(ids); cost["full"] += time.time() - t0
        t0 = time.time(); kpre, vals = build_pool(ids)
        va = ask_pool(kpre, vals, list(range(0, len(s0))) + list(range(pa0, pa1)), aQ)
        cost["pool_A"] += time.time() - t0
        t0 = time.time()
        vb = ask_pool(kpre, vals, list(range(0, len(s0))) + list(range(pb0, pb1)), aQ)
        cost["pool_B"] += time.time() - t0
        del kpre, vals
        torch.cuda.empty_cache()

        hit["full"] += int(vf == gold)
        hit["pool_A"] += int(va == gold)
        hit["pool_B"] += int(vb == gold)
        print("  %2d) ctx=%d A块=%d tok | full=%s pool_A=%s pool_B=%s"
              % (t + 1, len(ids), pa1 - pa0,
                 "O" if vf == gold else "X",
                 "O" if va == gold else "X",
                 "O" if vb == gold else "X"), flush=True)

    print("\nn=%d（四选一，随机 25%%）" % used)
    for k in ("full", "pool_A", "pool_B"):
        print("  %-8s %2d/%d  %.3fs/次" % (k, hit[k], used, cost[k] / used))
    print("  峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
