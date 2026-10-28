# -*- coding: utf-8 -*-
"""可行性测试 v2：**长上下文只 prefill 一次**（建池），之后按请求从池里挑块。

流程：
  ① 建池：对 1674 token 的长上下文做一次 prefill，保存每层的
     - K_pre（未旋转，用于重定位）
     - V（不含位置，直接复用）
  ② 请求：按位置挑出需要的块（A、B 两段）
     - gapped  : 原位置，不重排
     - compact : 重旋转到 0..W-1（fp32 旋转）
  ③ 问题在 W 处 decode

对照：
  full      每次重 prefill 全部 1674 token
  fresh     只 prefill 选中的 ~100 token（等于 RAG 直接拼接）
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
FILL = 800
rnd = random.Random(20261104)
PLAYERS = ["Messi", "Ronaldo", "Pelé", "Zidane", "Maradona", "Cruyff"]
FILLW = "The warehouse log noted routine checks and nothing unusual. "


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    letters = [enc(" A")[0], enc(" B")[0], enc(" C")[0], enc(" D")[0]]
    unit = enc(FILLW)
    d_head = model.config.head_dim if hasattr(model.config, "head_dim") else 128

    def fill(n):
        o = []
        while len(o) < n:
            o += unit
        return o[:n]

    def mk_mask(L, blocks):
        allowed = torch.tril(torch.ones(L, L, device="cuda", dtype=torch.bool))
        m = torch.full((L, L), float("-inf"), device="cuda")
        m[allowed] = 0.0
        for b0, b1, a0, a1 in blocks:
            m[b0:b1, a0:a1] = float("-inf")
        return m[None, None]

    def pick_from_logits(lg):
        lg = lg.float()
        return max(range(4), key=lambda i: float(lg[letters[i]]))

    @torch.no_grad()
    def run_plain(ids, blocks):
        att = mk_mask(len(ids), blocks)
        out = model(input_ids=torch.tensor([ids], device="cuda"), attention_mask=att)
        v = pick_from_logits(out.logits[0, -1])
        del out, att
        torch.cuda.empty_cache()
        return v

    @torch.no_grad()
    def build_pool(ids):
        """一次 prefill，返回每层 K_pre 与 V。"""
        with rc.capture_rope() as cap:
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        kpre = [k.float() for k in cap.pre]                       # fp32 保存
        vals = [cache.layers[li].values.clone() for li in range(len(cache.layers))]
        del out, cache
        torch.cuda.empty_cache()
        return kpre, vals

    @torch.no_grad()
    def ask_from_pool(kpre, vals, sel_idx, q_ids, mode):
        W = len(sel_idx)
        idx = torch.tensor(sel_idx, device="cuda")
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
        cache = DynamicCache()
        for li in range(len(kpre)):
            k_sel = kpre[li][:, :, idx, :]
            if mode == "gapped":
                pos = sel_idx
                c, s = rc.extract_cos_sin(model, pos, d_head)
            else:
                c, s = cos, sin
            k = rc.rotate_k(k_sel, c.float(), s.float()).to(torch.bfloat16)
            cache.update(k, vals[li][:, :, idx, :], li)
        o = model(input_ids=torch.tensor([q_ids], device="cuda"),
                  past_key_values=cache,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long))
        v = pick_from_logits(o.logits[0, -1])
        del o, cache
        torch.cuda.empty_cache()
        return v

    hit = {k: 0 for k in ("full", "fresh", "gapped", "compact")}
    cost = {k: 0.0 for k in hit}
    pool_cost = 0.0
    used = 0
    ctx_tokens = 0

    for t in range(N):
        a, b = rnd.sample(PLAYERS, 2)
        ga = rnd.choice([0, 1, 2, 3, 17, 18, 19, 20])
        gb = rnd.choice([0, 1, 2, 3, 17, 18, 19, 20])
        if abs(ga - gb) < 10:
            gb = 20 - ga if ga >= 10 else 20 - ga
        gold = 0 if ga > gb else 1

        S0 = "You read two records separated by logs, then answer with one letter.\n"
        A = "Record 1: %s scored %d goals.\n" % (a, ga)
        B = "Record 2: %s scored %d goals.\n" % (b, gb)
        Q = ("\nQuestion: did %s score more goals than %s?\n" % (a, b)
             + " A) yes\n B) no\n C) they scored the same\n D) cannot tell\nAnswer:")

        s0, aA, aB, aQ = enc(S0), enc(A), enc(B), enc(Q)
        fid = fill(FILL)
        ids = s0 + aA + fid + aB + fid + aQ
        ctx_tokens += len(ids); used += 1
        pa0, pa1 = len(s0), len(s0) + len(aA)
        pb0 = pa1 + len(fid); pb1 = pb0 + len(aB)
        sel_idx = list(range(pa0, pa1)) + list(range(pb0, pb1))
        hide = [(pa1, pb0, 0, len(ids)), (pb1, pb1 + len(fid), 0, len(ids))]

        t0 = time.time(); vf = run_plain(ids, []); cost["full"] += time.time() - t0
        t0 = time.time(); vfr = run_plain(s0 + aA + aB + aQ, []); cost["fresh"] += time.time() - t0
        t0 = time.time(); kpre, vals = build_pool(ids); pool_cost += time.time() - t0
        t0 = time.time(); vg = ask_from_pool(kpre, vals, sel_idx, aQ, "gapped"); cost["gapped"] += time.time() - t0
        t0 = time.time(); vc = ask_from_pool(kpre, vals, sel_idx, aQ, "compact"); cost["compact"] += time.time() - t0
        del kpre, vals
        torch.cuda.empty_cache()

        for k, v in (("full", vf), ("fresh", vfr), ("gapped", vg), ("compact", vc)):
            hit[k] += int(v == gold)
        print("  %2d) ctx=%d W=%d %s=%d %s=%d | full=%s fresh=%s gap=%s cmp=%s"
              % (t + 1, len(ids), len(sel_idx), a, ga, b, gb,
                 "O" if vf == gold else "X", "O" if vfr == gold else "X",
                 "O" if vg == gold else "X", "O" if vc == gold else "X"), flush=True)

    print("\nn=%d  平均上下文 %d token  选中 %d token" % (used, ctx_tokens // used, len(sel_idx)))
    for k in ("full", "fresh", "gapped", "compact"):
        print("  %-8s %2d/%d   每次请求 %.3fs" % (k, hit[k], used, cost[k] / used))
    print("  建池（一次）每次 %.3fs" % (pool_cost / used))
    print("  峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
