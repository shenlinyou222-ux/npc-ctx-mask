# -*- coding: utf-8 -*-
"""一次性重紧凑 + 之后正常多轮对话（角色扮演形态）——单 trial 版。

用法：python session_once.py <trial_index>
每次只跑 1 个 trial 后退出，避免同进程反复大块分配把驱动搞崩。
输出一行 RESULT 供外层汇总。
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
FILL = 400
TRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PLAYERS = ["Lindqvist", "Okonkwo", "Ferreira", "Nakamura"]
FILLW = "The archive index lists routine entries with no bearing on the story. "


def main():
    rnd = random.Random(1000 + TRIAL)
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
    def compact_prefix(kpre, vals, sel_idx):
        W = len(sel_idx)
        idx = torch.tensor(sel_idx, device="cuda")
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
        c = DynamicCache()
        for li in range(len(kpre)):
            k = rc.rotate_k(kpre[li][:, :, idx, :], cos.float(), sin.float()).to(torch.bfloat16)
            c.update(k, vals[li][:, :, idx, :], li)
        return c, W

    @torch.no_grad()
    def turn(cache, pos, text_ids):
        o = model(input_ids=torch.tensor([text_ids], device="cuda"),
                  past_key_values=cache,
                  attention_mask=torch.ones((1, pos + len(text_ids)),
                                            device="cuda", dtype=torch.long))
        v = pick(o.logits[0, -1])
        del o
        torch.cuda.empty_cache()
        return v, pos + len(text_ids)

    a, b = rnd.sample(PLAYERS, 2)
    ga = rnd.randint(12, 20) if TRIAL % 2 == 0 else rnd.randint(0, 7)
    gb = rnd.randint(0, 7) if TRIAL % 2 == 0 else rnd.randint(12, 20)
    gold_cmp = 0 if ga > gb else 1

    S0 = "You are narrating a story. Answer each question with one letter.\n"
    A = "Story: %s trained at the northern keep and scored %d victories.\n" % (a, ga)
    B = "Story: %s trained at the southern keep and scored %d victories.\n" % (b, gb)
    Q1 = ("\nQuestion: who scored more victories?\n"
          " A) %s\n B) %s\n C) same\n D) unknown\nAnswer:" % (a, b))
    Q2 = ("\nQuestion: how many victories did %s have?\n"
          " A) %d\n B) %d\n C) %d\n D) %d\nAnswer:" % (a, ga, ga + 3, ga + 7, ga - 2))

    s0, aA, aB, q1, q2 = enc(S0), enc(A), enc(B), enc(Q1), enc(Q2)
    fid = fill(FILL)
    ids_full = s0 + aA + fid + aB + fid + q1
    pa0, pa1 = len(s0), len(s0) + len(aA)
    pb0 = pa1 + len(fid)
    pb1 = pb0 + len(aB)

    # ---- full ----
    o = model(input_ids=torch.tensor([ids_full], device="cuda"), use_cache=True)
    cf, posf = o.past_key_values, len(ids_full)
    v1f = pick(o.logits[0, -1])
    del o
    v2f, posf = turn(cf, posf, q2)
    del cf
    torch.cuda.empty_cache()

    # ---- compact（只做一次重紧凑）----
    kpre, vals = build_pool(ids_full[:-len(q1)])
    sel = list(range(0, len(s0))) + list(range(pa0, pa1)) + list(range(pb0, pb1))
    cc, W = compact_prefix(kpre, vals, sel)
    del kpre, vals
    torch.cuda.empty_cache()
    v1c, posc = turn(cc, W, q1)
    v2c, posc = turn(cc, posc, q2)

    print("RESULT trial=%d ctx=%d W=%d | cmp_full=%d cmp_compact=%d cnt_full=%d cnt_compact=%d"
          % (TRIAL, len(ids_full), W,
             int(v1f == gold_cmp), int(v1c == gold_cmp),
             int(v2f == 0), int(v2c == 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
