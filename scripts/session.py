# -*- coding: utf-8 -*-
"""一次性重紧凑 + 之后正常多轮对话（角色扮演形态）。

流程：
  ① 建池：对长上下文（约 1.5K token，含多段设定 + 无关填充）做一次 prefill
  ② 开会话：只挑出相关块，重紧凑成 W token 的**前缀**（只做这一次）
  ③ 之后：多轮问答，全部在紧凑前缀上正常追加 decode，不再做任何重排

对照：
  full     保留完整长上下文，正常多轮对话
  compact  紧凑前缀，正常多轮对话

指标：每轮答案准确率、每轮耗时、紧凑前缀长度。
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FILL = 400
rnd = random.Random(20261106)
PLAYERS = ["Lindqvist", "Okonkwo", "Ferreira", "Nakamura"]
FILLW = "The archive index lists routine entries with no bearing on the story. "


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
        """正常追加一轮：把 text 接到缓存末尾，取答案。"""
        o = model(input_ids=torch.tensor([text_ids], device="cuda"),
                  past_key_values=cache,
                  attention_mask=torch.ones((1, pos + len(text_ids)), device="cuda", dtype=torch.long))
        v = pick(o.logits[0, -1])
        del o
        torch.cuda.empty_cache()
        return v, pos + len(text_ids), None

    hit = {"full": 0, "compact": 0}
    cost = {"full": 0.0, "compact": 0.0}
    used = 0
    Ws = []

    for t in range(N):
        a, b = rnd.sample(PLAYERS, 2)
        ga = rnd.randint(12, 20) if t % 2 == 0 else rnd.randint(0, 7)
        gb = rnd.randint(0, 7) if t % 2 == 0 else rnd.randint(12, 20)
        gold_cmp = 0 if ga > gb else 1

        S0 = "You are narrating a story. Answer each question with one letter.\n"
        A = "Story: %s trained at the northern keep and scored %d victories.\n" % (a, ga)
        B = "Story: %s trained at the southern keep and scored %d victories.\n" % (b, gb)
        Q1 = ("\nQuestion: who scored more victories?\n"
              " A) %s\n B) %s\n C) same\n D) unknown\nAnswer:" % (a, b))
        Q2 = "\nQuestion: how many victories did %s have?\n A) %d\n B) %d\n C) %d\n D) %d\nAnswer:" % (
            a, ga, ga + 3, ga + 7, ga - 2)
        gold2 = 0

        s0, aA, aB, q1, q2 = enc(S0), enc(A), enc(B), enc(Q1), enc(Q2)
        fid = fill(FILL)
        ids = s0 + aA + fid + aB + fid + q1
        pa0, pa1 = len(s0), len(s0) + len(aA)
        pb0 = pa1 + len(fid)
        pb1 = pb0 + len(aB)
        used += 1

        # ---- full：完整长上下文，两轮 ----
        t0 = time.time()
        o = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache_f, pos_f = o.past_key_values, len(ids)
        v1f = pick(o.logits[0, -1])
        del o
        v2f, pos_f, _ = turn(cache_f, pos_f, q2)
        cost["full"] += time.time() - t0
        del cache_f
        torch.cuda.empty_cache()

        # ---- compact：一次重紧凑，然后两轮正常追加 ----
        t0 = time.time()
        kpre, vals = build_pool(ids[:-len(q1)])          # 池不含第一问
        sel = list(range(0, len(s0))) + list(range(pa0, pa1)) + list(range(pb0, pb1))
        cache_c, W = compact_prefix(kpre, vals, sel)
        del kpre, vals
        torch.cuda.empty_cache()
        Ws.append(W)
        v1c, pos_c, _ = turn(cache_c, W, q1)
        v2c, pos_c, _ = turn(cache_c, pos_c, q2)
        cost["compact"] += time.time() - t0
        del cache_c
        torch.cuda.empty_cache()

        hit["full"] += int(v1f == gold_cmp) + int(v2f == gold2)
        hit["compact"] += int(v1c == gold_cmp) + int(v2c == gold2)
        print("  %2d) ctx=%d W=%d | 比较 full=%s cmp=%s | 计数 full=%s cmp=%s"
              % (t + 1, len(ids), W,
                 "O" if v1f == gold_cmp else "X", "O" if v1c == gold_cmp else "X",
                 "O" if v2f == gold2 else "X", "O" if v2c == gold2 else "X"), flush=True)

    print("\nn=%d  两轮共 %d 题" % (used, used * 2))
    for k in ("full", "compact"):
        print("  %-8s %2d/%d   平均 %.3fs（含建池/紧凑化）" % (k, hit[k], used * 2, cost[k] / used))
    print("  紧凑前缀平均 %d token（原上下文约 %d）" % (sum(Ws) // len(Ws), len(ids)))
    print("  峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
