# -*- coding: utf-8 -*-
"""可行性测试：长上下文 prefill 一次后，decode 只关注部分块 + 重紧凑。

上下文结构（约 2K token，控制显存）：
    [块A: X 的进球数] + 填充 800 + [块B: Y 的进球数] + 填充 800 + 问题

三臂：
  full    全部可见，原位置                 —— 质量上限，成本最高
  gapped  只留 A、B，其余 mask，**原位置**   —— 有洞
  compact 只留 A、B，**重旋转成紧凑位置**    —— 本次要验的

判据：四选一闭集（比较 X 与 Y 的进球数）。
另记录：上下文 token 数、每臂耗时、显存峰值。
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

    @torch.no_grad()
    def run_full(ids, blocks):
        att = mk_mask(len(ids), blocks)
        out = model(input_ids=torch.tensor([ids], device="cuda"), attention_mask=att)
        lg = out.logits[0, -1].float()
        best = max(range(4), key=lambda i: float(lg[letters[i]]))
        del out, lg, att
        torch.cuda.empty_cache()
        return best

    @torch.no_grad()
    def run_compact(sel_ids, q_ids, q_pos0):
        """只对选中的 token 建缓存：K 用 pre-RoPE 重旋转到 0..W-1，V 原样。"""
        W = len(sel_ids)
        with rc.capture_rope() as cap:
            out = model(input_ids=torch.tensor([sel_ids], device="cuda"), use_cache=True)
        d = cap.pre[0].shape[-1]
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d)
        cache = out.past_key_values
        new = DynamicCache()
        for li in range(len(cache.layers)):
            k_pre = cap.pre[li].float()                     # fp32 旋转（bf16 会漂）
            k_rot = rc.rotate_k(k_pre, cos.float(), sin.float()).to(torch.bfloat16)
            v = cache.layers[li].values
            new.update(k_rot, v, li)
        # 问题从紧凑坐标 W 开始
        pos = list(range(q_pos0, q_pos0 + len(q_ids)))
        o = model(input_ids=torch.tensor([q_ids], device="cuda"),
                  past_key_values=new,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long),
                  position_ids=torch.tensor([pos], device="cuda"))
        lg = o.logits[0, -1].float()
        best = max(range(4), key=lambda i: float(lg[letters[i]]))
        del out, o, lg, cache, new
        torch.cuda.empty_cache()
        return best

    hit = {"full": 0, "gapped": 0, "compact": 0}
    t_cost = {"full": 0.0, "gapped": 0.0, "compact": 0.0}
    ctx_tokens = 0
    used = 0

    for t in range(N):
        a, b = rnd.sample(PLAYERS, 2)
        ga = rnd.choice([0, 1, 2, 3, 17, 18, 19, 20])
        gb = rnd.choice([0, 1, 2, 3, 17, 18, 19, 20])
        if abs(ga - gb) < 10:
            gb = 20 - ga if ga >= 10 else 20 - gb
        gold = 0 if ga > gb else 1

        S0 = "You read two records separated by logs, then answer with one letter.\n"
        A = "Record 1: %s scored %d goals.\n" % (a, ga)
        B = "Record 2: %s scored %d goals.\n" % (b, gb)
        Q = ("\nQuestion: did %s score more goals than %s?\n" % (a, b)
             + " A) yes\n B) no\n C) they scored the same\n D) cannot tell\nAnswer:")

        s0, aA, aB, aQ = enc(S0), enc(A), enc(B), enc(Q)
        fid = fill(FILL)
        ids = s0 + aA + fid + aB + fid + aQ
        ctx_tokens += len(ids)
        used += 1
        pa0, pa1 = len(s0), len(s0) + len(aA)
        pb0 = pa1 + len(fid)
        pb1 = pb0 + len(aB)
        # gapped：遮掉两段填充
        hide = [(pa1, pb0, 0, len(ids)), (pb1, pb1 + len(fid), 0, len(ids))]

        t0 = time.time(); v_full = run_full(ids, []); t_cost["full"] += time.time() - t0
        t0 = time.time(); v_gap = run_full(ids, hide); t_cost["gapped"] += time.time() - t0
        t0 = time.time()
        sel = s0 + aA + aB
        v_cmp = run_compact(sel, aQ, len(sel))
        t_cost["compact"] += time.time() - t0

        hit["full"] += int(v_full == gold)
        hit["gapped"] += int(v_gap == gold)
        hit["compact"] += int(v_cmp == gold)
        print("  %2d) ctx=%d %s=%d %s=%d | full=%s gap=%s cmp=%s"
              % (t + 1, len(ids), a, ga, b, gb,
                 "O" if v_full == gold else "X",
                 "O" if v_gap == gold else "X",
                 "O" if v_cmp == gold else "X"), flush=True)

    print("\nn=%d  平均上下文 %d token  填充 %d×2" % (used, ctx_tokens // max(used, 1), FILL))
    for k in ("full", "gapped", "compact"):
        print("  %-8s %2d/%d   平均耗时 %.3fs"
              % (k, hit[k], used, t_cost[k] / max(used, 1)))
    print("  峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
