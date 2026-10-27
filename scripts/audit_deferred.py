# -*- coding: utf-8 -*-
"""时序实验：主智能体先回答 q2，再被通知子智能体已答 q1，然后回答 q1。

序列：
  S0 + 中性填充 + [槽位 L] + Q1(需要子智能体) + Q2(可立即回答) + "Answer Q2 first:"
  1) 生成 Q2 的答案            <- 此时槽位仍是占位
  2) 槽位替换为子智能体的 K/V
  3) prefill 通知："子智能体已回答 Q1，现在回答 Q1"
  4) 生成 Q1 的答案            <- 必须回到前面那个问题

三臂：full / swap / noswap
"""
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
L = 24
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
      "- The night shift reported nothing unusual.\n")
ENT, ATTR = "valve DANDAN-9171", "coolant code"
GOLD = "KARTHA-7346"
Q1 = "\nQ1: what is the %s for the %s?\n" % (ATTR, ENT)
Q2 = "Q2: when was the dock crane serviced?\n"
NOTICE = ("\n\n[System note] The sub-agent has now returned the answer to Q1. "
          "The result is shown above. Now answer Q1.\n")


@torch.no_grad()
def gen(model, tok, ids, past, n_new):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=past, use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    toks = []
    for _ in range(n_new):
        t = int(logits.argmax())
        if t == tok.eos_token_id:
            break
        toks.append(t)
        st = model(input_ids=torch.tensor([[t]], device="cuda"),
                   past_key_values=cache, use_cache=True)
        cache, logits = st.past_key_values, st.logits[0, -1]
    return toks, cache


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    def fit(text, n):
        ids = enc(text)
        while len(ids) < n:
            ids += enc(" .")
        return ids[:n]

    priv = ("Private log: the %s for %s is %s.\n" % (ATTR, ENT, GOLD)
            + "Example: Finding: The valve code for the pump ABC-1234 is XYZ-9999.\n"
            + "Now write the finding in one full sentence.\nFinding:")
    a_priv = enc(priv)
    Ftot = len(a_priv)
    aS0 = enc(S0)
    a_fill = fit("Log: all readings within tolerance, nothing to report.", Ftot)
    P = len(aS0) + Ftot

    sub_toks, sub_cache = gen(model, tok, aS0 + a_priv, None, L)
    sub_text = tok.decode(sub_toks, skip_special_tokens=True).strip()
    if norm(GOLD) not in norm(sub_text):
        print("子智能体没生成答案，退出")
        return 1
    n_cache = sub_cache.layers[0].keys.shape[-2]
    sl = slice(n_cache - len(sub_toks), n_cache)

    a_q1, a_q2 = enc(Q1), enc(Q2)
    a_head = enc("\nCoordinator: I will answer the easy question first.\nAnswer Q2:")
    a_notice = enc(NOTICE + "Answer Q1:")
    a_ph = fit(".... .... .... .... .... .... .... .... .", L)
    a_full = fit(sub_text, L)

    print("子智能体生成: %r" % sub_text)
    print("槽位区间 = [%d, %d)\n" % (P, P + L))

    def gen_from(mode, logits, cache, n_new):
        toks = []
        for _ in range(n_new):
            t = int(logits.argmax())
            if t == tok.eos_token_id:
                break
            toks.append(t)
            st = model(input_ids=torch.tensor([[t]], device="cuda"),
                       past_key_values=cache, use_cache=True)
            cache, logits = st.past_key_values, st.logits[0, -1]
        return toks, cache

    for mode in ("full", "swap", "noswap"):
        slot = a_full if mode == "full" else a_ph
        ids = aS0 + a_fill + slot + a_q1 + a_q2 + a_head
        out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache, lg = out.past_key_values, out.logits[0, -1]
        # 1) 先答 Q2
        t2, cache = gen_from(mode, lg, cache, 12)
        ans2 = tok.decode(t2, skip_special_tokens=True).strip()
        # 2) 替换槽位（模拟子智能体此时返回）
        if mode == "swap":
            rb = DynamicCache()
            for li in range(len(cache.layers)):
                k = cache.layers[li].keys.clone()
                v = cache.layers[li].values.clone()
                k[:, :, P:P + L, :] = sub_cache.layers[li].keys[:, :, sl, :]
                v[:, :, P:P + L, :] = sub_cache.layers[li].values[:, :, sl, :]
                rb.update(k, v, li)
            cache = rb
        # 3) 通知 + 答 Q1
        t1, _ = gen(model, tok, a_notice, cache, 30)
        ans1 = tok.decode(t1, skip_special_tokens=True).strip()
        print("【%s】" % mode)
        print("  Q2 答案: %r   %s" % (ans2[:50], "✔" if "monday" in norm(ans2) else "✘"))
        print("  Q1 答案: %r   %s" % (ans1[:70], "✔" if norm(GOLD) in norm(ans1) else "✘"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
