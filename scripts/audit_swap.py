# -*- coding: utf-8 -*-
"""审计：swap 臂到底是不是把「那句话的 K/V」换进了主序列？

检查项：
  A. 位置对齐：子智能体生成 token 的位置 == 槽位区间
  B. 负对照1：注入随机 K/V（同位置同形状）-> 必须失败
  C. 负对照2：注入另一条样本的 K/V -> 必须失败
  D. 分布一致性：full（槽位放真文本）vs swap（槽位放真 K/V）的下一词分布
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
L = 24
rnd = random.Random(7)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine", "sensor", "coolant"]
CUE = ("\n\n[System note] The sub-agent has finished its work and returned the "
       "result shown above. You may now use that result to answer the question.\nAnswer:")


@torch.no_grad()
def gen(model, tok, ids, past, n_new):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=past, use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    toks, all_logits = [], []
    for _ in range(n_new):
        all_logits.append(logits.float().clone())
        t = int(logits.argmax())
        if t == tok.eos_token_id:
            break
        toks.append(t)
        st = model(input_ids=torch.tensor([[t]], device="cuda"),
                   past_key_values=cache, use_cache=True)
        cache, logits = st.past_key_values, st.logits[0, -1]
    return toks, cache, all_logits


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build(tok, model, ent, attr, gold, other_cache=None):
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    def fit(text, n):
        ids = enc(text)
        while len(ids) < n:
            ids += enc(" .")
        return ids[:n]

    S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
          "- The night shift reported nothing unusual.\n")
    priv = ("Private log: the %s for %s is %s.\n" % (attr, ent, gold)
            + "Example: Finding: The valve code for the pump ABC-1234 is XYZ-9999.\n"
            + "Now write the finding in one full sentence.\nFinding:")
    a_priv = enc(priv)
    Ftot = len(a_priv)
    a_fill = fit("Log: all readings within tolerance, nothing to report.", Ftot)
    aS0 = enc(S0)
    P = len(aS0) + Ftot

    sub_toks, sub_cache, _ = gen(model, tok, aS0 + a_priv, None, L)
    sub_text = tok.decode(sub_toks, skip_special_tokens=True).strip()
    n_cache = sub_cache.layers[0].keys.shape[-2]
    sl = slice(n_cache - len(sub_toks), n_cache)

    q = ("\nQuestion: what is the %s for the %s? Answer with the sentence.\n" % (attr, ent))
    wait = "Coordinator: I do not know this. I must wait for the sub-agent.\n<WAIT>"
    a_q, a_wait, a_cue = enc(q), enc(wait), enc(CUE)
    a_ph = fit(".... .... .... .... .... .... .... .... .", L)
    a_full = fit(sub_text, L)
    return dict(S0=S0, priv=priv, sub_text=sub_text, sub_toks=sub_toks,
                sub_cache=sub_cache, sl=sl, aS0=aS0, a_fill=a_fill, a_ph=a_ph,
                a_full=a_full, a_q=a_q, a_wait=a_wait, a_cue=a_cue, P=P, Ftot=Ftot, q=q)


@torch.no_grad()
def run_arm(model, tok, d, mode, other_cache=None):
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    slot = d["a_full"] if mode == "full" else d["a_ph"]
    ids = d["aS0"] + d["a_fill"] + slot + d["a_q"] + d["a_wait"]
    out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
    cache = out.past_key_values
    P, L_ = d["P"], L

    if mode in ("swap", "swap_rand", "swap_other"):
        rb = DynamicCache()
        for li in range(len(cache.layers)):
            k = cache.layers[li].keys.clone()
            v = cache.layers[li].values.clone()
            if mode == "swap":
                kk = d["sub_cache"].layers[li].keys[:, :, d["sl"], :]
                vv = d["sub_cache"].layers[li].values[:, :, d["sl"], :]
            elif mode == "swap_rand":
                kk = torch.randn_like(k[:, :, P:P + L_, :]) * 2.0
                vv = torch.randn_like(v[:, :, P:P + L_, :]) * 2.0
            else:
                kk = other_cache.layers[li].keys[:, :, other_cache_sl[0], :]
                vv = other_cache.layers[li].values[:, :, other_cache_sl[0], :]
            k[:, :, P:P + L_, :] = kk
            v[:, :, P:P + L_, :] = vv
            rb.update(k, v, li)
        cache = rb

    toks, _, logits = gen(model, tok, d["a_cue"], cache, 30)
    return tok.decode(toks, skip_special_tokens=True).strip(), logits


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    ent, attr, gold = "valve DANDAN-9171", "coolant code", "KARTHA-7346"
    d = build(tok, model, ent, attr, gold)
    print("S0 = %d token, 私有段 = %d token, 槽位起点 P = %d, 槽位长度 L = %d"
          % (len(d["aS0"]), d["Ftot"], d["P"], L))
    print("子智能体生成 %d token: %r" % (len(d["sub_toks"]), d["sub_text"]))
    print("子序列长度 = %d，取最后 %d 个位置注入"
          % (d["sub_cache"].layers[0].keys.shape[-2], len(d["sub_toks"])))
    print("=> 子生成位置区间 = [%d, %d)，主槽位区间 = [%d, %d)  对齐=%s\n"
          % (d["sub_cache"].layers[0].keys.shape[-2] - len(d["sub_toks"]),
             d["sub_cache"].layers[0].keys.shape[-2], d["P"], d["P"] + L,
             d["sub_cache"].layers[0].keys.shape[-2] - len(d["sub_toks"]) == d["P"]))

    res = {}
    for mode in ("full", "swap", "swap_rand", "noswap"):
        txt, lg = run_arm(model, tok, d, mode)
        res[mode] = (txt, lg)
        print("  %-10s %s  %s" % (mode,
                                  "✔" if norm(gold) in norm(txt) else "✘",
                                  txt[:95].replace("\n", " ")))

    # 分布一致性：full vs swap 的第一个生成位置
    a = res["full"][1][0]
    b = res["swap"][1][0]
    p = torch.softmax(a, -1)
    qq = torch.softmax(b, -1)
    kl = float((p * (p.clamp_min(1e-12).log() - qq.clamp_min(1e-12).log())).sum())
    top1 = int(a.argmax()) == int(b.argmax())
    top5 = len(set(torch.topk(a, 5).indices.tolist()) & set(torch.topk(b, 5).indices.tolist()))
    print("\nfull vs swap：KL=%.5f  top1一致=%s  top5重合=%d/5" % (kl, top1, top5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
