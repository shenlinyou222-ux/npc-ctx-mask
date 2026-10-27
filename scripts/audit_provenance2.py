# -*- coding: utf-8 -*-
"""审计 3：来源消融（带生成过滤）+ 最短槽位长度。

每个私有编码重试直到子智能体产出合格句子（以 The 开头、含 is、含该编码），
然后看主智能体最终答的是哪个编码，以及别的编码有没有串进来。
"""
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
CUE = ("\n\n[System note] The sub-agent has finished its work and returned the "
       "result shown above. You may now use that result to answer the question.\nAnswer:")
S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
      "- The night shift reported nothing unusual.\n")
ENT, ATTR = "valve DANDAN-9171", "coolant code"
GOLDS = ["KARTHA-7346", "VELOSK-8091", "NYMKAR-4413"]


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


def attempt(tok, model, gold, L, variant):
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    def fit(text, n):
        ids = enc(text)
        while len(ids) < n:
            ids += enc(" .")
        return ids[:n]

    fact = "Private log: the %s for %s is %s.\n" % (ATTR, ENT, gold)
    ex = "Example: Finding: The valve code for the pump ABC-1234 is XYZ-9999.\n"
    tail = "Now write the finding in one full sentence.\nFinding:"
    priv = (fact + tail) if variant == 0 else (fact + ex + tail)
    a_priv = enc(priv)
    Ftot = len(a_priv)
    a_fill = fit("Log: all readings within tolerance, nothing to report.", Ftot)
    aS0 = enc(S0)
    P = len(aS0) + Ftot

    sub_toks, sub_cache = gen(model, tok, aS0 + a_priv, None, L)
    sub_text = tok.decode(sub_toks, skip_special_tokens=True).strip()
    good = (norm(gold) in norm(sub_text) and sub_text.lower().startswith("the ")
            and " is " in sub_text)
    n_cache = sub_cache.layers[0].keys.shape[-2]
    sl = slice(n_cache - len(sub_toks), n_cache)

    q = "\nQuestion: what is the %s for the %s? Answer with the sentence.\n" % (ATTR, ENT)
    wait = "Coordinator: I do not know this. I must wait for the sub-agent.\n<WAIT>"
    a_q, a_wait, a_cue = enc(q), enc(wait), enc(CUE)
    a_ph = fit(".... .... .... .... .... .... .... .... .", L)
    a_full = fit(sub_text, L)

    def run(mode):
        slot = a_full if mode == "full" else a_ph
        ids = aS0 + a_fill + slot + a_q + a_wait
        out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        if mode == "swap":
            rb = DynamicCache()
            for li in range(len(cache.layers)):
                k = cache.layers[li].keys.clone()
                v = cache.layers[li].values.clone()
                k[:, :, P:P + L, :] = sub_cache.layers[li].keys[:, :, sl, :]
                v[:, :, P:P + L, :] = sub_cache.layers[li].values[:, :, sl, :]
                rb.update(k, v, li)
            cache = rb
        toks, _ = gen(model, tok, a_cue, cache, 30)
        return tok.decode(toks, skip_special_tokens=True).strip()

    return good, sub_text, run("full"), run("swap")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    print("=== A. 来源消融（每例重试到生成合格，L=24）===")
    for gold in GOLDS:
        got = None
        for variant in (1, 0):
            for _ in range(12):
                good, sub, full, swap = attempt(tok, model, gold, 24, variant)
                if good:
                    got = (sub, full, swap)
                    break
            if got:
                break
        if not got:
            print("  私有日志=%s -> 子智能体始终没生成合格句子，跳过" % gold)
            continue
        sub, full, swap = got
        others = [g for g in GOLDS if g != gold]
        leaked = [g for g in others if norm(g) in norm(full) or norm(g) in norm(swap)]
        print("  私有日志=%s" % gold)
        print("    子生成 : %r" % sub)
        print("    full   : %s  %r" % ("✔" if norm(gold) in norm(full) else "✘", full[:64]))
        print("    swap   : %s  %r" % ("✔" if norm(gold) in norm(swap) else "✘", swap[:64]))
        print("    其它编码串入: %s" % (leaked or "无"))

    print("\n=== B. 最短可用槽位（句子 = 13 token）===")
    for L in (16, 20, 22, 24, 28):
        good, sub, full, swap = attempt(tok, model, "KARTHA-7346", L, 1)
        ok = norm("KARTHA-7346") in norm(swap)
        print("  L=%-3d 子生成=%r -> swap %s" % (L, sub[:50], "✔" if ok else "✘"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
