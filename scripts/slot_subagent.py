# -*- coding: utf-8 -*-
"""子智能体实验 v4：子智能体必须产出「完整陈述句」，再拼回主序列。

子提示（私有段，长度与主提示的中性填充等长）：
    Private log: the <attr> for <ent> is <gold>.
    Example: Finding: The valve code for the pump ABC-1234 is XYZ-9999.
    Now write the finding in one full sentence.
    Finding:
子智能体从这里续写 -> 期望产出 "The <attr> for <ent> is <gold>."

主提示：
    S0 + 中性填充(等长) + [槽位 L 个占位] + 问题 + "Coordinator: ... <WAIT>"

三臂：full（把生成的整句放进槽位）/ swap（换成生成的 K/V）/ noswap
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
L = int(sys.argv[2]) if len(sys.argv) > 2 else 24
rnd = random.Random(20261104)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine", "sensor", "coolant"]

CUE = ("\n\n[System note] The sub-agent has finished its work and returned the "
       "result shown above. You may now use that result to answer the question.\nAnswer:")


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


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

    hit = {"full": 0, "swap": 0, "noswap": 0}
    used, tries = 0, 0
    while used < N and tries < N * 30:
        tries += 1
        ent = "%s %s" % (rnd.choice(KIND), code())
        attr = "%s code" % rnd.choice(KIND)
        gold = code()
        S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
              "- The night shift reported nothing unusual.\n")
        example = ("Example: Finding: The valve code for the pump ABC-1234 is XYZ-9999.\n")
        task = "Now write the finding in one full sentence.\nFinding:"
        priv = ("Private log: the %s for %s is %s.\n" % (attr, ent, gold)) + example + task
        a_priv = enc(priv)
        Ftot = len(a_priv)
        a_fill = fit("Log: all readings within tolerance, nothing to report.", Ftot)
        aS0 = enc(S0)
        P = len(aS0) + Ftot

        sub_toks, sub_cache = gen(model, tok, aS0 + a_priv, None, L)
        sub_text = tok.decode(sub_toks, skip_special_tokens=True).strip()
        ok_sentence = (norm(gold) in norm(sub_text)
                       and sub_text.lower().startswith("the ")
                       and " is " in sub_text)
        if not ok_sentence:
            continue
        used += 1
        n_cache = sub_cache.layers[0].keys.shape[-2]
        sl = slice(n_cache - len(sub_toks), n_cache)

        q = ("\nQuestion: what is the %s for the %s? Answer with the sentence.\n" % (attr, ent))
        wait = "Coordinator: I do not know this. I must wait for the sub-agent.\n<WAIT>"
        a_q, a_wait, a_cue = enc(q), enc(wait), enc(CUE)
        a_ph = fit(".... .... .... .... .... .... .... .... .", L)
        a_full = fit(sub_toks and tok.decode(sub_toks, skip_special_tokens=True), L)

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
            toks, _ = gen(model, tok, a_cue, cache, 40)
            return tok.decode(toks, skip_special_tokens=True).strip()

        r = {m: run(m) for m in ("full", "swap", "noswap")}
        for m in hit:
            hit[m] += int(norm(gold) in norm(r[m]))

        print("=" * 70)
        print("【共享前缀 S0】\n" + S0.rstrip())
        print("【子智能体私有段（%d token）】\n%s" % (Ftot, priv.rstrip()))
        print("【子智能体生成（%d token）】\n  %s" % (len(sub_toks), sub_text))
        print("【主智能体提问】\n  %s" % q.strip().replace("\n", " "))
        print("【槽位内容】%d token" % L)
        print("【三臂输出】")
        for m in ("full", "swap", "noswap"):
            mark = "✔" if norm(gold) in norm(r[m]) else "✘"
            print("  %-7s %s %s" % (m, mark, r[m][:110].replace("\n", " ")))
        print("【gold】%s" % gold, flush=True)

    print("\nn=%d L=%d tries=%d" % (used, L, tries))
    for m in ("full", "swap", "noswap"):
        print("  %-7s %d/%d" % (m, hit[m], used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
