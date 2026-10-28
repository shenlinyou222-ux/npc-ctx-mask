# -*- coding: utf-8 -*-
"""多槽位：答案重新 prefill 成 K/V，换回空白掩码块。

流程（每个槽位独立）：
  ① 主序列：前缀 + [空槽1][空槽2]… + 问题 —— 先算好，K/V 进缓存
  ② 子智能体得到答案（文本）
  ③ 答案 token 只对 [0, P_k) 的缓存做 prefill，得到 [P_k, P_k+len) 的 K/V
  ④ 把这段 K/V 换进空槽
  ⑤ 解除掩码，之后 decode

等价于"当时询问就立刻得到答案"。
三臂：swap（③④）/ inline（答案直接内联，整条重算）/ empty（不填）
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
L = 32
rnd = random.Random(20261104)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine", "sensor", "coolant"]

CUE = ("\n\n[System note] The sub-agents have finished and returned their results above. "
       "You may now use them.\nAnswers:")


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

    def truncate(cache, upto):
        rb = DynamicCache()
        for li in range(len(cache.layers)):
            rb.update(cache.layers[li].keys[:, :, :upto, :].clone(),
                      cache.layers[li].values[:, :, :upto, :].clone(), li)
        return rb

    hit = {"swap": 0, "inline": 0, "empty": 0}
    used, tries = 0, 0
    while used < N and tries < N * 40:
        tries += 1
        ents = ["%s %s" % (rnd.choice(KIND), code()) for _ in range(NS)]
        golds = [code() for _ in range(NS)]
        S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
              "- The night shift reported nothing unusual.\n")

        answers, ok = [], True
        for k in range(NS):
            priv = ("Private log: the access code for the %s is %s.\n"
                    "Write one full sentence giving the finding.\nFinding:"
                    % (ents[k], golds[k]))
            toks, _ = gen(model, tok, enc(priv), None, 30)
            text = tok.decode(toks, skip_special_tokens=True).strip()
            if not (norm(golds[k]) in norm(text) and text.lower().startswith("the ")):
                ok = False
                break
            answers.append(text)
        if not ok:
            continue
        used += 1

        aS0 = enc(S0)
        fill_unit = "Log: all readings within tolerance. "
        F = 16
        q = "\nQuestions:\n" + "".join(
            "Q%d: what is the access code for the %s?\n" % (k + 1, ents[k]) for k in range(NS))
        a_ph = fit(".... .... .... .... .... .... .", L)
        a_q = enc(q)
        a_cue = enc(CUE)

        # 槽位位置
        P, cur = [], len(aS0)
        for k in range(NS):
            cur += F
            P.append(cur)
            cur += L

        def main_ids(use_inline):
            ids = list(aS0)
            for k in range(NS):
                ids += fit(fill_unit, F)
                ids += fit(answers[k], L) if use_inline else a_ph
            return ids + a_q

        def run(mode):
            ids = main_ids(mode == "inline")
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            cache = out.past_key_values
            if mode == "swap":
                cur_cache = cache
                for k in range(NS):
                    pre = truncate(cur_cache, P[k])
                    a_ans = fit(answers[k], L)
                    ao = model(input_ids=torch.tensor([a_ans], device="cuda"),
                               past_key_values=pre, use_cache=True)
                    ln = ao.past_key_values.layers[0].keys.shape[-2] - P[k]
                    rb = DynamicCache()
                    for li in range(len(cur_cache.layers)):
                        kk = cur_cache.layers[li].keys.clone()
                        vv = cur_cache.layers[li].values.clone()
                        kk[:, :, P[k]:P[k] + ln, :] = \
                            ao.past_key_values.layers[li].keys[:, :, P[k]:, :]
                        vv[:, :, P[k]:P[k] + ln, :] = \
                            ao.past_key_values.layers[li].values[:, :, P[k]:, :]
                        rb.update(kk, vv, li)
                    cur_cache = rb
                cache = cur_cache
            t, _ = gen(model, tok, a_cue, cache, 24 * NS + 20)
            return tok.decode(t, skip_special_tokens=True).strip()

        r = {m: run(m) for m in ("swap", "inline", "empty")}
        got = {m: sum(1 for g in golds if norm(g) in norm(r[m])) for m in r}
        for m in hit:
            hit[m] += got[m]
        print("=" * 70)
        print("NS=%d P=%s gold=%s" % (NS, P, golds))
        for k in range(NS):
            print("  答案%d: %s" % (k + 1, answers[k][:66]))
        print("  命中 swap=%d inline=%d empty=%d" % (got["swap"], got["inline"], got["empty"]))
        for m in ("swap", "inline", "empty"):
            print("    %-6s %s" % (m, r[m][:70].replace("\n", " ")), flush=True)

    den = used * NS
    print("\nNS=%d n=%d L=%d tries=%d 满分=%d" % (NS, used, L, tries, den))
    for m in ("swap", "inline", "empty"):
        print("  %-6s %d/%d" % (m, hit[m], den))
    return 0


if __name__ == "__main__":
    sys.exit(main())
