# -*- coding: utf-8 -*-
"""按真实顺序验证：答案只对「槽位之前的缓存」prefill，后缀不重算。

两种做法对比（单槽位）：
  inline       整条序列重算（答案内联）              —— 我之前做的，偏乐观
  incremental  前缀+空槽+后缀 + cue 先算好            —— 真实顺序
               然后答案 token 只 attend 前缀 [0,P) 生成 K/V
               把这段 K/V 放进 [P, P+len)，后缀不动
               从 cue 之后继续 decode

若 incremental 与 inline 持平，说明"延迟写入"在真实顺序下也成立。
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
L = 32
rnd = random.Random(20261104)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine"]

CUE = ("\n\n[System note] The sub-agent has finished and returned the result above. "
       "You may now use it.\nAnswer:")


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

    hit = {"inline": 0, "incremental": 0, "empty": 0}
    used, tries = 0, 0
    while used < N and tries < N * 30:
        tries += 1
        ent = "%s %s" % (rnd.choice(KIND), code())
        gold = code()
        S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
              "- The night shift reported nothing unusual.\n")
        priv = ("Private log: the access code for the %s is %s.\n"
                "Write one full sentence giving the finding.\nFinding:" % (ent, gold))
        toks, _ = gen(model, tok, enc(priv), None, 30)
        ans_text = tok.decode(toks, skip_special_tokens=True).strip()
        if not (norm(gold) in norm(ans_text) and ans_text.lower().startswith("the ")):
            continue
        used += 1

        q = "\nQuestion: what is the access code for the %s?\nAnswer:" % ent
        aS0 = enc(S0)
        a_fill = fit("Log: all readings within tolerance. ", 12)
        a_ph = fit(".... .... .... .... .... .... .", L)
        a_ans = fit(ans_text, L)
        a_q = enc(q)
        a_cue = enc(CUE)
        P = len(aS0) + len(a_fill)          # 槽位起点

        def run_inline(use_ans):
            ids = list(aS0) + a_fill + (a_ans if use_ans else a_ph) + a_q
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            t, _ = gen(model, tok, a_cue, out.past_key_values, 40)
            return tok.decode(t, skip_special_tokens=True).strip()

        def run_incremental():
            # ① 前缀 + 空槽 + 问题，全部先算好（此时答案还没到）
            ids = list(aS0) + a_fill + a_ph + a_q
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            cache = out.past_key_values
            # ② 答案 token 只 attend [0, P) 的缓存
            rb = DynamicCache()
            for li in range(len(cache.layers)):
                k = cache.layers[li].keys[:, :, :P, :].clone()
                v = cache.layers[li].values[:, :, :P, :].clone()
                rb.update(k, v, li)
            ans_out = model(input_ids=torch.tensor([a_ans], device="cuda"),
                            past_key_values=rb, use_cache=True)
            # ③ 把答案这段 K/V 放回 [P, P+len)，后缀 K/V 不动
            n = ans_out.past_key_values.layers[0].keys.shape[-2]
            ln = n - P
            fin = DynamicCache()
            for li in range(len(cache.layers)):
                k = cache.layers[li].keys.clone()
                v = cache.layers[li].values.clone()
                k[:, :, P:P + ln, :] = ans_out.past_key_values.layers[li].keys[:, :, P:, :]
                v[:, :, P:P + ln, :] = ans_out.past_key_values.layers[li].values[:, :, P:, :]
                fin.update(k, v, li)
            t, _ = gen(model, tok, a_cue, fin, 40)
            return tok.decode(t, skip_special_tokens=True).strip()

        r = {"inline": run_inline(True),
             "empty": run_inline(False),
             "incremental": run_incremental()}
        for m in hit:
            hit[m] += int(norm(gold) in norm(r[m]))
        print("=" * 66)
        print("答案(%d tok): %s | gold=%s" % (len(a_ans), ans_text[:60], gold))
        for m in ("inline", "incremental", "empty"):
            mark = "✔" if norm(gold) in norm(r[m]) else "✘"
            print("  %-12s %s %s" % (m, mark, r[m][:70].replace("\n", " ")), flush=True)

    print("\nn=%d tries=%d" % (used, tries))
    for m in ("inline", "incremental", "empty"):
        print("  %-12s %d/%d" % (m, hit[m], used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
