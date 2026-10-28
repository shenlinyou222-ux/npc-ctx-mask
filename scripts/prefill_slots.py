# -*- coding: utf-8 -*-
"""多槽位：子智能体各自回答 -> 回到槽位位置 prefill 答案 -> 解除掩码。

流程：
  ① 子智能体 k 在自己的私有上下文里生成结论（正常生成，无位置技巧）
  ② 把结论文本 prefill 到主序列的槽位 [P_k, P_k+L)
  ③ 解除掩码，主智能体回答

三臂：
  filled   所有槽位都填
  empty    都不填（占位符）
  only2    只填第 2 个
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    hit = {"filled": 0, "empty": 0, "only2": 0}
    used, tries = 0, 0
    while used < N and tries < N * 40:
        tries += 1
        ents = ["%s %s" % (rnd.choice(KIND), code()) for _ in range(NS)]
        golds = [code() for _ in range(NS)]
        S0 = ("Records:\n- The dock crane was serviced on Monday.\n"
              "- The night shift reported nothing unusual.\n")
        task = "Write one full sentence giving the finding.\nFinding:"

        # ① 子智能体各自在私有上下文里回答
        answers = []
        ok_all = True
        for k in range(NS):
            priv = ("Private log: the access code for the %s is %s.\n" % (ents[k], golds[k])) \
                + task
            toks, _ = gen(model, tok, enc(priv), None, 30)
            text = tok.decode(toks, skip_special_tokens=True).strip()
            if not (norm(golds[k]) in norm(text) and text.lower().startswith("the ")):
                if tries <= 3:
                    print("  [debug try %d k=%d] gold=%s 生成=%r"
                          % (tries, k, golds[k], text[:80]), flush=True)
                ok_all = False
                break
            answers.append(text)
        if not ok_all:
            continue
        used += 1

        # ② 主序列：槽位位置固定，把答案 prefill 进去
        head = S0
        a_head = enc(head)
        q = "\nQuestions:\n" + "".join(
            "Q%d: what is the access code for the %s?\n" % (k + 1, ents[k]) for k in range(NS))
        fill_unit = "Log: all readings within tolerance, nothing to report. "

        def build(mode):
            ids = list(a_head)
            for k in range(NS):
                ids += fit(fill_unit, 16)
                if mode == "filled" or (mode == "only2" and k == 1):
                    ids += fit(answers[k], L)
                else:
                    ids += fit(".... .... .... .... .... .... ....", L)
            ids += enc(q) + enc("\nCoordinator: results are in.\n<WAIT>")
            return ids

        def run(mode):
            ids = build(mode)
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            toks, _ = gen(model, tok, enc(CUE), out.past_key_values, 24 * NS + 20)
            return tok.decode(toks, skip_special_tokens=True).strip()

        r = {m: run(m) for m in ("filled", "empty", "only2")}
        got = {m: sum(1 for g in golds if norm(g) in norm(r[m])) for m in r}
        for m in hit:
            hit[m] += got[m]

        print("=" * 70)
        print("gold=%s" % golds)
        for k in range(NS):
            print("  子智能体%d: %s" % (k + 1, answers[k][:78]))
        print("  命中 filled=%d empty=%d only2=%d" % (got["filled"], got["empty"], got["only2"]))
        for m in ("filled", "empty", "only2"):
            print("    %-7s %s" % (m, r[m][:78].replace("\n", " ")), flush=True)

    den = used * NS
    print("\nNS=%d n=%d L=%d tries=%d 满分=%d" % (NS, used, L, tries, den))
    for m in ("filled", "empty", "only2"):
        print("  %-7s %d/%d" % (m, hit[m], den))
    return 0


if __name__ == "__main__":
    sys.exit(main())
