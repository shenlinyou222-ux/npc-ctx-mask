# -*- coding: utf-8 -*-
"""回填槽位是否损伤对「历史」的记忆。

上下文结构：
    S0（历史事实 ×5） + [槽1][槽2][槽3] + 提示 + 问题

三臂：
    noslot   完全没有槽位区
    empty    有槽位但都不填（占位符）
    filled   槽位回填子智能体的答案

两类问题各问一遍：
    H  历史题（答案在 S0 里，与槽位无关）
    S  槽位题（答案在槽位里）
若 filled 的 H 明显低于 noslot，说明回填挤掉了历史。
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
L = 32
NS = 3
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rnd = random.Random(20261104)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine"]


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

    hit = {("noslot", "H"): 0, ("empty", "H"): 0, ("filled", "H"): 0,
           ("filled", "S"): 0, ("empty", "S"): 0}
    den = {("noslot", "H"): 0, ("empty", "H"): 0, ("filled", "H"): 0,
           ("filled", "S"): 0, ("empty", "S"): 0}
    used, tries = 0, 0
    while used < N and tries < N * 40:
        tries += 1
        owner, room, phase = code(), code(), code()
        hist = ("Project history:\n"
                "- The project owner is %s.\n"
                "- The meeting room is %s.\n"
                "- The current phase is %s.\n"
                "- The dock crane was serviced on Monday.\n"
                "- The night shift reported nothing unusual.\n" % (owner, room, phase))
        ents = ["%s %s" % (rnd.choice(KIND), code()) for _ in range(NS)]
        golds = [code() for _ in range(NS)]

        answers = []
        ok = True
        for k in range(NS):
            priv = ("Private log: the access code for the %s is %s.\n"
                    "Write one full sentence giving the finding.\nFinding:" % (ents[k], golds[k]))
            toks, _ = gen(model, tok, enc(priv), None, 30)
            text = tok.decode(toks, skip_special_tokens=True).strip()
            if not (norm(golds[k]) in norm(text) and text.lower().startswith("the ")):
                ok = False
                break
            answers.append(text)
        if not ok:
            continue
        used += 1

        a_hist = enc(hist)
        fill_unit = "Log: all readings within tolerance. "

        def build(arm):
            ids = list(a_hist)
            if arm != "noslot":
                for k in range(NS):
                    ids += fit(fill_unit, 12)
                    if arm == "filled":
                        ids += fit(answers[k], L)
                    else:
                        ids += fit(".... .... .... .... .... .... ....", L)
            return ids

        hqs = [("project owner", owner), ("meeting room", room), ("current phase", phase)]
        sqs = [("access code for the %s" % ents[k], golds[k]) for k in range(NS)]

        def ask(arm, qtext, gold):
            ids = build(arm) + enc("\nQuestion: what is the %s?\nAnswer:" % qtext)
            st = model.generate(
                input_ids=torch.tensor([ids], device="cuda"),
                do_sample=False, max_new_tokens=18,
                attention_mask=torch.ones((1, len(ids)), device="cuda", dtype=torch.long))
            txt = tok.decode(st[0][len(ids):], skip_special_tokens=True).strip()
            return norm(gold) in norm(txt), txt

        for q, g in hqs:
            for arm in ("noslot", "empty", "filled"):
                okh, _ = ask(arm, q, g)
                hit[(arm, "H")] += int(okh)
                den[(arm, "H")] += 1
        for q, g in sqs:
            for arm in ("empty", "filled"):
                oks, _ = ask(arm, q, g)
                hit[(arm, "S")] += int(oks)
                den[(arm, "S")] += 1

        print("  %d) owner=%s room=%s phase=%s | 累计 filled-H=%d noslot-H=%d filled-S=%d"
              % (used, owner, room, phase,
                 hit[("filled", "H")], hit[("noslot", "H")], hit[("filled", "S")]),
              flush=True)

    print("\nn=%d questions 满分: H 每臂 %d, S 每臂 %d"
          % (used, den[("noslot", "H")], den[("filled", "S")]))
    for k in [("noslot", "H"), ("empty", "H"), ("filled", "H"), ("empty", "S"), ("filled", "S")]:
        print("  %-6s-%-1s  %d/%d" % (k[0], k[1], hit[k], den[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
