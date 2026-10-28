# -*- coding: utf-8 -*-
"""长历史测试：历史事实从 4 条加到 64 条，槽位固定 3 个并回填。

每档：
  历史 H 条事实 + 3 个槽位（回填）+ 逐题单问
  问 6 道历史题（从 H 条里随机抽）+ 3 道槽位题
记录：上下文 token 数、历史准确率、槽位准确率
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
L = 32
NS = 3
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
HIST = [4, 16, 32, 64]
rnd = random.Random(20261104)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine", "sensor", "coolant", "boiler", "condenser", "tank"]


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

    stat = {h: {"hh": 0, "hd": 0, "sh": 0, "sd": 0, "toks": 0} for h in HIST}

    for trial in range(N):
        # 生成足够多的历史事实
        labels = ["label%d" % i for i in range(max(HIST))]
        values = [code() for _ in labels]

        ents = ["%s %s" % (rnd.choice(KIND), code()) for _ in range(NS)]
        golds = [code() for _ in range(NS)]
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

        for hn in HIST:
            facts = "".join("- The %s is %s.\n" % (labels[i], values[i]) for i in range(hn))
            hist = "Project history:\n" + facts
            ids = list(enc(hist))
            for k in range(NS):
                ids += fit("Log: all readings within tolerance. ", 12)
                ids += fit(answers[k], L)
            stat[hn]["toks"] += len(ids)

            def ask(qtext, gold):
                full = ids + enc("\nQuestion: what is the %s?\nAnswer:" % qtext)
                st = model.generate(
                    input_ids=torch.tensor([full], device="cuda"),
                    do_sample=False, max_new_tokens=18,
                    attention_mask=torch.ones((1, len(full)), device="cuda", dtype=torch.long))
                txt = tok.decode(st[0][len(full):], skip_special_tokens=True).strip()
                return norm(gold) in norm(txt)

            picks = rnd.sample(range(hn), min(6, hn))
            for i in picks:
                stat[hn]["hh"] += int(ask(labels[i], values[i]))
                stat[hn]["hd"] += 1
            for k in range(NS):
                stat[hn]["sh"] += int(ask("access code for the %s" % ents[k], golds[k]))
                stat[hn]["sd"] += 1
            print("  trial %d hist=%d (%d tok) 完成" % (trial + 1, hn, len(ids)), flush=True)

    print("\nn=%d trials, NS=%d" % (N, NS))
    print("  历史条数  平均 token   历史准确率   槽位准确率")
    for hn in HIST:
        s = stat[hn]
        avg = s["toks"] / max(N, 1)
        print("  %-8d %-11.0f %-12s %s"
              % (hn, avg, "%d/%d" % (s["hh"], s["hd"]), "%d/%d" % (s["sh"], s["sd"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
