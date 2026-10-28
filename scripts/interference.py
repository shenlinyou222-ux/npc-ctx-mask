# -*- coding: utf-8 -*-
"""干扰曲线：固定历史，槽位数从 0 增到 8，看历史记忆什么时候开始掉。

每个 (NS, trial)：
  历史 4 条事实 + NS 个槽位（都回填子智能体答案）
  问 4 道历史题 + NS 道槽位题（逐题单问）
输出：历史准确率 vs NS，槽位准确率 vs NS
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
L = 32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NS_LIST = [0, 1, 3, 5, 8]
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

    hist_hit = {ns: 0 for ns in NS_LIST}
    hist_den = {ns: 0 for ns in NS_LIST}
    slot_hit = {ns: 0 for ns in NS_LIST}
    slot_den = {ns: 0 for ns in NS_LIST}

    for trial in range(N):
        owner, room, phase, lead = code(), code(), code(), code()
        hist = ("Project history:\n"
                "- The project owner is %s.\n"
                "- The meeting room is %s.\n"
                "- The current phase is %s.\n"
                "- The team lead is %s.\n" % (owner, room, phase, lead))
        hqs = [("project owner", owner), ("meeting room", room),
               ("current phase", phase), ("team lead", lead)]

        for ns in NS_LIST:
            ents = ["%s %s" % (rnd.choice(KIND), code()) for _ in range(ns)]
            golds = [code() for _ in range(ns)]
            answers, ok = [], True
            for k in range(ns):
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

            ids = list(enc(hist))
            for k in range(ns):
                ids += fit("Log: all readings within tolerance. ", 12)
                ids += fit(answers[k], L)

            def ask(qtext, gold):
                full = ids + enc("\nQuestion: what is the %s?\nAnswer:" % qtext)
                st = model.generate(
                    input_ids=torch.tensor([full], device="cuda"),
                    do_sample=False, max_new_tokens=18,
                    attention_mask=torch.ones((1, len(full)), device="cuda", dtype=torch.long))
                txt = tok.decode(st[0][len(full):], skip_special_tokens=True).strip()
                return norm(gold) in norm(txt)

            for q, g in hqs:
                hist_hit[ns] += int(ask(q, g))
                hist_den[ns] += 1
            for k in range(ns):
                slot_hit[ns] += int(ask("access code for the %s" % ents[k], golds[k]))
                slot_den[ns] += 1

        print("  trial %d 完成" % (trial + 1), flush=True)

    print("\nn=%d trials，历史 4 题/次" % N)
    print("  NS   历史准确率      槽位准确率")
    for ns in NS_LIST:
        h = "%d/%d" % (hist_hit[ns], hist_den[ns]) if hist_den[ns] else "—"
        s = "%d/%d" % (slot_hit[ns], slot_den[ns]) if slot_den[ns] else "—"
        print("  %-4d %-15s %s" % (ns, h, s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
