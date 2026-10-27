# -*- coding: utf-8 -*-
"""预留槽位热替换：批量测试。

每例随机生成实体/编码（不可记忆），四臂：
  full      总结一开始就在槽位
  hotswap   先思考 plan_len 个 token（槽位是空占位），再热替换
  hotswap_noplan  不思考，直接替换
  empty     从不替换（下限）
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
PLAN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rnd = random.Random(20261028)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine", "sensor", "coolant"]


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


@torch.no_grad()
def gen(model, tok, ids, cache, n_new):
    cur, toks, logits = list(ids), [], None
    if cur:
        out = model(input_ids=torch.tensor([cur], device="cuda"),
                    past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
    for _ in range(n_new):
        t = int(logits.argmax())
        if t == tok.eos_token_id:
            break
        toks.append(t)
        step = model(input_ids=torch.tensor([[t]], device="cuda"),
                     past_key_values=cache, use_cache=True)
        cache, logits = step.past_key_values, step.logits[0, -1]
    return toks, cache


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    arms = ("full", "hotswap", "hotswap_noplan", "empty")
    hit = {a: 0 for a in arms}
    used = 0
    while used < N:
        other = "%s %s" % (rnd.choice(KIND), code())
        ent = "%s %s" % (rnd.choice(KIND), code())
        attr = "%s code" % rnd.choice(KIND)
        gold, decoy = code(), code()
        S = ("Records:\n- The %s is fitted with the %s %s.\n- The %s runs at %d rpm.\n"
             % (other, attr, decoy, other, rnd.randint(100, 900)))
        summary = "The %s for the %s is %s. " % (attr, ent, gold)
        placeholder = ".... .... .... .... .... .... "
        q2 = "\nQuestion: What is the %s for the %s?\nAnswer:" % (attr, ent)
        hdr = "\nPlan:"

        aS, a_sum, a_q2, a_hdr = enc(S), enc(summary), enc(q2), enc(hdr)
        a_ph = enc(placeholder)
        while len(a_ph) < len(a_sum):
            a_ph += enc(".")
        a_ph = a_ph[:len(a_sum)]
        P, L = len(aS), len(a_sum)
        used += 1
        g = norm(gold)

        def arm_run(mode, plen):
            base = aS + (a_sum if mode == "full" else a_ph) + a_hdr
            out = model(input_ids=torch.tensor([base], device="cuda"), use_cache=True)
            cache, logits = out.past_key_values, out.logits[0, -1]
            for _ in range(plen):
                t = int(logits.argmax())
                step = model(input_ids=torch.tensor([[t]], device="cuda"),
                             past_key_values=cache, use_cache=True)
                cache, logits = step.past_key_values, step.logits[0, -1]
            if mode.startswith("hotswap"):
                pos = torch.arange(0, P + L, device="cuda").unsqueeze(0)
                s_out = model(input_ids=torch.tensor([aS + a_sum], device="cuda"),
                              position_ids=pos, use_cache=True)
                scf = s_out.past_key_values
                kk = [scf.layers[i].keys[:, :, P:P + L, :] for i in range(len(scf.layers))]
                vv = [scf.layers[i].values[:, :, P:P + L, :] for i in range(len(scf.layers))]
                rb = DynamicCache()
                for li in range(len(cache.layers)):
                    k = cache.layers[li].keys.clone()
                    v = cache.layers[li].values.clone()
                    k[:, :, P:P + L, :] = kk[li]
                    v[:, :, P:P + L, :] = vv[li]
                    rb.update(k, v, li)
                cache = rb
            toks, _ = gen(model, tok, a_q2, cache, 40)
            return g in norm(tok.decode(toks, skip_special_tokens=True))

        r = {"full": arm_run("full", PLAN),
             "hotswap": arm_run("hotswap", PLAN),
             "hotswap_noplan": arm_run("hotswap", 0),
             "empty": arm_run("empty", PLAN)}
        for a in arms:
            hit[a] += int(r[a])
        print("  %02d full=%d swap=%d swap0=%d empty=%d | gold=%s"
              % (used, r["full"], r["hotswap"], r["hotswap_noplan"], r["empty"], gold),
              flush=True)

    print("\nplan_len=%d  n=%d" % (PLAN, used))
    for a in arms:
        print("  %-14s %d/%d" % (a, hit[a], used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
