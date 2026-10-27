# -*- coding: utf-8 -*-
"""正确流程：主智能体被明确告知「不要自己答，等子智能体返回」。

时序：
  1. S + [槽位空] + 问题 + "Coordinator: I do not know this. I must wait for the sub-agent.\n<WAIT>"
     （<WAIT> 作为助手回合的前缀被强制写入，保证它不作答、不押注）
  2. 热替换槽位 = 子智能体总结（位置不变）
  3. 追加 "\nSub-agent returned: " 然后生成答案

臂：
  wait_swap    第2步替换      （目标行为）
  wait_noswap  第2步不替换    （应失败）
  wait_full    总结一开始就在槽位（上限参照）
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rnd = random.Random(20261030)
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

    arms = ("reprefill", "wait_swap", "wait_noswap", "wait_full")
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
        q = ("\nQuestion: What is the %s for the %s?\n" % (attr, ent))
        wait = ("Coordinator: I do not know this. I must wait for the sub-agent.\n<WAIT>")
        cue = "\nSub-agent returned: "

        aS, a_sum, aq, a_wait, a_cue = enc(S), enc(summary), enc(q), enc(wait), enc(cue)
        a_ph = enc(".... .... .... .... .... .... ")
        while len(a_ph) < len(a_sum):
            a_ph += enc(".")
        a_ph = a_ph[:len(a_sum)]
        P, L = len(aS), len(a_sum)
        used += 1
        g = norm(gold)

        def run(mode):
            slot = a_sum if mode == "wait_full" else a_ph
            ids = aS + slot + aq + a_wait
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
            cache = out.past_key_values
            if mode == "wait_swap":
                pos = torch.arange(0, P + L, device="cuda").unsqueeze(0)
                so = model(input_ids=torch.tensor([aS + a_sum], device="cuda"),
                           position_ids=pos, use_cache=True)
                scf = so.past_key_values
                rb = DynamicCache()
                for li in range(len(cache.layers)):
                    k = cache.layers[li].keys.clone()
                    v = cache.layers[li].values.clone()
                    k[:, :, P:P + L, :] = scf.layers[li].keys[:, :, P:P + L, :]
                    v[:, :, P:P + L, :] = scf.layers[li].values[:, :, P:P + L, :]
                    rb.update(k, v, li)
                cache = rb
            cue = enc(
                "\n\n[System note] The sub-agent has finished its work and "
                "returned the result shown above. You may now use that result "
                "to answer the question.\nAnswer:")
            if mode in ("wait_noswap",):
                cue = a_cue
            if mode == "reprefill":
                ids2 = aS + a_sum + aq + a_wait + enc(
                    "\n\n[System note] The sub-agent has finished its work and "
                    "returned the result shown above. You may now use that result "
                    "to answer the question.\nAnswer:")
                o = model(input_ids=torch.tensor([ids2], device="cuda"), use_cache=True)
                toks = []
                lg = o.logits[0, -1]
                cc = o.past_key_values
                for _ in range(40):
                    t = int(lg.argmax())
                    if t == tok.eos_token_id:
                        break
                    toks.append(t)
                    st = model(input_ids=torch.tensor([[t]], device="cuda"),
                               past_key_values=cc, use_cache=True)
                    cc, lg = st.past_key_values, st.logits[0, -1]
                return tok.decode(toks, skip_special_tokens=True)
            if mode == "flush_swap":
                # 1) 截断掉陈旧的 <WAIT> 尾巴（保留问题）
                keep = P + L + len(aq)
                rb2 = DynamicCache()
                for li in range(len(cache.layers)):
                    rb2.update(cache.layers[li].keys[:, :, :keep].clone(),
                               cache.layers[li].values[:, :, :keep].clone(), li)
                cache = rb2
                # 2) 用一句完整描述重新 prefill，告诉它子智能体回来了
                cue = enc(
                    "\n\n[System note] The sub-agent has finished its work and "
                    "returned the result shown above. You may now use that result "
                    "to answer the question.\nAnswer:")
            toks, _ = gen(model, tok, cue, cache, 40)
            return tok.decode(toks, skip_special_tokens=True)

        r = {a: run(a) for a in arms}
        for a in arms:
            hit[a] += int(g in norm(r[a]))
        print("  %02d swap=%d noswap=%d full=%d | gold=%s | swap_out=%r"
              % (used, g in norm(r["wait_swap"]), g in norm(r["wait_noswap"]),
                 g in norm(r["wait_full"]), gold, r["wait_swap"][:30]), flush=True)

    print("\nn=%d" % used)
    for a in arms:
        print("  %-11s %d/%d" % (a, hit[a], used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
