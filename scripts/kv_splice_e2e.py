# -*- coding: utf-8 -*-
"""端到端（v2）：分支消息用 KV 拼接 vs 文本重 prefill。

结构：
  S      共享记录（只含"别的实体"，不含答案）
  q_main 主分支占位问句（与 q_sub 等长，保证 msg 位置一致）
  q_sub  子分支问句
  msg    子智能体产出的消息（含答案）
  q2     末尾的问题（答案只能从 msg 得到）

三臂：
  no_msg  S + q_main + q2
  text    S + q_main + msg + q2
  splice  KV(S+q_main) + KV(msg，来自 S+q_sub) + q2
"""
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/npc-models-qwen3-1.7b"
if not __import__("os").path.exists(MODEL):
    MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rnd = random.Random(20261027)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
WORDS = ["coolant", "turbine", "valve", "pump"]


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


@torch.no_grad()
def gen(model, tok, ids, past=None, n_new=32):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=past, use_cache=True)
    logits, cache = out.logits[0, -1], out.past_key_values
    toks = []
    for _ in range(n_new):
        t = int(logits.argmax())
        if t == tok.eos_token_id:
            break
        toks.append(t)
        step = model(input_ids=torch.tensor([[t]], device="cuda"),
                     past_key_values=cache, use_cache=True)
        cache, logits = step.past_key_values, step.logits[0, -1]
    return tok.decode(toks, skip_special_tokens=True)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    stats = {"no_msg": 0, "text": 0, "splice": 0}
    used, samples = 0, []
    while used < N:
        ent = rnd.choice(WORDS) + " " + code()
        other = rnd.choice(WORDS) + " " + code()
        attr = rnd.choice(WORDS) + " code"
        gold = code()
        S = ("Records:\n- The %s is fitted with the %s %s.\n"
             "- The %s runs at %d rpm.\n" % (other, attr, code(), other, rnd.randint(100, 900)))
        q_main = "Note A: verify the sensor calibration. "
        q_sub = "Note B: verify the sensor calibration. "
        msg = "The %s for the %s is %s. " % (attr, ent, gold)
        q2 = "\nQuestion: What is the %s for the %s?\nAnswer:" % (attr, ent)

        aS, aqm, aqs, amsg, aq2 = enc(S), enc(q_main), enc(q_sub), enc(msg), enc(q2)
        if len(aqm) != len(aqs):
            continue
        used += 1
        n_main = len(aS) + len(aqm)
        n_sub_end = len(aS) + len(aqs) + len(amsg)
        g = norm(gold)

        out_no = gen(model, tok, enc(S + q_main + q2))
        out_txt = gen(model, tok, enc(S + q_main + msg + q2))

        sub = model(input_ids=torch.tensor([aS + aqs + amsg], device="cuda"), use_cache=True)
        sc = sub.past_key_values
        mainc = model(input_ids=torch.tensor([aS + aqm], device="cuda"), use_cache=True).past_key_values
        cache = DynamicCache()
        for li in range(len(mainc.layers)):
            kM, vM = mainc.layers[li].keys, mainc.layers[li].values
            kS = sc.layers[li].keys[:, :, n_main:n_sub_end, :]
            vS = sc.layers[li].values[:, :, n_main:n_sub_end, :]
            cache.update(torch.cat([kM, kS], dim=2), torch.cat([vM, vS], dim=2), li)
        out_spl = gen(model, tok, aq2, past=cache)

        hit = {"no_msg": g in norm(out_no), "text": g in norm(out_txt),
               "splice": g in norm(out_spl)}
        for k in stats:
            stats[k] += int(hit[k])
        print("  %02d no=%d txt=%d spl=%d | gold=%s | txt=%r spl=%r"
              % (used, hit["no_msg"], hit["text"], hit["splice"], gold,
                 out_txt[:26], out_spl[:26]), flush=True)

    print("\n%-8s %-10s" % ("arm", "命中"))
    for k, v in stats.items():
        print("%-8s %d/%d" % (k, v, used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
