# -*- coding: utf-8 -*-
"""长过洞 token：区分「KV 侧信道」与「文本泄漏」。

关键：记录过洞生成文本里是否已出现编码。
  - 若已出现 -> 文本泄漏（不算侧信道），单独统计
  - 若未出现 -> 此时 p(code) 的提升才来自 KV
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
SYS = "You are a careful assistant.\n\n"
DOC = "Record: the access code for the blue door is %s. Maintenance was routine."
ASK = "\nDescribe the maintenance situation briefly.\nDescription:"
Q = "\nQuestion: what is the access code?\nAnswer:"
rnd = random.Random(20261028)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
KS = [1, 4, 16, 32, 48, 64, 96, 128]


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


@torch.no_grad()
def fwd(model, ids, past=None, positions=None):
    kw = {}
    if positions is not None:
        kw["position_ids"] = torch.tensor([positions], device="cuda")
    return model(input_ids=torch.tensor([ids], device="cuda"),
                 past_key_values=past, use_cache=True, **kw)


@torch.no_grad()
def gen1(model, logits, cache):
    t = int(logits.argmax())
    out = fwd(model, [t], cache)
    return t, out.past_key_values, out.logits[0, -1]


def prune(cache, keep):
    rb = DynamicCache()
    for li in range(len(cache.layers)):
        k = cache.layers[li].keys
        v = cache.layers[li].values
        idx = torch.cat([torch.arange(s, e, device="cuda") for s, e in keep])
        rb.update(k[:, :, idx, :].clone(), v[:, :, idx, :].clone(), li)
    return rb


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    acc = {K: {"clean_p": [], "clean_rank": [], "leak_text": 0, "n": 0} for K in KS}
    for i in range(N):
        c = code()
        head = enc(SYS)
        dids = enc(DOC % c)
        a, b = len(head), len(head) + len(dids)
        pre = head + dids + enc(ASK)
        qids = enc(Q)
        target = enc(" " + c)[0]
        base = len(pre)
        o0 = fwd(model, pre)
        toks_all = []

        for K in KS:
            cc, lg = o0.past_key_values, o0.logits[0, -1]
            toks = []
            while len(toks) < K:
                t, cc, lg = gen1(model, lg, cc)
                toks.append(t)
            text = tok.decode(toks, skip_special_tokens=True)
            has = norm(c) in norm(text)
            pcc = prune(cc, [(0, a), (b, base + K)])
            b2 = base + K
            oC = fwd(model, qids, pcc, list(range(b2, b2 + len(qids))))
            p = torch.softmax(oC.logits[0, -1].float(), -1)
            prob = float(p[target])
            rank = int((p > p[target]).sum()) + 1
            acc[K]["n"] += 1
            acc[K]["leak_text"] += int(has)
            if not has:
                acc[K]["clean_p"].append(prob)
                acc[K]["clean_rank"].append(rank)
            if K == KS[-1]:
                print("  末段文本示例: %r" % text[:80], flush=True)

        print("  %d) %s 各 K 的文本是否含编码：%s"
              % (i + 1, c, {K: acc[K]["leak_text"] - (0 if i == 0 else 0) for K in KS}),
              flush=True)

    print("\nn=%d" % N)
    print("  K    文本含码次数   纯 KV 样本   p(净)       rank(净)")
    for K in KS:
        s = acc[K]
        ncl = len(s["clean_p"])
        if ncl:
            print("  %-4d %-14d %-10d %.6f   %.0f"
                  % (K, s["leak_text"], ncl, sum(s["clean_p"]) / ncl,
                     sum(s["clean_rank"]) / ncl))
        else:
            print("  %-4d %-14d %-10d  (无干净样本)" % (K, s["leak_text"], 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
