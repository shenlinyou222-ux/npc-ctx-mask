# -*- coding: utf-8 -*-
"""挖洞的依赖条件：泄漏量与「过洞 token 数」的关系。

三臂（位置始终对齐，只剪 K/V，不改位置）：
  A 不剪                          上限
  B 剪掉，且没有过洞 token          从第一步就遮
  C 剪掉，但先让 K 个 token 在洞可见时生成（它们依赖洞）

指标：问题末尾 gold 编码首 token 的概率与排名。
"""
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
SYS = "You are a careful assistant.\n\n"
DOC = "Record: the access code for the blue door is %s. Maintenance was routine."
ASK = "\nIs this record about maintenance? Answer:"
Q = "\nQuestion: what is the access code?\nAnswer:"
rnd = random.Random(20261028)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
KS = [1, 4, 16, 32]


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


def metric(logits, target_id):
    p = torch.softmax(logits.float(), -1)
    prob = float(p[target_id])
    rank = int((p > p[target_id]).sum()) + 1
    return prob, rank


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    res = {K: [] for K in KS}
    a_list, b_list = [], []
    for i in range(N):
        c = code()
        head = enc(SYS)
        dids = enc(DOC % c)
        a, b = len(head), len(head) + len(dids)
        pre = head + dids + enc(ASK)
        qids = enc(Q)
        target = enc(" " + c)[0]
        base = len(pre)

        oA = fwd(model, pre + qids)
        pa = metric(oA.logits[0, -1], target)
        a_list.append(pa)

        o0 = fwd(model, pre)
        pb_cache = prune(o0.past_key_values, [(0, a), (b, base)])
        oB = fwd(model, qids, pb_cache, list(range(base, base + len(qids))))
        pB = metric(oB.logits[0, -1], target)
        b_list.append(pB)

        line = "  %d) %s  A=%.4f  B=%.5f  " % (i + 1, c, pa[0], pB[0])
        for K in KS:
            cc, lg = o0.past_key_values, o0.logits[0, -1]
            for _ in range(K):
                _, cc, lg = gen1(model, lg, cc)
            pcc = prune(cc, [(0, a), (b, base + K)])
            b2 = base + K
            oC = fwd(model, qids, pcc, list(range(b2, b2 + len(qids))))
            m = metric(oC.logits[0, -1], target)
            res[K].append(m)
            line += "K%d=%.5f " % (K, m[0])
        print(line, flush=True)

    def mean(lst, idx):
        return sum(x[idx] for x in lst) / len(lst)

    print("\nn=%d" % N)
    print("  A 不剪：            p=%.5f  平均 rank=%.1f" % (mean(a_list, 0), mean(a_list, 1)))
    print("  B 剪·无过洞 token： p=%.6f  平均 rank=%.1f" % (mean(b_list, 0), mean(b_list, 1)))
    b0 = max(mean(b_list, 0), 1e-12)
    for K in KS:
        print("  C 剪·%2d 个过洞 token： p=%.6f  平均 rank=%.1f  (x%.1f vs B)"
              % (K, mean(res[K], 0), mean(res[K], 1), mean(res[K], 0) / b0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
