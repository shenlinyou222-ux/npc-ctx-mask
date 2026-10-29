# -*- coding: utf-8 -*-
"""长上下文 + wiki 角色资料（约 4.8K token），bf16 存 K_pre。

题目：
  查找类：某人赢了几场？          （四选一数字）
  比较类：谁赢的更多？            （四选一人名，必须看两块）

臂：full / fresh / pool_look / pool_reason / only_one(捷径检测)
用法：python wiki_test.py [bf16|fp32] [n] [trial]
"""
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
DTYPE_KPRE = sys.argv[1] if len(sys.argv) > 1 else "bf16"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TRIAL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FILL_BETWEEN = 700
rnd = random.Random(20261107 + TRIAL)

NAMES = ["Aurelia Vane", "Bram Hollow", "Cyrus Pell", "Dara Quill",
         "Emeric Stane", "Fenna Rook", "Galen Moss", "Hestra Lune"]
ROLES = ["archivist", "ranger", "alchemist", "duelist",
         "navigator", "warden", "historian", "sentinel"]
HOUSES = ["House Vey", "House Marr", "House Tollen", "House Ash",
          "House Quen", "House Brann", "House Ost", "House Ild"]
WEAPONS = ["a brass compass", "a cracked longsword", "a silver ledger",
           "a glass spear", "a coil of wire", "a weathered map",
           "a smoking censer", "a black banner"]
YEARS = [1178, 1191, 1204, 1219, 1233, 1247, 1260, 1274]
WINPOOL = [11, 3, 18, 6, 14, 1, 19, 8]
FILLW = "The archive index lists unrelated ledgers and shipping manifests. "


def entry(name, role, house, weapon, year, wins):
    return (
        "%s is a %s sworn to %s. Born in the year %d, %s has won %d victories "
        "in border skirmishes. %s spent a decade cataloguing disputes before "
        "taking up %s. %s is known for refusing to speak during negotiations "
        "and for keeping a private ledger of every debt owed to %s. Those who "
        "have seen the ledger say %s never forgets a favour or a slight, and "
        "that the count of victories is recorded on its first page. In the "
        "winter of %d %s withdrew from the capital and has been seen only at "
        "the northern archive since then.\n"
        % (name, role, house, year, name, wins, name, weapon, name, house,
           name, year + 3, name))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    letters = [enc(" A")[0], enc(" B")[0], enc(" C")[0], enc(" D")[0]]
    d_head = model.config.head_dim
    unit = enc(FILLW)

    def fill(n):
        o = []
        while len(o) < n:
            o += unit
        return o[:n]

    def pick(lg):
        lg = lg.float()
        return max(range(4), key=lambda i: float(lg[letters[i]]))

    @torch.no_grad()
    def pool_of(ids):
        with rc.capture_rope() as cap:
            out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True,
                        logits_to_keep=1)
        cache = out.past_key_values
        kpre = [k.clone() if DTYPE_KPRE == "bf16" else k.float() for k in cap.pre]
        vals = [cache.layers[li].values.clone() for li in range(len(cache.layers))]
        del out, cache
        torch.cuda.empty_cache()
        return kpre, vals

    @torch.no_grad()
    def assemble(kpre, vals, sel_idx):
        W = len(sel_idx)
        idx = torch.tensor(sel_idx, device="cuda")
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
        c = DynamicCache()
        for li in range(len(kpre)):
            k = kpre[li][:, :, idx, :]
            k = rc.rotate_k(k.float(), cos.float(), sin.float()).to(torch.bfloat16)
            c.update(k, vals[li][:, :, idx, :], li)
        return c, W

    @torch.no_grad()
    def ask(cache, W, q_ids):
        o = model(input_ids=torch.tensor([q_ids], device="cuda"), past_key_values=cache,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long),
                  logits_to_keep=1)
        v = pick(o.logits[0, -1])
        del o, cache
        torch.cuda.empty_cache()
        return v

    @torch.no_grad()
    def plain(ids):
        o = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
        v = pick(o.logits[0, -1])
        del o
        torch.cuda.empty_cache()
        return v

    stat = {k: 0 for k in ("full", "fresh", "pool_look", "pool_reason", "only_one")}
    cost = {k: 0.0 for k in stat}
    ctx_len = W_used = 0
    used = 0

    for t in range(N):
        order = list(range(8))
        rnd.shuffle(order)
        names = [NAMES[i] for i in order]
        roles = [ROLES[i] for i in order]
        houses = [HOUSES[i] for i in order]
        weaps = [WEAPONS[i] for i in order]
        years = [YEARS[i] for i in order]
        wins = WINPOOL[:]
        rnd.shuffle(wins)

        S0 = "You read a wiki of noble houses and answer with one letter.\n"
        base = enc(S0)
        cur = len(base)
        spans = []
        body = []
        for k in range(8):
            e = enc(entry(names[k], roles[k], houses[k], weaps[k], years[k], wins[k]))
            spans.append((cur, cur + len(e)))
            body += e
            body += fill(FILL_BETWEEN)
            cur += len(e) + FILL_BETWEEN

        tgt, oth = 0, 1
        w1, w2 = wins[tgt], wins[oth]
        opts = [w1, w1 + 3, w1 + 7, w1 - 2]
        rnd.shuffle(opts)
        gold_look = opts.index(w1)
        q_look = ("\nQuestion: how many victories has %s won?\n" % names[tgt]
                  + "".join(" %s) %d\n" % ("ABCD"[i], opts[i]) for i in range(4))
                  + "Answer:")
        gold_cmp = 0 if w1 > w2 else 1
        q_cmp = ("\nQuestion: who has won more victories?\n"
                 " A) %s\n B) %s\n C) equal\n D) unknown\nAnswer:" % (names[tgt], names[oth]))

        ids_full = base + body + enc(q_look)
        ctx_len += len(ids_full)
        used += 1
        pre = list(range(0, len(S0)))
        s_t, e_t = spans[tgt]
        s_o, e_o = spans[oth]

        t0 = time.time(); vf = plain(ids_full); cost["full"] += time.time() - t0
        t0 = time.time(); kpre, vals = pool_of(base + body)
        v_look = ask(*(assemble(kpre, vals, pre + list(range(s_t, e_t))) + (enc(q_look),)))
        cost["pool_look"] += time.time() - t0
        t0 = time.time()
        v_cmp = ask(*(assemble(kpre, vals, pre + list(range(s_t, e_t)) + list(range(s_o, e_o)))
                      + (enc(q_cmp),)))
        cost["pool_reason"] += time.time() - t0
        t0 = time.time()
        v_one = ask(*(assemble(kpre, vals, pre + list(range(s_t, e_t))) + (enc(q_cmp),)))
        cost["only_one"] += time.time() - t0
        del kpre, vals
        torch.cuda.empty_cache()
        W_used = len(pre) + (e_t - s_t) + (e_o - s_o)

        fi = base + enc(entry(names[tgt], roles[tgt], houses[tgt], weaps[tgt],
                              years[tgt], w1)) + enc(q_cmp)
        t0 = time.time(); v_fr = plain(fi); cost["fresh"] += time.time() - t0

        stat["full"] += int(vf == gold_look)
        stat["fresh"] += int(v_fr == gold_cmp)
        stat["pool_look"] += int(v_look == gold_look)
        stat["pool_reason"] += int(v_cmp == gold_cmp)
        stat["only_one"] += int(v_one == gold_cmp)
        print("  %d) ctx=%d 块=%d/%d | 查找 full=%s pool=%s | 比较 fresh=%s pool=%s only1=%s"
              % (t + 1, len(ids_full), e_t - s_t, e_o - s_o,
                 "O" if vf == gold_look else "X", "O" if v_look == gold_look else "X",
                 "O" if v_fr == gold_cmp else "X", "O" if v_cmp == gold_cmp else "X",
                 "O" if v_one == gold_cmp else "X"), flush=True)

    print("K_pre=%s ctx=%d W=%d" % (DTYPE_KPRE, ctx_len // max(used, 1), W_used))
    for k in ("full", "fresh", "pool_look", "pool_reason", "only_one"):
        print("  %-12s %d/%d %.3fs" % (k, stat[k], used, cost[k] / max(used, 1)))
    print("  峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
