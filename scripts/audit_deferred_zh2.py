# -*- coding: utf-8 -*-
"""中文完整例子 v2：子智能体走 chat template，主序列前缀补到等长。

位置对齐：子智能体生成起点 = L_sub；主序列槽位也从 L_sub 开始。
"""
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
L = 48
PAD = "。"

S0 = ("一、仓库的龙门吊在周一下午做过保养。\n"
      "二、夜班值班员报告当夜没有异常。\n")
ENT, ATTR, GOLD = "阀门 DANDAN-9171", "冷却液编码", "KARTHA-7346"
FACT = "私有记录：%s的%s是%s。" % (ENT, ATTR, GOLD)

SYS_SUB = "你是子智能体。只根据下面这条私有记录，写出一句完整的中文结论，只输出这一句话。"
SYS_MAIN = "你是主智能体。你只能看到共享档案，看不到任何私有记录。"

Q1 = "\n问题一：%s的%s是什么？\n" % (ENT, ATTR)
Q2 = "问题二：龙门吊是什么时候保养的？\n"
HEAD = "\n请先直接回答第二个问题，只给答案。\n二的答案："
NOTICE = ("\n\n【系统提示】子智能体已经返回了问题一的答案，结果就在上面。"
          "现在直接回答问题一，只给答案。\n一的答案：")


@torch.no_grad()
def step(model, tok, ids, past):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[0, -1]


@torch.no_grad()
def gen(model, tok, logits, cache, n_new, stop=True):
    toks = []
    for _ in range(n_new):
        t = int(logits.argmax())
        if stop and t == tok.eos_token_id:
            break
        toks.append(t)
        cache, logits = step(model, tok, [t], cache)
    return toks, cache


def norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(s).lower())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    # --- 子智能体：chat template ---
    msgs = [{"role": "system", "content": SYS_SUB},
            {"role": "user", "content": FACT}]
    ids_sub = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=True, return_dict=False,
                                      enable_thinking=False)
    L_sub = len(ids_sub)
    cache, lg = step(model, tok, ids_sub, None)
    t_sub, sub_cache = gen(model, tok, lg, cache, L, stop=False)
    sub_text = tok.decode(t_sub, skip_special_tokens=True).strip()
    n_cache = sub_cache.layers[0].keys.shape[-2]
    sl = slice(n_cache - len(t_sub), n_cache)

    print("=" * 74)
    print("【共享档案 S0】\n%s" % S0.rstrip())
    print("\n【子智能体：系统提示】%s" % SYS_SUB)
    print("【子智能体：用户消息】共享档案 + %s" % FACT)
    print("【子智能体 prefill 长度】%d token（生成从这里开始）" % L_sub)
    print("\n【子智能体生成】%r" % sub_text)

    # --- 主智能体：前缀补到等长 ---
    a_shared = enc(S0)
    filler_n = L_sub - len(a_shared)
    if filler_n < 0:
        print("共享段比子智能体前缀还长，无法对齐，退出")
        return 1
    a_fill = enc("日志：所有读数在正常范围内，无需上报。")
    while len(a_fill) < filler_n:
        a_fill += enc("日志：所有读数在正常范围内，无需上报。")
    a_fill = a_fill[:filler_n]
    P = L_sub

    print("\n【主序列】共享档案(%d) + 中性填充(%d) + [槽位 %d] + 问题一 + 问题二 + 先答问题二"
          % (len(a_shared), len(a_fill), L))
    print("【槽位区间】[%d, %d)  与子智能体生成起点对齐=%s"
          % (P, P + L, P == L_sub))

    a_q1, a_q2, a_head, a_notice = enc(Q1), enc(Q2), enc(HEAD), enc(NOTICE)
    a_ph = enc("……" * 60)[:L]
    a_full = enc(sub_text)
    while len(a_full) < L:
        a_full += enc(PAD)
    a_full = a_full[:L]

    for mode in ("full", "swap", "noswap"):
        slot = a_full if mode == "full" else a_ph
        cache, lg = step(model, tok, a_shared + a_fill + slot + a_q1 + a_q2 + a_head, None)
        t2, cache = gen(model, tok, lg, cache, 12)
        ans2 = tok.decode(t2, skip_special_tokens=True).strip()
        if mode == "swap":
            rb = DynamicCache()
            for li in range(len(cache.layers)):
                k = cache.layers[li].keys.clone()
                v = cache.layers[li].values.clone()
                k[:, :, P:P + L, :] = sub_cache.layers[li].keys[:, :, sl, :]
                v[:, :, P:P + L, :] = sub_cache.layers[li].values[:, :, sl, :]
                rb.update(k, v, li)
            cache = rb
        cache, lg = step(model, tok, a_notice, cache)
        t1, _ = gen(model, tok, lg, cache, 40)
        ans1 = tok.decode(t1, skip_special_tokens=True).strip()
        print("\n  【%s】" % mode)
        print("    先答问题二 -> %s  %r" % ("✔" if "周一" in ans2 else "✘", ans2[:40]))
        print("    再答问题一 -> %s  %r" % ("✔" if norm(GOLD) in norm(ans1) else "✘", ans1[:70]))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
