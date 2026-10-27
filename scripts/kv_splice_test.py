# -*- coding: utf-8 -*-
"""KV 拼接实验：分支输出拼回主上下文，看分布是否被破坏。

控制变量：
  S      共享前缀
  q_main 主分支问句（长度 Lq）
  q_sub  子分支问句（长度也为 Lq，保证拼接时位置不变）
  a      子分支生成的答案 token

对照：
  baseline  全量 prefill: S + q_main + a           （标准做法）
  splice    KV(S+q_main) + KV(a 来自 S+q_sub)      （拼接，缺跨注意力到 q_main）
因为 q_main / q_sub 等长，a 的绝对位置在两种情况下**完全相同**，
所以差异只来自"a 的 K/V 是在 q_sub 语境下算的，没看过 q_main"。
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"

CASES = [
    {
        "S": "Reference: Paris has 2.1 million residents. Rome has 2.8 million residents. ",
        "q_main": "Which city is larger? Answer:",
        "q_sub": "Which city is warmer? Answer:",
        "a": "Rome is the larger city, with about 2.8 million residents.",
    },
    {
        "S": "Log: server A responded in 20 ms. Server B responded in 90 ms. ",
        "q_main": "Which server is faster? Answer:",
        "q_sub": "Which server is slower? Answer:",
        "a": "Server A is faster because it responded in only 20 milliseconds.",
    },
    {
        "S": "Reference: the Eiffel Tower is in Paris. The Colosseum is in Rome. ",
        "q_main": "Where is the Colosseum? Answer:",
        "q_sub": "Where is the Colosseum? Answer:",
        "a": "The Colosseum is located in the city of Rome, in Italy.",
    },
]


@torch.no_grad()
def run(model, ids, past=None):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=past, use_cache=True)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    for ci, c in enumerate(CASES):
        S, qm, qs = enc(c["S"]), enc(c["q_main"]), enc(c["q_sub"])
        a = enc(c["a"])
        if len(qm) != len(qs):
            print("  [skip case %d] q 长度不等: %d vs %d" % (ci, len(qm), len(qs)))
            continue

        ids_main = S + qm
        ids_sub = S + qs
        ids_full = ids_main + a

        # baseline：全量 prefill
        out_full = run(model, ids_full)
        logits_base = out_full.logits[0, -1].float()

        # 子分支语境下算 a 的 KV
        out_sub = run(model, ids_sub + a)
        sub_cache = out_sub.past_key_values

        # 主分支 cache
        out_main = run(model, ids_main)
        main_cache = out_main.past_key_values

        # 拼接：主 cache + a（去掉最后一个 token，最后那个单独 forward）
        n_main = len(ids_main)
        n_sub = len(ids_sub) + len(a)
        combined_layers = []
        for li in range(len(main_cache.layers)):
            kM = main_cache.layers[li].keys
            vM = main_cache.layers[li].values
            kS = sub_cache.layers[li].keys[:, :, n_main:n_sub - 1, :]
            vS = sub_cache.layers[li].values[:, :, n_main:n_sub - 1, :]
            combined_layers.append((torch.cat([kM, kS], dim=2),
                                    torch.cat([vM, vS], dim=2)))

        # 用拼好的 cache 再喂 a 的最后一个 token
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for li, (k, v) in enumerate(combined_layers):
            cache.update(k, v, li)
        out_splice = run(model, [a[-1]], past=cache)
        logits_spl = out_splice.logits[0, -1].float()

        # 对比
        p = torch.softmax(logits_base, -1)
        q = torch.softmax(logits_spl, -1)
        kl = float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())
        top1 = int(logits_base.argmax()) == int(logits_spl.argmax())
        top5b = set(torch.topk(logits_base, 5).indices.tolist())
        top5s = set(torch.topk(logits_spl, 5).indices.tolist())
        print("  case %d  KL=%.5f  top1一致=%s  top5重合=%d/5"
              % (ci, kl, top1, len(top5b & top5s)))
        print("     baseline 下一词: %r" % tok.decode([int(logits_base.argmax())]))
        print("     splice   下一词: %r" % tok.decode([int(logits_spl.argmax())]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
