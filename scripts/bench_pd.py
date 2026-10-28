# -*- coding: utf-8 -*-
"""prefill 与 decode 的耗时对比。

测：
  T_prefill(L)   一次性前向 L 个 token 的耗时
  T_step(L)      已有 L 个 token 的缓存时，生成下一个 token 的耗时
据此算：
  生成 N 个 token 的总时间 ≈ T_prefill(L) + N * T_step(L)
  交叉点：输出多少 token 时 decode 超过 prefill
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
LS = [500, 2000, 8000]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    vocab = model.config.vocab_size

    for L in LS:
        ids = torch.randint(100, 5000, (1, L), device="cuda")

        # 预热
        with torch.no_grad():
            model(input_ids=ids)

        # prefill
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        t_pre = time.time() - t0
        cache = out.past_key_values
        lg = out.logits[0, -1]
        del out

        # decode：连续 30 步，取平均
        n = 30
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n):
                t = int(lg.argmax())
                o = model(input_ids=torch.tensor([[t]], device="cuda"),
                          past_key_values=o_cache if False else cache, use_cache=True)
                cache, lg = o.past_key_values, o.logits[0, -1]
                del o
        torch.cuda.synchronize()
        t_dec = (time.time() - t0) / n

        cross = t_pre / t_dec
        print("L=%-5d  prefill %.3fs (%.0f tok/s) | decode %.4fs/tok (%.0f tok/s) "
              "| 交叉点 %d 个 token"
              % (L, t_pre, L / t_pre, t_dec, 1 / t_dec, int(cross)), flush=True)

        del cache, lg
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
