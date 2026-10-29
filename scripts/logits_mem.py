# -*- coding: utf-8 -*-
"""验证：logits 是显存大户，且可以只算最后一个位置。

对比三种取法在 4378 token 输入下的显存峰值：
  A 默认（全序列 logits）
  B logits_to_keep=1
  C 只跑 backbone，再对最后一位手动做 lm_head
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
import sys
ATTN = sys.argv[1] if len(sys.argv) > 1 else "eager"
L = 4378


def peak():
    return torch.cuda.max_memory_allocated() / 1e9


def reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=ATTN).eval()
    ids = torch.randint(100, 20000, (1, L), device="cuda")
    V = model.config.vocab_size
    print("attention=%s  输入 %d token，词表 %d" % (ATTN, L, V))
    print("全序列 logits 理论大小: %.2f GB" % (L * V * 4 / 1e9))

    # A 默认
    reset()
    with torch.no_grad():
        o = model(input_ids=ids)
    print("A 默认            logits=%s  峰值 %.2f GB" % (tuple(o.logits.shape), peak()))
    del o
    torch.cuda.empty_cache()

    # B logits_to_keep=1
    ok_b = True
    reset()
    try:
        with torch.no_grad():
            o = model(input_ids=ids, logits_to_keep=1)
        print("B logits_to_keep=1 logits=%s  峰值 %.2f GB" % (tuple(o.logits.shape), peak()))
        del o
    except TypeError as e:
        ok_b = False
        print("B 不支持 logits_to_keep: %s" % str(e)[:80])
    torch.cuda.empty_cache()

    # C 只跑 backbone + 手动 lm_head
    reset()
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[:, -1:, :]
        import torch.nn.functional as F
        lg = F.linear(h, model.lm_head.weight)
    print("C 手动最后一位     logits=%s  峰值 %.2f GB" % (tuple(lg.shape), peak()))
    print("\n节省: A %.2f GB -> C %.2f GB" % (2.66, 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
