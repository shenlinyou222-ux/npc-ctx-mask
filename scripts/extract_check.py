# -*- coding: utf-8 -*-
"""自检：若池 = 选中内容本身，则“提取 + 重紧凑”必须与直接前向完全一致。

若不一致 -> 我的提取/重旋转代码有 bug；
若一致   -> 说明 feasibility2 里池化退化是“跨块条件化”问题，不是实现问题。
"""
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    d_head = model.config.head_dim

    rnd = random.Random(7)
    ids = [101] + [rnd.randint(1000, 20000) for _ in range(40)]
    q = [rnd.randint(1000, 20000) for _ in range(5)]

    @torch.no_grad()
    def logits_direct(seq):
        out = model(input_ids=torch.tensor([seq], device="cuda"))
        return out.logits[0, -1].float()

    @torch.no_grad()
    def logits_via_pool(pool_ids, sel_idx, q_ids):
        with rc.capture_rope() as cap:
            out = model(input_ids=torch.tensor([pool_ids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        W = len(sel_idx)
        idx = torch.tensor(sel_idx, device="cuda")
        cos, sin = rc.extract_cos_sin(model, list(range(W)), d_head)
        new = DynamicCache()
        for li in range(len(cache.layers)):
            k_sel = cap.pre[li].float()[:, :, idx, :]
            k = rc.rotate_k(k_sel, cos.float(), sin.float()).to(torch.bfloat16)
            new.update(k, cache.layers[li].values[:, :, idx, :], li)
        o = model(input_ids=torch.tensor([q_ids], device="cuda"),
                  past_key_values=new,
                  attention_mask=torch.ones((1, W + len(q_ids)), device="cuda", dtype=torch.long))
        return o.logits[0, -1].float()

    # A) 池就是全部内容，选中全部 -> 应与直接前向一致
    a = logits_direct(ids + q)
    b = logits_via_pool(ids, list(range(len(ids))), q)
    d_full = (a - b).abs().max().item()
    print("A) 池=内容全集，全选：max|Δlogits| = %.4f" % d_full)

    # B) 池更长，只选前 41 个（即原 ids），重紧凑 -> 应与 A 一致
    pool_long = ids + [rnd.randint(1000, 20000) for _ in range(60)]
    c = logits_via_pool(pool_long, list(range(len(ids))), q)
    d_sub = (a - c).abs().max().item()
    print("B) 池更长、只选前 %d 个：max|Δlogits| = %.4f" % (len(ids), d_sub))

    ok = d_full < 0.5
    print("\n结论：提取+重紧凑 %s" % ("与原前向一致 ✔" if ok else "不一致 ✘（实现有 bug）"))
    print("  子集与全集是否一致：%s" % ("是" if d_sub < 0.5 else "否（说明池上下文影响了被选块）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
