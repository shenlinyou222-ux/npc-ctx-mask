# -*- coding: utf-8 -*-
"""等价性自检：rotate_k(旋转前K) 是否等于 HF 自己算的旋转后 K。

再补一项：把某段 K 搬到新位置后，它与 query 的注意力分数
是否等于“该段本来就在那个位置”时算出来的分数（Theorem 1 的数值验证）。
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/_ab_scratch")
import recompact as rc  # noqa: E402

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()

    ids = torch.randint(100, 20000, (1, 64), device="cuda")
    with torch.no_grad():
        with rc.capture_rope() as cap:
            model(input_ids=ids, use_cache=False)

    print("捕获层数: %d" % len(cap.pre))
    d = cap.pre[0].shape[-1]
    positions = torch.arange(ids.shape[1], device="cuda")
    cos, sin = rc.extract_cos_sin(model, positions, d)

    worst = 0.0
    for li, (k_pre, k_post) in enumerate(zip(cap.pre, cap.post)):
        k_hat = rc.rotate_k(k_pre, cos, sin)
        diff = (k_hat.float() - k_post.float()).abs().max().item()
        worst = max(worst, diff)
        if li < 3:
            print("  layer %-2d max|diff| = %.3e" % (li, diff))

    print("\n最大逐元素误差: %.3e  (bf16 下应 < 1e-2)" % worst)
    ok = worst < 1e-2
    print("等价性: %s" % ("通过 ✔" if ok else "不通过 ✘"))

    # 相对位置性质：K 段与 query 同步平移 delta，注意力分数应完全不变
    with torch.no_grad():
        k_seg = cap.pre[0][:, :, :4, :].clone()          # (1, heads, 4, d)
        q_seg = torch.randn_like(k_seg)                    # (1, heads, 4, d)
        probe = [0, 1, 2, 3]                              # K 的位置
        qpos = [4, 5, 6, 7]                               # query 的位置
        D = 1000
        ca, sa = rc.extract_cos_sin(model, probe, d)
        qca, qsa = rc.extract_cos_sin(model, qpos, d)
        k0, q0 = rc.rotate_k(k_seg, ca, sa), rc.rotate_k(q_seg, qca, qsa)

        cb, sb = rc.extract_cos_sin(model, [p + D for p in probe], d)
        qcb, qsb = rc.extract_cos_sin(model, [p + D for p in qpos], d)
        k1, q1 = rc.rotate_k(k_seg, cb, sb), rc.rotate_k(q_seg, qcb, qsb)

        def att(qq, kk):
            return torch.einsum("bhqd,bhkd->bhqk", qq.float(), kk.float())

        sa_s, sb_s = att(q0, k0), att(q1, k1)
        dmax = (sa_s - sb_s).abs().max().item()
        rel = ((sa_s - sb_s).abs() / sa_s.abs().clamp_min(1e-6)).max().item()
        # fp32 旋转
        def rot32(x, c, s_):
            return rc.rotate_k(x.float(), c, s_)
        q0f, k0f = rot32(q_seg, qca, qsa), rot32(k_seg, ca, sa)
        q1f, k1f = rot32(q_seg, qcb, qsb), rot32(k_seg, cb, sb)
        d32 = (att(q0f, k0f) - att(q1f, k1f)).abs().max().item()
    print("整段平移 %d：" % D)
    print("  bf16 旋转：绝对差 %.3e，相对差 %.3e" % (dmax, rel))
    print("  fp32 旋转：绝对差 %.3e" % d32)
    good = d32 < 1e-2
    print("平移等价(fp32): %s" % ("通过 ✔" if good else "不通过 ✘"))
    return 0 if (ok and good) else 1


if __name__ == "__main__":
    sys.exit(main())
