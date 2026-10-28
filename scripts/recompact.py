# -*- coding: utf-8 -*-
"""块级稀疏 decode 的重紧凑（recompaction）工具。

思路（与 InferScale 的 chunked RoPE 同源）：
  存**未旋转**的 K；在真正参与注意力时，按“紧凑坐标”施加旋转。

本文件提供三个东西：
  capture_pre_rope()         拦截 HF 的 apply_rotary_pos_emb，拿到旋转前的 K
  extract_cos_sin(model,pos) 取任意位置的 cos/sin 表
  rotate_k(k_pre, cos, sin)  对未旋转的 K 施加旋转

以及一个等价性自检：rotate_k(旋转前的K) 是否等于 HF 自己算出的旋转后 K。
"""
from __future__ import annotations

import contextlib
import importlib

import torch

_ROPE_MODULES = (
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
)


class RopeCapture:
    def __init__(self):
        self.pre = []
        self.post = []

    def add(self, k_pre, k_post):
        self.pre.append(k_pre.detach().clone())
        self.post.append(k_post.detach().clone())


@contextlib.contextmanager
def capture_rope():
    """在一次前向中同时收集旋转前/后的 K（逐层）。"""
    cap = RopeCapture()
    patched = []
    for path in _ROPE_MODULES:
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        orig = getattr(mod, "apply_rotary_pos_emb", None)
        if orig is None:
            continue

        def make(original):
            def wrapper(q, k, cos, sin, *a, **kw):
                out_q, out_k = original(q, k, cos, sin, *a, **kw)
                cap.add(k, out_k)
                return out_q, out_k
            return wrapper

        mod.apply_rotary_pos_emb = make(orig)
        patched.append((mod, orig))
    if not patched:
        raise RuntimeError("找不到任何 apply_rotary_pos_emb，模型没加载？")
    try:
        yield cap
    finally:
        for mod, orig in patched:
            mod.apply_rotary_pos_emb = orig


def _find_rotary(model):
    try:
        return model.model.rotary_emb
    except AttributeError:
        pass
    return model.model.layers[0].self_attn.rotary_emb


def extract_cos_sin(model, positions, head_dim):
    rotary = _find_rotary(model)
    dev = None
    for tensors in (rotary.parameters(), rotary.buffers(), model.parameters()):
        t = next(iter(tensors), None)
        if t is not None:
            dev = t.device
            break
    if dev is None:
        dev = next(model.parameters()).device
    pos = torch.as_tensor(positions, dtype=torch.long, device=dev).unsqueeze(0)
    dummy = torch.zeros(1, pos.shape[-1], head_dim, device=dev, dtype=torch.float32)
    cos, sin = rotary(dummy, pos)
    return cos.squeeze(0), sin.squeeze(0)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def rotate_k(k_pre, cos, sin):
    """对未旋转的 K 施加旋转。k_pre: (b, heads, n, d)。"""
    if cos.dim() == 3:
        cos = cos.squeeze(0)
    if sin.dim() == 3:
        sin = sin.squeeze(0)
    while cos.dim() < k_pre.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(dtype=k_pre.dtype, device=k_pre.device)
    sin = sin.to(dtype=k_pre.dtype, device=k_pre.device)
    return (k_pre * cos) + (_rotate_half(k_pre) * sin)
