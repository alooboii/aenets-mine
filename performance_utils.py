from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _as_tensor(output):
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        return _as_tensor(output[0])
    if torch.is_tensor(output):
        return output
    return None


def estimate_model_flops(
    model: nn.Module,
    *,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> int:
    """
    Rough FLOPs estimate for one forward pass.

    Counts Conv/Linear FLOPs and adds attention core + qkv projection FLOPs
    for nn.MultiheadAttention (out_proj is counted via Linear hook).
    """
    total_flops = 0
    hooks = []

    conv_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )

    def conv_hook(module: nn.Module, inputs, output):
        nonlocal total_flops
        x = _as_tensor(inputs)
        y = _as_tensor(output)
        if x is None or y is None:
            return
        batch = int(y.shape[0]) if y.ndim >= 1 else 1
        out_channels = int(y.shape[1]) if y.ndim >= 2 else int(getattr(module, "out_channels", 1))
        spatial = int(math.prod(y.shape[2:])) if y.ndim > 2 else 1

        kernel_size = getattr(module, "kernel_size", 1)
        if isinstance(kernel_size, tuple):
            kernel_ops = int(math.prod(kernel_size))
        else:
            kernel_ops = int(kernel_size)
        in_channels = int(getattr(module, "in_channels", 1))
        groups = int(getattr(module, "groups", 1))
        kernel_ops *= max(in_channels // max(groups, 1), 1)

        flops = 2 * batch * out_channels * spatial * kernel_ops
        if getattr(module, "bias", None) is not None:
            flops += batch * out_channels * spatial
        total_flops += int(flops)

    def linear_hook(module: nn.Linear, _inputs, output):
        nonlocal total_flops
        y = _as_tensor(output)
        if y is None:
            return
        out_elems = int(y.numel())
        in_features = int(module.in_features)
        flops = 2 * out_elems * in_features
        if module.bias is not None:
            flops += out_elems
        total_flops += int(flops)

    def mha_hook(module: nn.MultiheadAttention, inputs, _output):
        nonlocal total_flops
        if len(inputs) < 3:
            return
        q = _as_tensor(inputs[0])
        k = _as_tensor(inputs[1])
        v = _as_tensor(inputs[2])
        if q is None or k is None or v is None:
            return

        if q.ndim == 3:
            if getattr(module, "batch_first", False):
                b, sq, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
                sk = int(k.shape[1])
                sv = int(v.shape[1])
            else:
                sq, b, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
                sk = int(k.shape[0])
                sv = int(v.shape[0])
        elif q.ndim == 2:
            b, sq, d = 1, int(q.shape[0]), int(q.shape[1])
            sk, sv = int(k.shape[0]), int(v.shape[0])
        else:
            return

        h = int(module.num_heads)
        if h <= 0 or d <= 0:
            return
        head_dim = max(d // h, 1)

        # q/k/v projections (out_proj excluded here; counted by Linear hook).
        flops_proj = 2 * b * (sq * d * d + sk * d * d + sv * d * d)

        # attention core: QK^T and Attn*V.
        flops_qk = 2 * b * h * sq * sk * head_dim
        flops_av = 2 * b * h * sq * sv * head_dim

        total_flops += int(flops_proj + flops_qk + flops_av)

    for module in model.modules():
        if isinstance(module, conv_types):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            x = torch.randn(*input_shape, device=device)
            _ = model(x)
            _sync_if_cuda(device)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return int(total_flops)


def benchmark_latency(
    model: nn.Module,
    *,
    input_shape: Tuple[int, ...],
    device: torch.device,
    warmup: int = 10,
    iters: int = 30,
) -> Dict[str, float]:
    model.eval()
    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            _ = model(x)
        _sync_if_cuda(device)

        t0 = time.perf_counter()
        for _ in range(max(iters, 1)):
            _ = model(x)
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - t0

    batch = max(int(input_shape[0]), 1)
    iters = max(int(iters), 1)
    samples = batch * iters

    latency_ms = (elapsed * 1000.0) / iters
    throughput = samples / max(elapsed, 1e-12)

    return {
        "latency_ms_per_batch": float(latency_ms),
        "throughput_samples_per_s": float(throughput),
        "elapsed_s": float(elapsed),
    }


@torch.no_grad()
def evaluate_inference_runtime(
    model: nn.Module,
    dataloader,
    *,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_samples = 0
    total_correct = 0
    total_batches = 0

    _sync_if_cuda(device)
    t0 = time.perf_counter()

    for batch in dataloader:
        data = batch[0].to(device)
        target = batch[1].to(device)
        logits = model(data)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        pred = logits.argmax(dim=1)
        total_samples += int(target.size(0))
        total_correct += int((pred == target).sum().item())
        total_batches += 1

    _sync_if_cuda(device)
    elapsed = time.perf_counter() - t0

    acc = total_correct / max(total_samples, 1)
    throughput = total_samples / max(elapsed, 1e-12)
    latency_batch_ms = (elapsed * 1000.0) / max(total_batches, 1)
    latency_sample_ms = (elapsed * 1000.0) / max(total_samples, 1)

    return {
        "eval_acc": float(acc),
        "eval_time_s": float(elapsed),
        "throughput_samples_per_s": float(throughput),
        "latency_ms_per_batch": float(latency_batch_ms),
        "latency_ms_per_sample": float(latency_sample_ms),
        "num_samples": int(total_samples),
        "num_batches": int(total_batches),
    }
