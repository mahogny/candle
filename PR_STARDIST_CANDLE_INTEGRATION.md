# StarDist/Cellpose Candle integration support

## Motivation

`stardist-rs` and the blockflow segmentation distribution use Candle as the
fast CUDA path for model inference. The integration currently needs two classes
of Candle changes:

- Candle must expose the 3D convolution surface used by generated StarDist model
  code. Even a 2D StarDist build currently compiles the Candle model module that
  imports `candle_nn::Conv3d` and `Conv3dConfig`.
- The Segment Anything image encoder should not pay for a generic broadcast
  expression when adding decomposed relative-position terms to the attention
  tensor. CP-SAM/Cellpose workloads spend enough time in that path for the
  specialized CUDA operation to matter.

Keeping these changes in Candle keeps downstream crates faithful to their
original model definitions. Blockflow, Cellpose and StarDist should select
models and devices; they should not fork model semantics or carry local tensor
workarounds for missing backend primitives.

## What Changes

- Adds `Tensor::conv3d` support in Candle core.
- Adds CPU, CUDA and Metal Conv3d backend implementations.
- Adds `candle_nn::Conv3d` and `Conv3dConfig`.
- Fixes CUDA 12.8 half compatibility guards.
- Fixes SAM image encoder batched absolute-position addition.
- Adds a CUDA in-place fast path for SAM decomposed relative-position addition:
  `attn += rel_h + rel_w`.
- Adds tests for Conv3d and for the SAM CUDA relative-position kernel.

## Correctness

The SAM CUDA kernel is a specialization of the existing expression:

```text
attn[b, q, kh, kw] += rel_h[b, q, kh] + rel_w[b, q, kw]
```

It is only selected for CUDA tensors. CPU and non-CUDA paths keep the existing
broadcast expression.

The Conv3d additions are exercised against reference cases for padding, stride,
dilation and groups, plus backend-specific CPU/CUDA/Metal test coverage where
available.

## Review Notes

This branch is consumer-valid for `stardist-rs`, but it may be easier to review
as smaller PRs:

1. CUDA 12.8 half compatibility guard.
2. Candle core and backend Conv3d support.
3. `candle_nn::Conv3d`.
4. SAM batched position embedding correctness fix.
5. SAM CUDA relative-position add specialization.
