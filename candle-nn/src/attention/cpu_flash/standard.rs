#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Single-batch (B=1) CPU flash attention kernels. B>1 routes to varlen via mod.rs.

use candle::{Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;
use std::f32;

use super::dot_f32;
use super::online_softmax::online_softmax_step;

/// Fused softmax(qk^T*scale)v on CPU, B=1 only. Output shape (1, H, S, Dv).
pub fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    max_bias: Option<f32>,
    softcap: Option<f32>,
    output_nhsd: bool,
    force_decoder_blocked: bool,
) -> Result<Tensor>
where
    T: WithDType,
{
    let b = q.shape().dims()[0];
    if b != 1 && !output_nhsd {
        candle::bail!(
            "standard::run_flash_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = if let Storage::Cpu(cpu) = &*q_guard {
        let data = cpu.as_slice::<T>()?;
        &data[q_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for q".into()));
    };
    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = if let Storage::Cpu(cpu) = &*k_guard {
        let data = cpu.as_slice::<T>()?;
        &data[k_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for k".into()));
    };
    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = if let Storage::Cpu(cpu) = &*v_guard {
        let data = cpu.as_slice::<T>()?;
        &data[v_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for v".into()));
    };
    // Kernel indexes mask as flat 2D (q_pos * kv_len + kv_pos), so force contiguous.
    let mask_cont = mask.map(|m| m.contiguous()).transpose()?;
    let mask_guard = mask_cont.as_ref().map(|m| m.storage_and_layout());
    let mask_data: Option<&[T]> = if let Some((guard, layout)) = &mask_guard {
        if let Storage::Cpu(cpu) = &**guard {
            let data = cpu.as_slice::<T>()?;
            Some(&data[layout.start_offset()..])
        } else {
            return Err(candle::Error::Msg("Expected CPU storage for mask".into()));
        }
    } else {
        None
    };

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    let max_bias = max_bias.unwrap_or(0.0);
    let logit_softcap = softcap.unwrap_or(0.0);
    let lean = mask_data.is_none() && max_bias == 0.0 && logit_softcap == 0.0;

    if output_nhsd && (q.shape().dims()[1] == 1 || !lean || T::DTYPE != candle::DType::F32) {
        candle::bail!(
            "NHSD flash-attention output requires f32, unmasked prefill with no bias or softcap"
        );
    }

    if q.shape().dims()[1] == 1 {
        if lean {
            return flash_attn_decode_lean::<T>(
                q_data,
                k_data,
                v_data,
                q.shape().dims(),
                k.shape().dims(),
                v.shape().dims(),
                q_stride,
                k_stride,
                v_stride,
                softmax_scale,
            );
        }
        return flash_attn_decode::<T>(
            q_data,
            k_data,
            v_data,
            mask_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            softmax_scale,
            max_bias,
            logit_softcap,
        );
    }

    if lean {
        if T::DTYPE == candle::DType::F32 {
            let qshape = q.shape().dims();
            let kshape = k.shape().dims();
            let vshape = v.shape().dims();
            // SAFETY: the runtime dtype guard above proves each generic slice
            // is backed by f32 values for this specialized kernel.
            let q =
                unsafe { std::slice::from_raw_parts(q_data.as_ptr() as *const f32, q_data.len()) };
            let k =
                unsafe { std::slice::from_raw_parts(k_data.as_ptr() as *const f32, k_data.len()) };
            let v =
                unsafe { std::slice::from_raw_parts(v_data.as_ptr() as *const f32, v_data.len()) };
            return flash_attn_prefill_f32(
                q,
                k,
                v,
                qshape,
                kshape,
                vshape,
                q_stride,
                k_stride,
                v_stride,
                softmax_scale,
                output_nhsd,
                force_decoder_blocked,
            );
        }
        return flash_attn_prefill_lean::<T>(
            q_data,
            k_data,
            v_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            softmax_scale,
        );
    }
    flash_attn_prefill::<T>(
        q_data,
        k_data,
        v_data,
        mask_data,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        q_stride,
        k_stride,
        v_stride,
        softmax_scale,
        max_bias,
        logit_softcap,
    )
}

#[inline]
fn dot_f32_scalar(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[inline]
fn axpy_f32_scalar(acc: &mut [f32], value: &[f32], scale: f32) {
    for (acc, value) in acc.iter_mut().zip(value) {
        *acc += scale * value;
    }
}

#[inline]
fn scale_f32_scalar(acc: &mut [f32], scale: f32) {
    for value in acc {
        *value *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dot_f32_avx512(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAM2 heads are commonly 32 or 64 elements wide. Two independent
    // accumulators hide FMA latency, while reducing in registers avoids a
    // 16-element scalar sum for every attention score.
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut index = 0;
    while index + 32 <= left.len() {
        sum0 = _mm512_fmadd_ps(
            _mm512_loadu_ps(left.as_ptr().add(index)),
            _mm512_loadu_ps(right.as_ptr().add(index)),
            sum0,
        );
        sum1 = _mm512_fmadd_ps(
            _mm512_loadu_ps(left.as_ptr().add(index + 16)),
            _mm512_loadu_ps(right.as_ptr().add(index + 16)),
            sum1,
        );
        index += 32;
    }
    let mut sum = _mm512_add_ps(sum0, sum1);
    while index + 16 <= left.len() {
        sum = _mm512_fmadd_ps(
            _mm512_loadu_ps(left.as_ptr().add(index)),
            _mm512_loadu_ps(right.as_ptr().add(index)),
            sum,
        );
        index += 16;
    }
    _mm512_reduce_add_ps(sum) + dot_f32_scalar(&left[index..], &right[index..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn axpy_f32_avx512(acc: &mut [f32], value: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale = _mm512_set1_ps(scale);
    let mut index = 0;
    while index + 16 <= acc.len() {
        let updated = _mm512_fmadd_ps(
            scale,
            _mm512_loadu_ps(value.as_ptr().add(index)),
            _mm512_loadu_ps(acc.as_ptr().add(index)),
        );
        _mm512_storeu_ps(acc.as_mut_ptr().add(index), updated);
        index += 16;
    }
    axpy_f32_scalar(&mut acc[index..], &value[index..], _mm512_cvtss_f32(scale));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scale_f32_avx512(acc: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale = _mm512_set1_ps(scale);
    let mut index = 0;
    while index + 16 <= acc.len() {
        _mm512_storeu_ps(
            acc.as_mut_ptr().add(index),
            _mm512_mul_ps(_mm512_loadu_ps(acc.as_ptr().add(index)), scale),
        );
        index += 16;
    }
    scale_f32_scalar(&mut acc[index..], _mm512_cvtss_f32(scale));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut index = 0;
    while index + 8 <= left.len() {
        sum = _mm256_fmadd_ps(
            _mm256_loadu_ps(left.as_ptr().add(index)),
            _mm256_loadu_ps(right.as_ptr().add(index)),
            sum,
        );
        index += 8;
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.into_iter().sum::<f32>() + dot_f32_scalar(&left[index..], &right[index..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(acc: &mut [f32], value: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale = _mm256_set1_ps(scale);
    let mut index = 0;
    while index + 8 <= acc.len() {
        let updated = _mm256_fmadd_ps(
            scale,
            _mm256_loadu_ps(value.as_ptr().add(index)),
            _mm256_loadu_ps(acc.as_ptr().add(index)),
        );
        _mm256_storeu_ps(acc.as_mut_ptr().add(index), updated);
        index += 8;
    }
    axpy_f32_scalar(&mut acc[index..], &value[index..], scale_f32(scale));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_f32_avx2(acc: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale = _mm256_set1_ps(scale);
    let mut index = 0;
    while index + 8 <= acc.len() {
        _mm256_storeu_ps(
            acc.as_mut_ptr().add(index),
            _mm256_mul_ps(_mm256_loadu_ps(acc.as_ptr().add(index)), scale),
        );
        index += 8;
    }
    scale_f32_scalar(&mut acc[index..], scale_f32(scale));
}

/// Rescale an online-softmax accumulator with AVX2 when available.  This is
/// kept separate from the row-wise fallback because the blocked kernel owns
/// its accumulator tile directly.  AVX-512 is intentionally avoided here:
/// SAM2's short head accumulators do not repay its frequency transition on
/// Skylake-SP CPUs.
#[inline]
fn rescale_f32_auto(acc: &mut [f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guards the target-feature
            // implementation; the helper handles arbitrary slice tails.
            unsafe { scale_f32_avx2(acc, scale) }
        } else {
            scale_f32_scalar(acc, scale)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    scale_f32_scalar(acc, scale);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_f32(value: std::arch::x86_64::__m256) -> f32 {
    std::arch::x86_64::_mm_cvtss_f32(std::arch::x86_64::_mm256_castps256_ps128(value))
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill_f32(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    output_nhsd: bool,
    force_decoder_blocked: bool,
) -> Result<Tensor> {
    let (batch, q_len, heads, dim) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let key_heads = kshape[2];
    let value_heads = vshape[2];
    let key_ratio = heads / key_heads;
    let value_ratio = heads / value_heads;
    let avx2 = cfg!(target_arch = "x86_64")
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma");
    let avx512 = cfg!(target_arch = "x86_64")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("fma")
        // Skylake-SP can downclock aggressively for AVX-512. Keep an
        // attribution switch so deployments can select the measured winner
        // without changing source; AVX-512 remains the default when present.
        && std::env::var_os("CANDLE_CPU_FLASH_AVX512").is_none_or(|value| value != "0");
    // SAM2 obtains Q/K/V as interleaved views of one projection. K and V are
    // then reread for every query, so pack them once into head-major rows when
    // their sequence stride is not dense. This is a bounded O(S*H*D) copy
    // that replaces O(S²*H*D) strided reads in long prefill attention.
    // PyTorch's f32 kernel consumes these last-dimension-contiguous views with
    // their leading strides. Keep that zero-copy policy by default; bounded
    // packing remains an explicit A/B experiment for a shape where it wins.
    let pack_strided_kv =
        std::env::var_os("CANDLE_CPU_FLASH_PACK_KV").is_some_and(|value| value == "1");
    let packed_k = (batch == 1 && pack_strided_kv && kstride[1] != key_heads * dim).then(|| {
        let mut packed = vec![0f32; key_heads * kv_len * dim];
        packed
            .par_chunks_mut(kv_len * dim)
            .enumerate()
            .for_each(|(head, destination)| {
                for key_pos in 0..kv_len {
                    let source = key_pos * kstride[1] + head * kstride[2];
                    let offset = key_pos * dim;
                    destination[offset..offset + dim]
                        .copy_from_slice(&k_data[source..source + dim]);
                }
            });
        packed
    });
    let packed_v = (batch == 1 && pack_strided_kv && vstride[1] != value_heads * dim).then(|| {
        let mut packed = vec![0f32; value_heads * kv_len * dim];
        packed
            .par_chunks_mut(kv_len * dim)
            .enumerate()
            .for_each(|(head, destination)| {
                for key_pos in 0..kv_len {
                    let source = key_pos * vstride[1] + head * vstride[2];
                    let offset = key_pos * dim;
                    destination[offset..offset + dim]
                        .copy_from_slice(&v_data[source..source + dim]);
                }
            });
        packed
    });
    // For SAM2 global attention, a row-wise dot/AXPY loop leaves most of the
    // CPU's matrix-multiply throughput unused.  The NHSD consumer is the
    // no-copy path, so tile by query positions there: each Rayon task owns a
    // contiguous output region containing every head for its query tile.
    // Keep the conventional NHSD->HSD API on the established row kernel until
    // it has a similarly no-copy blocked output arrangement.
    // Retain a process-level A/B switch for reproducible kernel attribution.
    // It is intentionally opt-out: supported long NHSD inference selects the
    // blocked implementation unless a benchmark sets it to `0`.
    let blocked_enabled =
        std::env::var_os("CANDLE_CPU_FLASH_BLOCKED").is_none_or(|value| value != "0");
    let decoder_asymmetric = (2..=16).contains(&q_len) && (512..=8192).contains(&kv_len);
    let decoder_blocked = decoder_asymmetric
        && (force_decoder_blocked
            || std::env::var_os("CANDLE_CPU_FLASH_DECODER_BLOCKED")
                .is_some_and(|value| value == "1"));
    if blocked_enabled
        && output_nhsd
        && (q_len >= 768 || decoder_blocked)
        && qstride[3] == 1
        && kstride[3] == 1
        && vstride[3] == 1
    {
        return flash_attn_prefill_f32_blocked(
            q_data,
            k_data,
            v_data,
            batch,
            q_len,
            kv_len,
            heads,
            dim,
            key_ratio,
            value_ratio,
            qstride,
            kstride,
            vstride,
            packed_k.as_deref(),
            packed_v.as_deref(),
            scale,
        );
    }
    // Hiera's window attention has many independent short batch entries.
    // Give each `(batch, head)` pair to one Rayon task and use two serial
    // GEMMs with a reusable score matrix. P×V writes directly to its NHSD
    // destination through strides, so no output-sized temporary is needed.
    let short_gemm_enabled =
        std::env::var_os("CANDLE_CPU_FLASH_SHORT_GEMM").is_none_or(|value| value != "0");
    let short_window = batch > 1 && (16..=256).contains(&q_len) && kv_len <= 256;
    if short_gemm_enabled
        && output_nhsd
        && (short_window || decoder_asymmetric)
        && qstride[3] == 1
        && kstride[3] == 1
        && vstride[3] == 1
    {
        return flash_attn_prefill_f32_short_gemm(
            q_data,
            k_data,
            v_data,
            batch,
            q_len,
            kv_len,
            heads,
            dim,
            key_ratio,
            value_ratio,
            qstride,
            kstride,
            vstride,
            scale,
        );
    }
    // Hiera's early windowed blocks arrive as many independent batch entries
    // with a short (usually 64-token) sequence.  A two-pass row softmax keeps
    // only one short thread-local score buffer, avoids Q/K/V packing, and can
    // use a vector exp implementation.  The long B=1 blocked route above is
    // intentionally unchanged.
    let short_two_pass =
        std::env::var_os("CANDLE_CPU_FLASH_SHORT_TWO_PASS").is_some_and(|value| value == "1");
    if short_two_pass && output_nhsd && batch > 1 && q_len <= 256 && kv_len <= 256 {
        return flash_attn_prefill_f32_short(
            q_data,
            k_data,
            v_data,
            q_len,
            kv_len,
            heads,
            dim,
            key_ratio,
            value_ratio,
            qstride,
            kstride,
            vstride,
            scale,
        );
    }
    let mut output = vec![0f32; batch * heads * q_len * dim];
    output
        .par_chunks_mut(dim)
        // Adjacent rows of one 8x8 window reuse its interleaved K/V region.
        // One complete window per Rayon task balances that locality with the
        // many independent windows in early Hiera blocks.
        .with_min_len(64)
        .enumerate()
        .for_each_init(
            || vec![0f32; dim],
            |acc, (row, output)| {
                let (batch_index, head, query_pos) = if output_nhsd {
                    let batch_index = row / (q_len * heads);
                    let within_batch = row % (q_len * heads);
                    (batch_index, within_batch % heads, within_batch / heads)
                } else {
                    let batch_index = row / (heads * q_len);
                    let within_batch = row % (heads * q_len);
                    (batch_index, within_batch / q_len, within_batch % q_len)
                };
                let query_base =
                    batch_index * qstride[0] + query_pos * qstride[1] + head * qstride[2];
                let query = &q_data[query_base..query_base + dim];
                let key_head = head / key_ratio;
                let value_head = head / value_ratio;
                let mut max = f32::NEG_INFINITY;
                let mut sum = 0f32;
                for key_pos in 0..kv_len {
                    let key = if let Some(packed) = &packed_k {
                        let offset = (key_head * kv_len + key_pos) * dim;
                        &packed[offset..offset + dim]
                    } else {
                        let key_base =
                            batch_index * kstride[0] + key_pos * kstride[1] + key_head * kstride[2];
                        &k_data[key_base..key_base + dim]
                    };
                    let score = {
                        #[cfg(target_arch = "x86_64")]
                        if avx512 {
                            unsafe { dot_f32_avx512(query, key) }
                        } else if avx2 {
                            unsafe { dot_f32_avx2(query, key) }
                        } else {
                            dot_f32_scalar(query, key)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            dot_f32_scalar(query, key)
                        }
                    } * scale;
                    let rescale = if score > max {
                        let factor = (max - score).exp();
                        max = score;
                        factor
                    } else {
                        1.
                    };
                    if rescale != 1. {
                        #[cfg(target_arch = "x86_64")]
                        if avx512 {
                            unsafe { scale_f32_avx512(acc, rescale) }
                        } else if avx2 {
                            unsafe { scale_f32_avx2(acc, rescale) }
                        } else {
                            scale_f32_scalar(acc, rescale)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        scale_f32_scalar(acc, rescale);
                    }
                    let weight = (score - max).exp();
                    let value = if let Some(packed) = &packed_v {
                        let offset = (value_head * kv_len + key_pos) * dim;
                        &packed[offset..offset + dim]
                    } else {
                        let value_base = batch_index * vstride[0]
                            + key_pos * vstride[1]
                            + value_head * vstride[2];
                        &v_data[value_base..value_base + dim]
                    };
                    #[cfg(target_arch = "x86_64")]
                    if avx512 {
                        unsafe { axpy_f32_avx512(acc, value, weight) }
                    } else if avx2 {
                        unsafe { axpy_f32_avx2(acc, value, weight) }
                    } else {
                        axpy_f32_scalar(acc, value, weight)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    axpy_f32_scalar(acc, value, weight);
                    sum = sum * rescale + weight;
                }
                let inverse = 1. / sum;
                for (output, value) in output.iter_mut().zip(acc) {
                    *output = *value * inverse;
                }
            },
        );
    let shape = if output_nhsd {
        (batch, q_len, heads, dim)
    } else {
        (batch, heads, q_len, dim)
    };
    Tensor::from_vec(output, shape, &Device::Cpu)
}

/// Exact two-pass softmax for independently batched short f32 NHSD sequences.
///
/// The one `kv_len` score buffer is worker-local.  In particular, this does
/// not materialize an attention matrix or pack the interleaved SAM2 Q/K/V
/// views.  It is faster than online rescaling at Hiera's fixed small windows
/// and gives vector math backends a contiguous exp input.
#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill_f32_short(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    q_len: usize,
    kv_len: usize,
    heads: usize,
    dim: usize,
    key_ratio: usize,
    value_ratio: usize,
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
) -> Result<Tensor> {
    let batch = q_data.len() / qstride[0];
    let avx2 = cfg!(target_arch = "x86_64")
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma");
    let mut output = vec![0f32; batch * q_len * heads * dim];
    output
        .par_chunks_mut(dim)
        .with_min_len(if kv_len >= 512 { 1 } else { 64 })
        .enumerate()
        .for_each_init(
            || vec![0f32; kv_len],
            |scores, (row, destination)| {
                let batch_index = row / (q_len * heads);
                let within_batch = row % (q_len * heads);
                let head = within_batch % heads;
                let query_pos = within_batch / heads;
                let query_base =
                    batch_index * qstride[0] + query_pos * qstride[1] + head * qstride[2];
                let query = &q_data[query_base..query_base + dim];
                let key_head = head / key_ratio;
                let value_head = head / value_ratio;
                for (key_pos, score) in scores.iter_mut().enumerate() {
                    let key_base =
                        batch_index * kstride[0] + key_pos * kstride[1] + key_head * kstride[2];
                    let key = &k_data[key_base..key_base + dim];
                    *score = {
                        #[cfg(target_arch = "x86_64")]
                        if avx2 {
                            // SAFETY: runtime feature detection guards the
                            // AVX2/FMA implementation and slices are valid.
                            unsafe { dot_f32_avx2(query, key) }
                        } else {
                            dot_f32_scalar(query, key)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            dot_f32_scalar(query, key)
                        }
                    } * scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for score in scores.iter_mut() {
                    *score -= maximum;
                }
                #[cfg(feature = "glibc-vector-math")]
                let vector_exp = candle::cpu::erf::exp_f32_inplace_glibc(scores);
                #[cfg(not(feature = "glibc-vector-math"))]
                let vector_exp = false;
                if !vector_exp {
                    for score in scores.iter_mut() {
                        *score = score.exp();
                    }
                }
                let inverse_sum = 1. / scores.iter().sum::<f32>();
                for (key_pos, weight) in scores.iter().enumerate() {
                    let value_base =
                        batch_index * vstride[0] + key_pos * vstride[1] + value_head * vstride[2];
                    let value = &v_data[value_base..value_base + dim];
                    let weight = *weight * inverse_sum;
                    #[cfg(target_arch = "x86_64")]
                    if avx2 {
                        // SAFETY: same feature and slice bounds rationale as
                        // the dot product above.
                        unsafe { axpy_f32_avx2(destination, value, weight) }
                    } else {
                        axpy_f32_scalar(destination, value, weight)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    axpy_f32_scalar(destination, value, weight);
                }
            },
        );
    Tensor::from_vec(output, (batch, q_len, heads, dim), &Device::Cpu)
}

/// Batched short-window attention with reusable scores and direct NHSD output.
#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill_f32_short_gemm(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    batch: usize,
    q_len: usize,
    kv_len: usize,
    heads: usize,
    dim: usize,
    key_ratio: usize,
    value_ratio: usize,
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
) -> Result<Tensor> {
    use gemm::{gemm, Parallelism};

    let mut output = vec![0f32; batch * q_len * heads * dim];
    let output_ptr = output.as_mut_ptr() as usize;
    (0..batch * heads).into_par_iter().for_each_init(
        || vec![0f32; q_len * kv_len],
        |scores, work_item| {
            let batch_index = work_item / heads;
            let head = work_item % heads;
            let key_head = head / key_ratio;
            let value_head = head / value_ratio;
            let query = unsafe {
                q_data
                    .as_ptr()
                    .add(batch_index * qstride[0] + head * qstride[2])
            };
            let key = unsafe {
                k_data
                    .as_ptr()
                    .add(batch_index * kstride[0] + key_head * kstride[2])
            };
            let value = unsafe {
                v_data
                    .as_ptr()
                    .add(batch_index * vstride[0] + value_head * vstride[2])
            };
            unsafe {
                // SAFETY: the dispatch gate proves unit final strides;
                // tensor layouts supply in-bounds leading strides. Each task
                // owns its score workspace and a distinct NHSD output head.
                gemm(
                    q_len,
                    kv_len,
                    dim,
                    scores.as_mut_ptr(),
                    1,
                    kv_len as isize,
                    false,
                    query,
                    1,
                    qstride[1] as isize,
                    key,
                    kstride[1] as isize,
                    1,
                    0.0f32,
                    scale,
                    false,
                    false,
                    false,
                    Parallelism::None,
                );
            }
            for row in scores.chunks_mut(kv_len) {
                let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for score in row.iter_mut() {
                    *score -= maximum;
                }
                #[cfg(feature = "glibc-vector-math")]
                let vector_exp = candle::cpu::erf::exp_f32_inplace_glibc(row);
                #[cfg(not(feature = "glibc-vector-math"))]
                let vector_exp = false;
                if !vector_exp {
                    for score in row.iter_mut() {
                        *score = score.exp();
                    }
                }
                let inverse_sum = 1.0 / row.iter().sum::<f32>();
                for score in row.iter_mut() {
                    *score *= inverse_sum;
                }
            }
            let destination =
                unsafe { (output_ptr as *mut f32).add((batch_index * q_len * heads + head) * dim) };
            unsafe {
                // SAFETY: each `(batch, head)` task writes distinct elements
                // of the interleaved output; GEMM receives its exact strides.
                gemm(
                    q_len,
                    dim,
                    kv_len,
                    destination,
                    1,
                    (heads * dim) as isize,
                    false,
                    scores.as_ptr(),
                    1,
                    kv_len as isize,
                    value,
                    1,
                    vstride[1] as isize,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    Parallelism::None,
                );
            }
        },
    );
    Tensor::from_vec(output, (batch, q_len, heads, dim), &Device::Cpu)
}

/// Blocked f32 online-softmax prefill for the direct NHSD output path.
///
/// The algorithm is deliberately bounded: a task owns one `(head, query-tile)`
/// pair and reuses a `Q_BLOCK × KV_BLOCK` score matrix plus a
/// `Q_BLOCK × D` output accumulator.  The two GEMMs are single-threaded while
/// Rayon distributes query tiles, avoiding nested parallelism.
#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill_f32_blocked(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    batch: usize,
    q_len: usize,
    kv_len: usize,
    heads: usize,
    dim: usize,
    key_ratio: usize,
    value_ratio: usize,
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    packed_k: Option<&[f32]>,
    packed_v: Option<&[f32]>,
    scale: f32,
) -> Result<Tensor> {
    use gemm::{gemm, Parallelism};

    // SAM2's long-token attention is modestly faster at this granularity: it
    // gives Rayon twice as many independent head/tile jobs without making the
    // score scratch large enough to hurt cache locality.
    const DEFAULT_Q_BLOCK: usize = 128;
    // A 1024-key tile halves the online-softmax/GEMM setup work on SAM2's
    // long prefill attention. The worker-local score allocation remains
    // bounded, and deployments can retain 512 through the runtime override
    // when memory is more important than this measured CPU throughput gain.
    const DEFAULT_KV_BLOCK: usize = 1024;
    // Process-level A/B control for the in-tree GEMM scheduler. Restrict the
    // range so each task keeps bounded scratch and sufficient arithmetic work.
    let decoder_asymmetric = q_len <= 16 && kv_len >= 512;
    let q_block = std::env::var("CANDLE_CPU_FLASH_Q_BLOCK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| (32..=512).contains(&value) && value.is_power_of_two())
        .unwrap_or(if decoder_asymmetric {
            32
        } else {
            DEFAULT_Q_BLOCK
        });
    let kv_block = std::env::var("CANDLE_CPU_FLASH_KV_BLOCK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| (128..=1024).contains(&value) && value.is_power_of_two())
        .unwrap_or(if decoder_asymmetric {
            512
        } else {
            DEFAULT_KV_BLOCK
        });
    let mut output = vec![0f32; batch * q_len * heads * dim];
    let output_ptr = output.as_mut_ptr() as usize;
    let query_blocks = q_len.div_ceil(q_block);

    // Keep the tile workspace local to a Rayon worker and reuse it for every
    // work item that worker receives.  GEMM overwrites the live score region,
    // so allocating and zeroing a `Q_BLOCK × KV_BLOCK` buffer for every
    // `(head, query-tile)` only adds allocator and memory-bandwidth traffic.
    (0..batch * heads * query_blocks)
        .into_par_iter()
        .for_each_init(
            || {
                (
                    vec![0f32; q_block * kv_block],
                    vec![0f32; q_block * dim],
                    vec![f32::NEG_INFINITY; q_block],
                    vec![0f32; q_block],
                )
            },
            |(scores, accumulator, maxima, sums), work_item| {
                let batch_index = work_item / (heads * query_blocks);
                let within_batch = work_item % (heads * query_blocks);
                let head = within_batch / query_blocks;
                let query_block = within_batch % query_blocks;
                let query_start = query_block * q_block;
                let query_rows = (q_len - query_start).min(q_block);
                let key_head = head / key_ratio;
                let value_head = head / value_ratio;
                let query = unsafe {
                    // SAFETY: the checked stride/dimension gate above and the
                    // tensor layouts guarantee that the final queried element
                    // is in `q_data`; GEMM receives the same row stride.
                    q_data.as_ptr().add(
                        batch_index * qstride[0] + query_start * qstride[1] + head * qstride[2],
                    )
                };
                accumulator[..query_rows * dim].fill(0.0);
                maxima[..query_rows].fill(f32::NEG_INFINITY);
                sums[..query_rows].fill(0.0);

                for key_start in (0..kv_len).step_by(kv_block) {
                    let key_rows = (kv_len - key_start).min(kv_block);
                    let key = if let Some(packed) = packed_k {
                        unsafe {
                            // SAFETY: `packed` is head-major `(H,S,D)`.
                            packed.as_ptr().add((key_head * kv_len + key_start) * dim)
                        }
                    } else {
                        unsafe {
                            // SAFETY: K's final dimension is contiguous and
                            // the leading strides index an in-bounds view.
                            k_data.as_ptr().add(
                                batch_index * kstride[0]
                                    + key_start * kstride[1]
                                    + key_head * kstride[2],
                            )
                        }
                    };
                    let key_col_stride = packed_k.map_or(kstride[1], |_| dim);
                    unsafe {
                        // C[m,n] = A[m,k] * B[k,n], where B's columns are K
                        // rows. `Parallelism::None` is essential: Rayon owns
                        // concurrency at the query-tile level.
                        gemm(
                            query_rows,
                            key_rows,
                            dim,
                            scores.as_mut_ptr(),
                            1,
                            kv_block as isize,
                            false,
                            query,
                            1,
                            qstride[1] as isize,
                            key,
                            key_col_stride as isize,
                            1,
                            0.0f32,
                            scale,
                            false,
                            false,
                            false,
                            Parallelism::None,
                        );
                    }

                    for row in 0..query_rows {
                        let score_row = &mut scores[row * kv_block..row * kv_block + key_rows];
                        let row_max = score_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let next_max = maxima[row].max(row_max);
                        let rescale = (maxima[row] - next_max).exp();
                        for score in score_row.iter_mut() {
                            *score -= next_max;
                        }
                        #[cfg(feature = "glibc-vector-math")]
                        let vector_exp = candle::cpu::erf::exp_f32_inplace_glibc(score_row);
                        #[cfg(not(feature = "glibc-vector-math"))]
                        let vector_exp = false;
                        if !vector_exp {
                            for score in score_row.iter_mut() {
                                *score = score.exp();
                            }
                        }
                        let tile_sum = score_row.iter().sum::<f32>();
                        if rescale != 1.0 {
                            rescale_f32_auto(&mut accumulator[row * dim..(row + 1) * dim], rescale);
                        }
                        maxima[row] = next_max;
                        sums[row] = sums[row] * rescale + tile_sum;
                    }

                    let value = if let Some(packed) = packed_v {
                        unsafe {
                            // SAFETY: `packed` is head-major `(H,S,D)`.
                            packed.as_ptr().add((value_head * kv_len + key_start) * dim)
                        }
                    } else {
                        unsafe {
                            // SAFETY: V's final dimension is contiguous and
                            // the leading strides index an in-bounds view.
                            v_data.as_ptr().add(
                                batch_index * vstride[0]
                                    + key_start * vstride[1]
                                    + value_head * vstride[2],
                            )
                        }
                    };
                    let value_row_stride = packed_v.map_or(vstride[1], |_| dim);
                    unsafe {
                        gemm(
                            query_rows,
                            dim,
                            key_rows,
                            accumulator.as_mut_ptr(),
                            1,
                            dim as isize,
                            true,
                            scores.as_ptr(),
                            1,
                            kv_block as isize,
                            value,
                            1,
                            value_row_stride as isize,
                            1.0f32,
                            1.0f32,
                            false,
                            false,
                            false,
                            Parallelism::None,
                        );
                    }
                }

                for row in 0..query_rows {
                    let destination_offset =
                        ((batch_index * q_len + query_start + row) * heads + head) * dim;
                    let destination = unsafe {
                        // SAFETY: work items have a unique `(head, query_block)`;
                        // therefore their `query_rows × D` NHSD regions are
                        // disjoint. The pointer derives from `output`, which stays
                        // allocated until Rayon completes.
                        std::slice::from_raw_parts_mut(
                            (output_ptr as *mut f32).add(destination_offset),
                            dim,
                        )
                    };
                    let inverse_sum = 1.0 / sums[row];
                    for (destination, value) in destination
                        .iter_mut()
                        .zip(&accumulator[row * dim..(row + 1) * dim])
                    {
                        *destination = *value * inverse_sum;
                    }
                }
            },
        );
    Tensor::from_vec(output, (batch, q_len, heads, dim), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_decode_lean<T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
) -> Result<Tensor> {
    let (h, d) = (qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;
    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * dv];

    out.par_chunks_mut(dv).enumerate().for_each_init(
        || vec![0f32; dv],
        |acc, (h_i, out_chunk)| {
            let k_head = h_i / rk2;
            let v_head = h_i / rv2;

            let q_base = h_i * qstride[2];
            let q_row = &q_data[q_base..q_base + d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                let k_row = &k_data[k_base..k_base + d];

                let score = dot_f32(q_row, k_row) * scale;

                let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    if v_contiguous {
                        let v_row = &v_data[v_base..v_base + dv];
                        for d_i in 0..dv {
                            acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                        }
                    } else {
                        for d_i in 0..dv {
                            acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                        }
                    }
                });
            }

            let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                *out_v = *acc_v * inv_s;
            }
        },
    );

    Tensor::from_vec(out, (1usize, h, 1usize, dv), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_decode<T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let (h, d) = (qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * dv];

    out.par_chunks_mut(dv).enumerate().for_each_init(
        || vec![0f32; dv],
        |acc, (h_i, out_chunk)| {
            let slope = if max_bias > 0.0 {
                2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
            } else {
                1.0
            };

            let k_head = h_i / rk2;
            let v_head = h_i / rv2;

            let q_base = h_i * qstride[2];
            let q_row = &q_data[q_base..q_base + d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let mv = if let Some(mv_vec) = mask_vec {
                    let mval = mv_vec[kv_pos];
                    slope * mval.to_f64() as f32
                } else {
                    0.0
                };
                if mv == f32::NEG_INFINITY {
                    continue;
                }

                let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                let k_row = &k_data[k_base..k_base + d];

                let mut score = dot_f32(q_row, k_row) * scale_pre;
                if do_softcap {
                    score = logit_softcap * score.tanh();
                }
                score += mv;

                let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    if v_contiguous {
                        let v_row = &v_data[v_base..v_base + dv];
                        for d_i in 0..dv {
                            acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                        }
                    } else {
                        for d_i in 0..dv {
                            acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                        }
                    }
                });
            }

            let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                *out_v = *acc_v * inv_s;
            }
        },
    );

    Tensor::from_vec(out, (1usize, h, 1usize, dv), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill_lean<T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
) -> Result<Tensor> {
    let (q_len, h, d) = (qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;
    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * q_len * dv];

    out.par_chunks_mut(dv)
        .with_min_len(64)
        .enumerate()
        .for_each_init(
            || vec![0f32; dv],
            |acc, (row_idx, out_chunk)| {
                let h_i = row_idx / q_len;
                let q_pos = row_idx % q_len;

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                let q_base = q_pos * qstride[1] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;

                for kv_pos in 0..kv_len {
                    let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row = &k_data[k_base..k_base + d];

                    let score = dot_f32(q_row, k_row) * scale;

                    let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        if v_contiguous {
                            let v_row = &v_data[v_base..v_base + dv];
                            for d_i in 0..dv {
                                acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                            }
                        } else {
                            for d_i in 0..dv {
                                acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                            }
                        }
                    });
                }

                let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                    *out_v = *acc_v * inv_s;
                }
            },
        );

    Tensor::from_vec(out, (1usize, h, q_len, dv), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill<T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let (q_len, h, d) = (qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * q_len * dv];

    out.par_chunks_mut(dv)
        .with_min_len(64)
        .enumerate()
        .for_each_init(
            || vec![0f32; dv],
            |acc, (row_idx, out_chunk)| {
                let h_i = row_idx / q_len;
                let q_pos = row_idx % q_len;

                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                let q_base = q_pos * qstride[1] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;

                for kv_pos in 0..kv_len {
                    let mv = if let Some(mv_vec) = mask_vec {
                        let mval = mv_vec[q_pos * kv_len + kv_pos];
                        slope * mval.to_f64() as f32
                    } else {
                        0.0
                    };
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

                    let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row = &k_data[k_base..k_base + d];

                    let mut score = dot_f32(q_row, k_row) * scale_pre;
                    if do_softcap {
                        score = logit_softcap * score.tanh();
                    }
                    score += mv;

                    let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        if v_contiguous {
                            let v_row = &v_data[v_base..v_base + dv];
                            for d_i in 0..dv {
                                acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                            }
                        } else {
                            for d_i in 0..dv {
                                acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                            }
                        }
                    });
                }

                let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                    *out_v = *acc_v * inv_s;
                }
            },
        );

    Tensor::from_vec(out, (1usize, h, q_len, dv), &Device::Cpu)
}
