use candle::{DType, Device, Result, Tensor};
use candle_nn::attention::{
    flash_attn, flash_attn_nhsd_f32, flash_attn_nhsd_f32_batched,
    flash_attn_nhsd_f32_decoder_blocked, AttnMask,
};

/// Reference SDPA: softmax(Q @ K^T * scale) @ V
fn reference_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let att = (q.clone() * scale as f64)?.matmul(&k.clone().t()?)?;
    let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
    att.matmul(&v.clone())
}

/// Reference causal SDPA with mask
fn reference_causal_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (_, _, s_q, _) = q.dims4()?;
    let (_, _, s_kv, _) = k.dims4()?;
    let offset = s_kv - s_q;

    let mask: Vec<f32> = (0..s_q)
        .flat_map(|i| {
            (0..s_kv).map(move |j| {
                if j as isize <= i as isize + offset as isize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, s_q, s_kv), q.device())?;

    let att = (q.clone() * scale as f64)?.matmul(&k.clone().t()?)?;
    let att = att.broadcast_add(&mask)?;
    let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
    att.matmul(&v.clone())
}

fn assert_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) -> Result<()> {
    let a_flat: Vec<f32> = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let b_flat: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let max_diff = a_flat
        .iter()
        .zip(b_flat.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < tol,
        "{label}: max diff {max_diff:.6e} > tol {tol:.6e}"
    );
    Ok(())
}

// Causal decode (q_len=1)

#[test]
fn causal_decode_b1() -> Result<()> {
    let (h, d, kv_len) = (4, 16, 32);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h, 1, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, kv_len, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, kv_len, d), dev)?;

    let expected = reference_sdpa(&q, &k, &v, scale)?;

    // flash_attn expects (B, S, H, D)
    let out = flash_attn::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
        AttnMask::causal_with_offset(kv_len - 1),
        None,
        None,
    )?;
    // Output (B, H, S, D), compare
    assert_close(&out, &expected, 1e-5, "causal_decode_b1")
}

// Causal prefill

#[test]
fn causal_prefill_b1() -> Result<()> {
    let (h, d, s) = (4, 16, 8);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;

    let expected = reference_causal_sdpa(&q, &k, &v, scale)?;

    let out = flash_attn::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
        AttnMask::causal(),
        None,
        None,
    )?;

    assert_close(&out, &expected, 1e-5, "causal_prefill_b1")
}

// Causal prefill with KV offset (simulating KV cache)

#[test]
fn causal_prefill_with_offset() -> Result<()> {
    let (h, d, s_q, s_kv) = (4, 16, 4, 12);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let offset = s_kv - s_q; // 8

    let q = Tensor::randn(0f32, 1f32, (1, h, s_q, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, s_kv, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, s_kv, d), dev)?;

    let expected = reference_causal_sdpa(&q, &k, &v, scale)?;

    let out = flash_attn::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
        AttnMask::causal_with_offset(offset),
        None,
        None,
    )?;

    assert_close(&out, &expected, 1e-5, "causal_prefill_with_offset")
}

// GQA (fewer KV heads)

#[test]
fn causal_decode_gqa() -> Result<()> {
    let (h_q, h_kv, d, kv_len) = (8, 2, 16, 20);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h_q, 1, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h_kv, kv_len, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h_kv, kv_len, d), dev)?;

    // Reference: expand K/V to match Q heads
    let reps = h_q / h_kv;
    let k_exp = k
        .reshape((1, h_kv, 1, kv_len, d))?
        .broadcast_as((1, h_kv, reps, kv_len, d))?
        .reshape((1, h_q, kv_len, d))?;
    let v_exp = v
        .reshape((1, h_kv, 1, kv_len, d))?
        .broadcast_as((1, h_kv, reps, kv_len, d))?
        .reshape((1, h_q, kv_len, d))?;

    let expected = reference_sdpa(&q, &k_exp, &v_exp, scale)?;

    let out = flash_attn::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
        AttnMask::causal_with_offset(kv_len - 1),
        None,
        None,
    )?;

    assert_close(&out, &expected, 1e-5, "causal_decode_gqa")
}

// Standard (no mask)

#[test]
fn standard_no_mask_b1() -> Result<()> {
    let (h, d, s) = (4, 16, 8);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;

    let expected = reference_sdpa(&q, &k, &v, scale)?;

    let out = flash_attn::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
        AttnMask::None,
        None,
        None,
    )?;

    assert_close(&out, &expected, 1e-5, "standard_no_mask_b1")
}

#[test]
fn standard_no_mask_nhsd_is_contiguous() -> Result<()> {
    let (h, d, s) = (4, 16, 8);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;

    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;

    assert_eq!(out.dims(), &[1, s, h, d]);
    assert!(out.is_contiguous());
    assert_close(&out, &expected, 1e-5, "standard_no_mask_nhsd")
}

#[test]
fn standard_no_mask_nhsd_batched_matches_reference() -> Result<()> {
    let (b, h, d, s) = (3, 2, 16, 8);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32_batched(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;

    assert_eq!(out.dims(), &[b, s, h, d]);
    assert!(out.is_contiguous());
    assert_close(&out, &expected, 1e-5, "standard_no_mask_nhsd_batched")
}

#[test]
fn standard_no_mask_nhsd_short_gemm_matches_reference() -> Result<()> {
    // S=64 and B>1 exercise the batched Hiera window GEMMs, including the
    // interleaved NHSD destination strides and contiguous-per-head inputs.
    let (b, h, d, s) = (3, 2, 96, 64);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32_batched(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;
    assert_close(&out, &expected, 2e-4, "standard_no_mask_nhsd_short_gemm")
}

#[test]
fn standard_no_mask_nhsd_asymmetric_gemm_matches_reference() -> Result<()> {
    // Decoder token-to-image attention uses a handful of queries and many
    // image keys; the K/V sequence stride is interleaved across heads.
    let (h, d, q_len, kv_len) = (8, 16, 5, 512);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (1, h, q_len, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, kv_len, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, kv_len, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;
    assert_close(
        &out,
        &expected,
        2e-4,
        "standard_no_mask_nhsd_asymmetric_gemm",
    )
}

#[test]
fn standard_no_mask_nhsd_decoder_blocked_matches_reference() -> Result<()> {
    // Exercise the explicit short-query/long-key blocked route without a
    // process-global environment toggle. This is the shape used by SAM2's
    // batch-one token-to-image decoder attention.
    let (h, d, q_len, kv_len) = (8, 16, 8, 512);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (1, h, q_len, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, kv_len, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, kv_len, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32_decoder_blocked(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;
    assert!(out.is_contiguous());
    assert_close(
        &out,
        &expected,
        2e-4,
        "standard_no_mask_nhsd_decoder_blocked",
    )
}

#[test]
fn standard_no_mask_nhsd_asymmetric_gemm_batched_matches_reference() -> Result<()> {
    // Prompt batches use the same short-query/long-key decoder shape as the
    // batch-one path, with each batch entry owning independent Q/K/V storage.
    let (b, h, d, q_len, kv_len) = (4, 8, 16, 8, 512);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (b, h, q_len, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32_batched(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;
    assert_close(
        &out,
        &expected,
        2e-4,
        "standard_no_mask_nhsd_asymmetric_gemm_batched",
    )
}

#[test]
fn standard_no_mask_nhsd_blocked_matches_reference() -> Result<()> {
    // S=768 selects the blocked f32 NHSD prefill path. The transposed inputs
    // reproduce SAM2's last-dimension-contiguous, leading-strided Q/K/V views.
    let (h, d, s) = (2, 16, 768);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;
    assert!(out.is_contiguous());
    assert_close(&out, &expected, 2e-4, "standard_no_mask_nhsd_blocked")
}

#[test]
fn standard_no_mask_nhsd_blocked_batched_matches_reference() -> Result<()> {
    // B=2 verifies that the blocked long-sequence path honors every leading
    // batch stride rather than reading or writing batch zero twice.
    let (b, h, d, s) = (2, 2, 16, 768);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), dev)?;
    let expected = reference_sdpa(&q, &k, &v, scale)?.permute((0, 2, 1, 3))?;
    let out = flash_attn_nhsd_f32_batched(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
    )?;
    assert!(out.is_contiguous());
    assert_close(
        &out,
        &expected,
        2e-4,
        "standard_no_mask_nhsd_blocked_batched",
    )
}

// Standard with explicit mask

#[test]
fn standard_with_mask_b1() -> Result<()> {
    let (h, d, s) = (2, 8, 4);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h, s, d), dev)?;

    // Causal mask as explicit tensor
    let mask: Vec<f32> = (0..s)
        .flat_map(|i| (0..s).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let mask_tensor = Tensor::from_vec(mask, (1, s, s), dev)?;

    // Reference with mask applied
    let att = (q.clone() * scale as f64)?.matmul(&k.clone().t()?)?;
    let att = att.broadcast_add(&mask_tensor.unsqueeze(0)?)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let expected = att.matmul(&v.clone())?;

    let out = flash_attn::<f32>(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        scale,
        AttnMask::Mask(mask_tensor),
        None,
        None,
    )?;

    assert_close(&out, &expected, 1e-4, "standard_with_mask_b1")
}

// Interleaved KV decode

#[test]
fn interleaved_kv_decode() -> Result<()> {
    use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;

    let (h_q, h_kv, d, kv_len) = (8, 2, 16, 20);
    let scale = 1.0 / (d as f32).sqrt();
    let dev = &Device::Cpu;

    let q = Tensor::randn(0f32, 1f32, (1, h_q, 1, d), dev)?;
    let k = Tensor::randn(0f32, 1f32, (1, h_kv, kv_len, d), dev)?;
    let v = Tensor::randn(0f32, 1f32, (1, h_kv, kv_len, d), dev)?;

    // Reference with GQA expansion
    let reps = h_q / h_kv;
    let k_exp = k
        .reshape((1, h_kv, 1, kv_len, d))?
        .broadcast_as((1, h_kv, reps, kv_len, d))?
        .reshape((1, h_q, kv_len, d))?;
    let v_exp = v
        .reshape((1, h_kv, 1, kv_len, d))?
        .broadcast_as((1, h_kv, reps, kv_len, d))?
        .reshape((1, h_q, kv_len, d))?;
    let expected = reference_sdpa(&q, &k_exp, &v_exp, scale)?;

    // Build interleaved KV: (kv_len, h_kv, 2*d).
    let k_seq = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    let v_seq = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
    let kv = Tensor::cat(&[&k_seq, &v_seq], 2)?.contiguous()?; // (kv_len, h_kv, 2*d)

    let kv_data: Vec<f32> = kv.flatten_all()?.to_vec1()?;
    let q_flat: Vec<f32> = q
        .squeeze(0)?
        .squeeze(1)?
        .contiguous()?
        .flatten_all()?
        .to_vec1()?;

    let out = causal_decode_f32_interleaved(&q_flat, &kv_data, h_q, h_kv, d, kv_len, scale)?;

    let out = out.unsqueeze(0)?;
    assert_close(&out, &expected, 1e-5, "interleaved_kv_decode")
}
