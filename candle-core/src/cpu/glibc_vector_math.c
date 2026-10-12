#include <immintrin.h>
#include <math.h>

extern __m512 _ZGVeN16v_erff(__m512);

void candle_glibc_gelu_erf_f32x16(const float *input, float *output) {
    const __m512 values = _mm512_loadu_ps(input);
    const __m512 scaled = _mm512_mul_ps(values, _mm512_set1_ps(0.7071067811865475244f));
    const __m512 erf = _ZGVeN16v_erff(scaled);
    const __m512 gelu = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_add_ps(erf, _mm512_set1_ps(1.0f)), _mm512_set1_ps(0.5f)),
        values);
    _mm512_storeu_ps(output, gelu);
}

extern __m512 _ZGVeN16v_expf(__m512);

void candle_glibc_expf_inplace(float *values, size_t len) {
    size_t index = 0;
    for (; index + 16 <= len; index += 16) {
        _mm512_storeu_ps(values + index, _ZGVeN16v_expf(_mm512_loadu_ps(values + index)));
    }
    for (; index < len; ++index) values[index] = expf(values[index]);
}
