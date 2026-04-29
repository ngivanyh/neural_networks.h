#ifndef HELPER_INCLUDED
#define HELPER_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/*
  nn_core.h - shared internal helpers for neural_networks.h single-header libraries.

  Design goals:
  - Safe to include in multiple translation units.
  - Safe to embed into generated stb-style headers.
  - Avoid linker conflicts: all functions are static inline.
  - Allow user overrides via macros (allocator/assert/inline/restrict).

  Note: this header intentionally exposes only internal helpers (nn__*).
*/

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* uint32_t, uint64_t */
#include <stdlib.h>  /* malloc, calloc, free */
#include <string.h>  /* memset */

/* -------------------------------- Configuration hooks -------------------------------- */

#ifndef NN_ASSERT
  #include <assert.h>
  #define NN_ASSERT(x) assert(x)
#endif

#ifndef NN_MALLOC
  #define NN_MALLOC(sz) malloc((sz))
#endif

#ifndef NN_CALLOC
  #define NN_CALLOC(n,sz) calloc((n),(sz))
#endif

#ifndef NN_FREE
  #define NN_FREE(p) free((p))
#endif

#ifndef NN_MEMSET
  #define NN_MEMSET(dst, val, sz) memset((dst),(val),(sz))
#endif

#ifndef NN_INLINE
  #if defined(_MSC_VER)
    #define NN_INLINE __forceinline
  #elif defined(__GNUC__) || defined(__clang__)
    #define NN_INLINE __attribute__((always_inline)) inline
  #else
    #define NN_INLINE inline
  #endif
#endif

#ifndef NN_RESTRICT
  #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
    #define NN_RESTRICT restrict
  #elif defined(_MSC_VER)
    #define NN_RESTRICT __restrict
  #elif defined(__GNUC__) || defined(__clang__)
    #define NN_RESTRICT __restrict__
  #else
    #define NN_RESTRICT
  #endif
#endif

/* -------------------------------- Small utilities -------------------------------- */

NN_INLINE static void nn__zero_bytes(void *p, size_t bytes)
{
    if (!p || bytes == 0) return;
    NN_MEMSET(p, 0, bytes);
}

NN_INLINE static void *nn__malloc_or_null(size_t bytes)
{
    if (bytes == 0) return NULL;
    return NN_MALLOC(bytes);
}

NN_INLINE static void *nn__calloc_or_null(size_t n, size_t elem_size)
{
    if (n == 0 || elem_size == 0) return NULL;
    /* overflow check */
    if (elem_size != 0 && n > ((size_t)-1) / elem_size) return NULL;
    return NN_CALLOC(n, elem_size);
}

/* -------------------------------- RNG (xorshift32) -------------------------------- */

/*
  Tiny deterministic RNG suitable for:
  - weight initialization
  - dropout masks (if you add later)
  Not cryptographically secure.

  Usage:
    uint32_t state = 123u;
    uint32_t r = nn__xorshift32(&state);
    double u = nn__rand_uniform01(&state);
*/
NN_INLINE static uint32_t nn__xorshift32(uint32_t *state)
{
    /* xorshift requires non-zero state */
    uint32_t x = (state && *state) ? *state : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    if (state) *state = x;
    return x;
}

/* uniform in [0,1) with ~24 bits of mantissa precision */
NN_INLINE static double nn__rand_uniform01(uint32_t *state)
{
    /* take top 24 bits */
    uint32_t r = nn__xorshift32(state);
    uint32_t mant = (r >> 8) & 0x00FFFFFFu;
    return (double)mant / 16777216.0; /* 2^24 */
}

/* uniform in (a,b) */
NN_INLINE static double nn__rand_uniform(double a, double b, uint32_t *state)
{
    double u = nn__rand_uniform01(state);
    return a + (b - a) * u;
}

/* uniform in [-scale, +scale] */
NN_INLINE static double nn__rand_symmetric(double scale, uint32_t *state)
{
    return nn__rand_uniform(-scale, scale, state);
}

/* -------------------------------- Simple init helpers -------------------------------- */

/* He uniform scale for ReLU: sqrt(6/fan_in) */
NN_INLINE static double nn__he_uniform_scale(size_t fan_in)
{
    if (fan_in == 0) return 0.0;
    /* sqrt(6/fan_in) */
    /* avoid pulling in <math.h> here; caller can do better if needed */
    /* crude Newton-Raphson sqrt for positive numbers (good enough for init) */
    double x = 6.0 / (double)fan_in;
    if (x <= 0.0) return 0.0;
    double g = x > 1.0 ? x : 1.0;
    for (int i = 0; i < 8; ++i) g = 0.5 * (g + x / g);
    return g;
}

/* Xavier/Glorot uniform scale: sqrt(6/(fan_in+fan_out)) */
NN_INLINE static double nn__xavier_uniform_scale(size_t fan_in, size_t fan_out)
{
    size_t denom = fan_in + fan_out;
    if (denom == 0) return 0.0;
    double x = 6.0 / (double)denom;
    if (x <= 0.0) return 0.0;
    double g = x > 1.0 ? x : 1.0;
    for (int i = 0; i < 8; ++i) g = 0.5 * (g + x / g);
    return g;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NN_CORE_INCLUDED */