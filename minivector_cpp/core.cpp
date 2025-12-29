/**
 * =============================================================================
 * MiniVector Core - SIMD-Accelerated Hamming Distance Implementation
 * =============================================================================
 */

#include "core.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <sstream>

#ifdef _MSC_VER
    #include <intrin.h>
    #define POPCOUNT64(x) __popcnt64(x)
    #define POPCOUNT32(x) __popcnt(x)
    #define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
    #include <x86intrin.h>
    #define POPCOUNT64(x) __builtin_popcountll(x)
    #define POPCOUNT32(x) __builtin_popcount(x)
    #define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#endif

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    #include <immintrin.h>
    #define MINIVECTOR_AVX2 1
#endif

#if defined(__SSE2__) || defined(_MSC_VER)
    #include <emmintrin.h>
    #define MINIVECTOR_SSE2 1
#endif

namespace minivector {

const char* get_version() { return "0.2.1-simd"; }

SIMDType detect_simd() {
#ifdef MINIVECTOR_AVX2
    return SIMDType::AVX2;
#elif defined(MINIVECTOR_SSE2)
    return SIMDType::SSE2;
#else
    return SIMDType::NONE;
#endif
}

const char* simd_type_name(SIMDType type) {
    switch (type) {
        case SIMDType::AVX2: return "AVX2";
        case SIMDType::SSE2: return "SSE2";
        case SIMDType::NONE: return "Scalar";
        default: return "Unknown";
    }
}

static inline uint32_t hamming_scalar(const uint8_t* a, const uint8_t* b, size_t bytes) {
    uint32_t dist = 0;
    const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a);
    const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b);
    size_t words = bytes / 8;
    for (size_t i = 0; i < words; ++i) dist += static_cast<uint32_t>(POPCOUNT64(a64[i] ^ b64[i]));
    for (size_t i = words * 8; i < bytes; ++i) dist += POPCOUNT32(a[i] ^ b[i]);
    return dist;
}

#ifdef MINIVECTOR_AVX2
static inline uint32_t hamming_avx2(const uint8_t* a, const uint8_t* b, size_t bytes) {
    const size_t VEC_SIZE = 32;
    size_t vec_count = bytes / VEC_SIZE;
    uint32_t dist = 0;
    for (size_t i = 0; i < vec_count; ++i) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i * VEC_SIZE));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i * VEC_SIZE));
        __m256i xor_res = _mm256_xor_si256(va, vb);
        uint64_t res[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), xor_res);
        for (int j = 0; j < 4; ++j) dist += static_cast<uint32_t>(POPCOUNT64(res[j]));
    }
    size_t processed = vec_count * VEC_SIZE;
    if (processed < bytes) dist += hamming_scalar(a + processed, b + processed, bytes - processed);
    return dist;
}
#endif

uint32_t hamming_distance_single(const uint8_t* a, const uint8_t* b, size_t bytes) {
#ifdef MINIVECTOR_AVX2
    return hamming_avx2(a, b, bytes);
#else
    return hamming_scalar(a, b, bytes);
#endif
}

std::vector<uint32_t> hamming_distance_batch(const uint8_t* q, const uint8_t* db, size_t n, size_t bytes) {
    std::vector<uint32_t> dists(n);
    for (size_t i = 0; i < n; ++i) {
        dists[i] = hamming_distance_single(q, db + i * bytes, bytes);
    }
    return dists;
}

SearchResult batch_search(const uint8_t* q, const uint8_t* db, size_t n, size_t bytes, size_t k) {
    k = std::min(k, n);
    auto dists = hamming_distance_batch(q, db, n, bytes);
    std::vector<size_t> idxs(n);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(), [&](size_t a, size_t b) { return dists[a] < dists[b]; });
    idxs.resize(k);
    SearchResult res;
    res.indices = idxs;
    res.distances.resize(k);
    for (size_t i = 0; i < k; ++i) res.distances[i] = dists[idxs[i]];
    return res;
}

} // namespace minivector
