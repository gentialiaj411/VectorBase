/**
 * =============================================================================
 * MiniVector Core - High-Performance SIMD Vector Search Engine
 * =============================================================================
 * 
 * This header defines the core interfaces for SIMD-accelerated Hamming distance
 * computation. The design is modular to support future HNSW graph index integration.
 * 
 * Architecture:
 *   - Scalar fallback for all platforms
 *   - SSE2 optimization (128-bit SIMD)
 *   - AVX2 optimization (256-bit SIMD) 
 *   - AVX-512 optimization (512-bit SIMD) with VPOPCNT when available
 * 
 * Performance Target: 10-20x speedup over NumPy np.unpackbits + sum
 * =============================================================================
 */

#ifndef MINIVECTOR_CORE_HPP
#define MINIVECTOR_CORE_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include <string>

namespace minivector {

// =============================================================================
// SIMD Capability Detection
// =============================================================================

/**
 * Available SIMD instruction set levels.
 * Used for runtime capability detection and reporting.
 */
enum class SIMDType {
    NONE,       // Scalar only
    SSE2,       // 128-bit SIMD
    AVX2,       // 256-bit SIMD  
    AVX512,     // 512-bit SIMD
    AVX512_VPOPCNT  // 512-bit with native VPOPCNT (best)
};

/**
 * Detect available SIMD instruction set at compile time.
 * Returns the highest supported SIMD level.
 */
SIMDType detect_simd();

/**
 * Get human-readable string for SIMD type.
 */
const char* simd_type_name(SIMDType type);

// =============================================================================
// Core Distance Functions
// =============================================================================

/**
 * Compute Hamming distance between a query vector and multiple database vectors.
 * Uses the best available SIMD instruction set.
 * 
 * @param query_vector      Pointer to bit-packed query vector (uint8_t array)
 * @param database_vectors  Pointer to contiguous bit-packed database vectors
 * @param num_vectors       Number of vectors in the database
 * @param vector_bytes      Size of each vector in bytes (e.g., 48 for 384-dim)
 * @return                  Vector of Hamming distances (one per database vector)
 * 
 * Performance: ~20x faster than NumPy unpackbits + sum for typical workloads
 */
std::vector<uint32_t> hamming_distance_batch(
    const uint8_t* query_vector,
    const uint8_t* database_vectors,
    size_t num_vectors,
    size_t vector_bytes
);

/**
 * Compute Hamming distance between two single vectors.
 * Useful for HNSW graph traversal where pairwise distances are needed.
 * 
 * @param vec_a         Pointer to first bit-packed vector
 * @param vec_b         Pointer to second bit-packed vector  
 * @param vector_bytes  Size of each vector in bytes
 * @return              Hamming distance (number of differing bits)
 */
uint32_t hamming_distance_single(
    const uint8_t* vec_a,
    const uint8_t* vec_b,
    size_t vector_bytes
);

// =============================================================================
// Search Functions
// =============================================================================

/**
 * Result of a top-k search operation.
 */
struct SearchResult {
    std::vector<size_t> indices;      // Indices of top-k vectors
    std::vector<uint32_t> distances;  // Corresponding Hamming distances
};

/**
 * Perform brute-force top-k search using SIMD-accelerated distance computation.
 * Uses partial_sort for O(n + k log k) complexity instead of full sort.
 * 
 * @param query_vector      Pointer to bit-packed query vector
 * @param database_vectors  Pointer to contiguous bit-packed database vectors
 * @param num_vectors       Number of vectors in the database
 * @param vector_bytes      Size of each vector in bytes
 * @param k                 Number of nearest neighbors to return
 * @return                  SearchResult with indices and distances
 */
SearchResult batch_search(
    const uint8_t* query_vector,
    const uint8_t* database_vectors,
    size_t num_vectors,
    size_t vector_bytes,
    size_t k
);

/**
 * Batch search for multiple queries (for throughput benchmarking).
 * Processes queries sequentially but allows for future parallelization.
 * 
 * @param query_vectors     Pointer to contiguous bit-packed query vectors
 * @param database_vectors  Pointer to contiguous bit-packed database vectors
 * @param num_queries       Number of query vectors
 * @param num_db_vectors    Number of database vectors
 * @param vector_bytes      Size of each vector in bytes
 * @param k                 Number of nearest neighbors per query
 * @return                  Vector of SearchResults (one per query)
 */
std::vector<SearchResult> multi_query_search(
    const uint8_t* query_vectors,
    const uint8_t* database_vectors,
    size_t num_queries,
    size_t num_db_vectors,
    size_t vector_bytes,
    size_t k
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get version string for the C++ core.
 */
const char* get_version();

/**
 * Get build configuration info (compiler, SIMD level, etc.).
 */
std::string get_build_info();

// =============================================================================
// HNSW Interface (Phase 2 - Future)
// =============================================================================
// The following interfaces are designed for future HNSW integration.
// They use the same distance functions for consistency.

namespace hnsw {

// Forward declarations for HNSW (to be implemented in Phase 2)
// class HNSWIndex;
// struct HNSWConfig;

} // namespace hnsw

} // namespace minivector

// =============================================================================
// Legacy namespace alias for backwards compatibility
// =============================================================================
namespace minivector_core = minivector;

#endif // MINIVECTOR_CORE_HPP
