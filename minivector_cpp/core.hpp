#ifndef MINIVECTOR_CORE_HPP
#define MINIVECTOR_CORE_HPP
#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include <string>
namespace minivector {
enum class SIMDType {
    NONE,        
    SSE2,        
    AVX2,        
    AVX512,      
    AVX512_VPOPCNT   
};
SIMDType detect_simd();
const char* simd_type_name(SIMDType type);
std::vector<uint32_t> hamming_distance_batch(
    const uint8_t* query_vector,
    const uint8_t* database_vectors,
    size_t num_vectors,
    size_t vector_bytes
);
uint32_t hamming_distance_single(
    const uint8_t* vec_a,
    const uint8_t* vec_b,
    size_t vector_bytes
);
struct SearchResult {
    std::vector<size_t> indices;       
    std::vector<uint32_t> distances;   
};
SearchResult batch_search(
    const uint8_t* query_vector,
    const uint8_t* database_vectors,
    size_t num_vectors,
    size_t vector_bytes,
    size_t k
);
std::vector<SearchResult> multi_query_search(
    const uint8_t* query_vectors,
    const uint8_t* database_vectors,
    size_t num_queries,
    size_t num_db_vectors,
    size_t vector_bytes,
    size_t k
);
const char* get_version();
std::string get_build_info();
namespace hnsw {
}  
}  
namespace minivector_core = minivector;
#endif  
