#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core.hpp"
namespace py = pybind11;
using namespace minivector;
PYBIND11_MODULE(minivector_core, m) {
    m.doc() = "MiniVector Core SIMD-accelerated backend";
    m.def("detect_simd_id", []() {
        return static_cast<int>(detect_simd());
    }, "Detect the best available SIMD instruction set (returns ID)");
    m.def("get_version", &get_version, "Get the version of the C++ core");
    m.def("batch_search", [](
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> query_vector,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> database_vectors,
        size_t k
    ) {
        auto query_buf = query_vector.request();
        auto db_buf = database_vectors.request();
        if (query_buf.ndim != 1) throw std::runtime_error("Query must be 1D");
        if (db_buf.ndim != 2) throw std::runtime_error("Database must be 2D");
        size_t vector_bytes = query_buf.shape[0];
        size_t num_vectors = db_buf.shape[0];
        if (db_buf.shape[1] != (ssize_t)vector_bytes) 
            throw std::runtime_error("Dimension mismatch");
        auto result = batch_search(
            static_cast<const uint8_t*>(query_buf.ptr),
            static_cast<const uint8_t*>(db_buf.ptr),
            num_vectors,
            vector_bytes,
            k
        );
        py::array_t<int64_t> indices(result.indices.size());
        auto idx_ptr = static_cast<int64_t*>(indices.request().ptr);
        for (size_t i = 0; i < result.indices.size(); ++i) {
            idx_ptr[i] = static_cast<int64_t>(result.indices[i]);
        }
        py::array_t<uint32_t> distances(result.distances.size());
        auto dist_ptr = static_cast<uint32_t*>(distances.request().ptr);
        std::memcpy(dist_ptr, result.distances.data(), result.distances.size() * sizeof(uint32_t));
        return py::make_tuple(indices, distances);
    }, "Perform batch search for a single query");
}
