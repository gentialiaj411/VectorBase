# =============================================================================
# MiniVector C++ Core - High-Performance SIMD Vector Search Engine
# =============================================================================
# This CMake configuration builds the minivector_core Python extension module
# with automatic detection and optimization for AVX2/AVX-512 SIMD instructions.
#
# Architecture: Hybrid Python/C++ for production-grade performance
# Target: 10-20x speedup over NumPy baseline
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(minivector_core LANGUAGES CXX)

# -----------------------------------------------------------------------------
# C++ Standard and Compiler Settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release build by default for maximum performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# -----------------------------------------------------------------------------
# SIMD Detection and Configuration
# -----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

# Detect AVX-512 support
set(AVX512_FLAGS "")
set(AVX2_FLAGS "")
set(SSE2_FLAGS "")

if(MSVC)
    # MSVC compiler flags
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    
    if(COMPILER_SUPPORTS_AVX512)
        set(AVX512_FLAGS "/arch:AVX512")
        message(STATUS "AVX-512 support: ENABLED (MSVC)")
    endif()
    
    if(COMPILER_SUPPORTS_AVX2)
        set(AVX2_FLAGS "/arch:AVX2")
        message(STATUS "AVX2 support: ENABLED (MSVC)")
    endif()
    
    # SSE2 is default on x64 MSVC
    set(SSE2_FLAGS "")
    message(STATUS "SSE2 support: ENABLED (default on x64)")
    
    # MSVC optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    
else()
    # GCC/Clang compiler flags
    check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vpopcntdq" COMPILER_SUPPORTS_AVX512)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
    check_cxx_compiler_flag("-mpopcnt" COMPILER_SUPPORTS_POPCNT)
    
    if(COMPILER_SUPPORTS_AVX512)
        set(AVX512_FLAGS "-mavx512f -mavx512bw -mavx512vpopcntdq")
        message(STATUS "AVX-512 support: ENABLED")
    endif()
    
    if(COMPILER_SUPPORTS_AVX2)
        set(AVX2_FLAGS "-mavx2")
        message(STATUS "AVX2 support: ENABLED")
    endif()
    
    if(COMPILER_SUPPORTS_SSE2)
        set(SSE2_FLAGS "-msse2")
        message(STATUS "SSE2 support: ENABLED")
    endif()
    
    if(COMPILER_SUPPORTS_POPCNT)
        set(POPCNT_FLAGS "-mpopcnt")
        message(STATUS "POPCNT support: ENABLED")
    endif()
    
    # GCC/Clang optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto")
endif()

# Combine SIMD flags (prefer highest available)
set(SIMD_FLAGS "")
if(COMPILER_SUPPORTS_AVX512)
    set(SIMD_FLAGS "${AVX512_FLAGS}")
    add_compile_definitions(USE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2)
    set(SIMD_FLAGS "${AVX2_FLAGS}")
    add_compile_definitions(USE_AVX2)
elseif(COMPILER_SUPPORTS_SSE2)
    set(SIMD_FLAGS "${SSE2_FLAGS}")
    add_compile_definitions(USE_SSE2)
endif()

if(COMPILER_SUPPORTS_POPCNT)
    set(SIMD_FLAGS "${SIMD_FLAGS} ${POPCNT_FLAGS}")
endif()

message(STATUS "SIMD flags: ${SIMD_FLAGS}")

# -----------------------------------------------------------------------------
# Find pybind11
# -----------------------------------------------------------------------------
# Try to find pybind11 via pip installation first
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(pybind11_DIR)
    message(STATUS "Found pybind11 via pip: ${pybind11_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# -----------------------------------------------------------------------------
# Source Files Configuration
# -----------------------------------------------------------------------------
# Core SIMD implementation (modular for HNSW integration)
set(CORE_SOURCES
    core.cpp
)

set(CORE_HEADERS
    core.hpp
)

# Future HNSW sources (Phase 2)
# set(HNSW_SOURCES
#     hnsw/hnsw_index.cpp
#     hnsw/hnsw_graph.cpp
#     hnsw/hnsw_search.cpp
# )

# Python bindings
set(BINDINGS_SOURCES
    bindings.cpp
)

# -----------------------------------------------------------------------------
# Build the Python Extension Module
# -----------------------------------------------------------------------------
pybind11_add_module(minivector_core MODULE
    ${CORE_SOURCES}
    ${BINDINGS_SOURCES}
)

# Apply SIMD flags
target_compile_options(minivector_core PRIVATE
    $<$<CONFIG:Release>:${SIMD_FLAGS}>
    $<$<CONFIG:RelWithDebInfo>:${SIMD_FLAGS}>
)

# Include directories
target_include_directories(minivector_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set output properties
set_target_properties(minivector_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""  # Remove 'lib' prefix on Linux
)

# Platform-specific output suffix
if(WIN32)
    set_target_properties(minivector_core PROPERTIES SUFFIX ".pyd")
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
# Install to the minivector package directory
install(TARGETS minivector_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/minivector
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/minivector  # For Windows DLLs
)

# -----------------------------------------------------------------------------
# Print Configuration Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== MiniVector C++ Core Configuration ===")
message(STATUS "Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "SIMD Support:    ${SIMD_FLAGS}")
message(STATUS "pybind11:        ${pybind11_VERSION}")
message(STATUS "Output:          minivector_core${CMAKE_SHARED_MODULE_SUFFIX}")
message(STATUS "==========================================")
message(STATUS "")

