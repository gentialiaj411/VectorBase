# 📊 VectorBase Performance Report
**Date:** Mon Dec 29 13:57:54 2025
**System:** Windows (Local Run)
--------------------------------------------------

## 1️⃣ Data Pipeline Performance
**Test:** Indexing 10,000 documents (Build & Quantize)

```text
Starting index build...
  -> Initializing Dummy Embedder (Random Vectors)
✨ Index build complete in 0.47 seconds!
```

## 2️⃣ C++ Core Performance
**Test:** 100,000 vectors, AVX2 vs NumPy (100 queries)

```text
======================================================================
MiniVector SIMD Performance Benchmark
======================================================================

Backend: C++ SIMD
SIMD: AVX2
C++ available: True
Creating synthetic dataset: 100,000 vectors, 384 dimensions

Running 10 warmup queries...
Running 100 benchmark queries...

======================================================================
SIMD BENCHMARK RESULTS
======================================================================
Database size: 100,000 vectors
Queries: 100
----------------------------------------------------------------------

         NumPy (baseline)          
-----------------------------------
  Average latency:    36.043 ms
  P50 latency:        35.877 ms
  P99 latency:        39.182 ms
  Min latency:        33.106 ms
  Max latency:        39.186 ms
  Throughput:           27.7 QPS

            C++ (AVX2)             
-----------------------------------
  Average latency:     0.661 ms
  P50 latency:         0.661 ms
  P99 latency:         0.773 ms
  Min latency:         0.591 ms
  Max latency:         0.799 ms
  Throughput:         1513.8 QPS

======================================================================
SPEEDUP ANALYSIS
======================================================================
  Average speedup:      54.6x faster
  P99 speedup:          50.7x faster

  ✓ TARGET MET: 54.6x speedup (target: 20x)
======================================================================
```

## 3️⃣ End-to-End API Throughput
**Test:** 50 Concurrent Users, 2000 Requests, HTTP/1.1

```text
🚀 Starting load test: 50 concurrent users, 2000 total requests
Target: http://localhost:8000/search

==================================================
LOAD TEST RESULTS
==================================================
Total Requests:      2000
Successful:          2000
Failed:              0
Total Time:          40.94 s
Throughput (QPS):    48.85 req/s
--------------------------------------------------
Avg Latency:         1020.88 ms
P50 Latency:         1019.44 ms
P95 Latency:         1038.34 ms
P99 Latency:         1053.28 ms
==================================================
```

