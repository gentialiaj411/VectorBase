import subprocess
import time
import sys
import os
def run_bench(cmd, name):
    print(f"⏳ Running {name}...")
    try:
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error running {name}:\n{result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Failed to execute {name}: {str(e)}"
def main():
    print("🚀 Starting Full System Verification on Local Machine...")
    report = "# 📊 VectorBase Performance Report\n"
    report += f"**Date:** {time.ctime()}\n"
    report += f"**System:** Windows (Local Run)\n"
    report += "-" * 50 + "\n\n"
    cmd_build = f"{sys.executable} scripts/generate_embeddings.py"
    out_build = run_bench(cmd_build, "Index Build (10k docs)")
    report += "## 1️⃣ Data Pipeline Performance\n"
    report += "**Test:** Indexing 10,000 documents (Build & Quantize)\n\n"
    report += "```text\n" + out_build.strip() + "\n```\n\n"
    cmd_simd = f"{sys.executable} scripts/benchmark_simd.py --vectors 100000 --queries 100"
    out_simd = run_bench(cmd_simd, "SIMD Kernel Speedup")
    report += "## 2️⃣ C++ Core Performance\n"
    report += "**Test:** 100,000 vectors, AVX2 vs NumPy (100 queries)\n\n"
    report += "```text\n" + out_simd.strip() + "\n```\n\n"
    cmd_load = f"{sys.executable} scripts/load_test.py --users 50 --requests 2000"
    out_load = run_bench(cmd_load, "API Load Test")
    report += "## 3️⃣ End-to-End API Throughput\n"
    report += "**Test:** 50 Concurrent Users, 2000 Requests, HTTP/1.1\n\n"
    report += "```text\n" + out_load.strip() + "\n```\n\n"
    with open("BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n✅ Verification Complete!")
    print("Results saved to: BENCHMARK_REPORT.md")
    print("-" * 50)
if __name__ == "__main__":
    main()
