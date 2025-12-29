import requests
import time
import numpy as np
import concurrent.futures

BASE_URL = "http://localhost:8000"
ENDPOINT = "/search"

def single_request(query):
    try:
        start = time.perf_counter()
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json={"query": query, "k": 10}, timeout=5)
        end = time.perf_counter()
        if response.status_code == 200:
            return (end - start) * 1000, response.json().get("took_ms", 0)
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def run_benchmark(num_requests=100, workers=5):
    print(f"Starting real-world benchmark: {num_requests} requests with {workers} workers...")
    
    queries = [
        "attention mechanisms in transformers",
        "convolutional neural networks for image classification",
        "reinforcement learning algorithms",
        "object detection with YOLO",
        "natural language processing with BERT"
    ] * (num_requests // 5)

    e2e_latencies = []
    server_took_times = []

    start_bench = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(single_request, queries))
    end_bench = time.perf_counter()

    for res in results:
        if res:
            e2e_latencies.append(res[0])
            server_took_times.append(res[1])

    if not e2e_latencies:
        print("No successful requests.")
        return

    total_time = end_bench - start_bench
    qps = len(e2e_latencies) / total_time

    print("\n" + "="*40)
    print("REAL-WORLD BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Requests: {len(e2e_latencies)}")
    print(f"Throughput (QPS): {qps:.2f}")
    print(f"Avg E2E Latency: {np.mean(e2e_latencies):.2f} ms")
    print(f"P50 E2E Latency: {np.percentile(e2e_latencies, 50):.2f} ms")
    print(f"P99 E2E Latency: {np.percentile(e2e_latencies, 99):.2f} ms")
    print(f"Avg Server-side 'took_ms': {np.mean(server_took_times):.2f} ms")
    print("="*40)

if __name__ == "__main__":
    # Wait for server to be ready
    print("Checking if server is up...")
    for _ in range(10):
        try:
            requests.get(f"{BASE_URL}/docs", timeout=1)
            print("Server is up!")
            break
        except:
            time.sleep(1)
    else:
        print("Server not found. Please start api/server.py first.")
        exit(1)

    run_benchmark(num_requests=100, workers=1) # Sequential first
    run_benchmark(num_requests=100, workers=10) # Concurrent
