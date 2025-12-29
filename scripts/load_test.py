import requests
import time
import argparse
import concurrent.futures
import numpy as np
import sys
def run_query(url, query):
    try:
        start = time.time()
        resp = requests.post(url, json={"query": query, "k": 10}, timeout=10)
        resp.raise_for_status()
        latency = (time.time() - start) * 1000
        return latency
    except Exception as e:
        return None
def main():
    parser = argparse.ArgumentParser(description="Load test VectorBase API")
    parser.add_argument("--url", default="http://localhost:8000/search", help="API URL")
    parser.add_argument("--users", type=int, default=50, help="Concurrent users")
    parser.add_argument("--requests", type=int, default=1000, help="Total requests")
    args = parser.parse_args()
    print(f"🚀 Starting load test: {args.users} concurrent users, {args.requests} total requests")
    print(f"Target: {args.url}")
    queries = [
        "machine learning", "quantum computing", "neural networks", "transformer architecture",
        "distributed systems", "vector databases", "approximate nearest neighbor", "semantic search",
        "binary quantization", "hamming distance"
    ]
    start_time = time.time()
    latencies = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as executor:
        futures = []
        for i in range(args.requests):
            query = queries[i % len(queries)]
            futures.append(executor.submit(run_query, args.url, query))
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                latencies.append(res)
            else:
                errors += 1
    total_time = time.time() - start_time
    qps = len(latencies) / total_time
    latencies = np.array(latencies)
    print("\n" + "="*50)
    print("LOAD TEST RESULTS")
    print("="*50)
    print(f"Total Requests:      {args.requests}")
    print(f"Successful:          {len(latencies)}")
    print(f"Failed:              {errors}")
    print(f"Total Time:          {total_time:.2f} s")
    print(f"Throughput (QPS):    {qps:.2f} req/s")
    print("-" * 50)
    print(f"Avg Latency:         {np.mean(latencies):.2f} ms")
    print(f"P50 Latency:         {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 Latency:         {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency:         {np.percentile(latencies, 99):.2f} ms")
    print("="*50)
if __name__ == "__main__":
    main()
