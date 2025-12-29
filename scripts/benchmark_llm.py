import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def benchmark_llm():
    print("=" * 50)
    print("LLM Benchmark (LLaMA 3.2 via Ollama)")
    print("=" * 50)
    
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in one sentence.",
        "What is a neural network?"
    ]
    
    latencies = []
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt[:40]}...")
        
        start = time.time()
        resp = requests.post(OLLAMA_URL, json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        end = time.time()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "")
            tokens = data.get("eval_count", len(response_text.split()))
            duration = data.get("eval_duration", 0) / 1e9
            
            tps = tokens / duration if duration > 0 else 0
            
            print(f"  Latency: {latency_ms:.0f}ms")
            print(f"  Tokens: {tokens}")
            print(f"  Tokens/sec: {tps:.1f}")
        else:
            print(f"  Error: {resp.status_code}")
    
    avg_latency = sum(latencies) / len(latencies)
    
    print("\n" + "=" * 50)
    print("RESUME METRICS")
    print("=" * 50)
    print(f"Average response latency: {avg_latency:.0f}ms")
    print("=" * 50)

if __name__ == "__main__":
    benchmark_llm()
