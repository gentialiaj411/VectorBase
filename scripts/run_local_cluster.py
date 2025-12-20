import subprocess
import sys
import time
import os
import signal
from pathlib import Path

PYTHON_EXE = sys.executable
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data" / "sharded"

if not DATA_DIR.exists():
    print(f"Data directory {DATA_DIR} not found!")
    print("Please run: python scripts/split_data.py --shards 3")
    sys.exit(1)

processes = []

def start_process(cmd, env, name):
    print(f"Starting {name}...")
    full_env = os.environ.copy()
    full_env.update(env)
    current_path = full_env.get("PYTHONPATH", "")
    full_env["PYTHONPATH"] = f"{str(BASE_DIR)}{os.pathsep}{current_path}"
    
    log_file = open(f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.log", "w")
    p = subprocess.Popen(
        cmd, 
        env=full_env, 
        cwd=str(BASE_DIR),
        shell=True if os.name == 'nt' else False,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    processes.append(p)
    return p

def cleanup(signum, frame):
    print("\nShutting down cluster...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    print(f"Launching Local Cluster (Coordinator + 3 Workers)")
    print(f"Base Dir: {BASE_DIR}")
    print("---------------------------------------------------")

    workers = []
    for i in range(3):
        port = 8001 + i
        env = {
            "SHARD_ID": str(i),
            "DATA_DIR": str(DATA_DIR),
            "PORT": str(port)
        }
        cmd = [PYTHON_EXE, "-m", "uvicorn", "distributed.worker:app", "--host", "127.0.0.1", "--port", str(port)]
        workers.append(start_process(cmd, env, f"Worker {i} (Port {port})"))

    time.sleep(2)

    coord_env = {
        "WORKER_URLS": "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003"
    }
    cmd = [PYTHON_EXE, "-m", "uvicorn", "distributed.coordinator:app", "--host", "127.0.0.1", "--port", "8000"]
    start_process(cmd, coord_env, "Coordinator (Port 8000)")

    print("\nCluster is RUNNING!")
    print("Coordinator: http://127.0.0.1:8000")
    print("Test it with: python scripts/test_distributed.py")
    print("Press Ctrl+C to stop.")
    
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
