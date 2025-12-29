@echo off
setlocal
set PATH=C:\msys64\mingw64\bin;%PATH%
echo 🚀 Starting VectorBase in PRODUCTION mode (4 Workers)...
echo ⚡ utilizing all CPU cores for maximum throughput.

:: Run uvicorn with 4 worker processes
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
