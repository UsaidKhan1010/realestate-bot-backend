#!/bin/bash
set -x

echo "=== START: Debug start.sh ==="
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "Python executable:"
which python || true
python --version || true

echo "Check if pip exists:"
python -m pip --version || true

echo "SHOW pip list before install (if any):"
python -m pip list || true

echo "Cat requirements.txt:"
if [ -f requirements.txt ]; then
  sed -n '1,200p' requirements.txt
else
  echo "No requirements.txt file found!"
fi

echo "Attempting to ensure pip is bootstrapped..."
python -m ensurepip --upgrade || true

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel || true

echo "Now installing requirements (verbose):"
python -m pip install -r requirements.txt --no-cache-dir -v

echo "SHOW pip list AFTER install:"
python -m pip list || true

echo "Check uvicorn module import directly:"
python - <<'PY'
try:
    import uvicorn
    print("uvicorn import OK, version:", getattr(uvicorn,'__version__', 'unknown'))
except Exception as e:
    import traceback
    print("uvicorn import FAILED:")
    traceback.print_exc()
PY

echo "If uvicorn import OK, starting app..."
python -m uvicorn app:app --host 0.0.0.0 --port $PORT
echo "=== END: Debug start.sh ==="
