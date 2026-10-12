"""Run the real API with isolated local sessions for browser validation."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
runtime = repo / "tmp" / "browser-validation"
runtime.mkdir(parents=True, exist_ok=True)
# Set before importing storage/router modules. Use file storage and leave raw
# retention off, so benchmark uploads do not become training examples.
tempfile.tempdir = str(runtime)
os.environ["DATABASE_URL"] = ""
os.environ["TRAINING_RETENTION_ENABLED"] = "0"
os.environ["TRAINING_DATA_DIR"] = str(runtime / "training")
os.environ["INCIDENTS_DATA_DIR"] = str(runtime / "incidents")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8901)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("webapp.api.main:app", host="127.0.0.1", port=args.port)
