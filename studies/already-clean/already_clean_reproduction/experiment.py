"""Immutable-run entrypoint and portable artifact manifest."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, verify_inputs, write_json


def main():
    config = json.loads((ROOT / "config.json").read_text())
    run_id = os.environ.get("ORX_RUN_ID", "manual-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out = Path(config["artifact_root"]) / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"run_id": run_id, "experiment": os.environ.get("ORX_EXPERIMENT"),
                  "commit": os.environ.get("ORX_COMMIT"), "config": config,
                  "python": sys.version, "platform": platform.platform(), "inputs": verify_inputs(),
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-learn", "sentence-transformers", "transformers", "torch")}}
    print("CONFIG", json.dumps(provenance), flush=True)
    write_json(out / "provenance.json", provenance)
    if config["phase"] == "reproduction":
        from .reproduce import run
        result = run(out)
    elif config["phase"] == "similarity":
        from .similarity import run
        result = run(out, config)
    elif config["phase"] == "followup":
        from .followup import run
        result = run(out, config)
    else:
        raise ValueError(config["phase"])
    write_json(out / "results.json", result)
    print("FINAL_SUMMARY", json.dumps({"phase": config["phase"], "run_id": run_id,
                                       "result_file": str(out / "results.json"), "results": result}), flush=True)


if __name__ == "__main__":
    main()
