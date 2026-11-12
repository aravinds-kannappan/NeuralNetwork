"""Standalone package runner: regenerate the three phases in a fresh directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from already_clean_reproduction.experiment import main as run
    from already_clean_reproduction.common import ROOT
    config_path, manifest_path = ROOT / "config.json", ROOT / "embedding_manifest.json"
    original_config, original_manifest = config_path.read_text(), manifest_path.read_text()
    config = json.loads(original_config)
    config["artifact_root"] = str(args.output.resolve())
    try:
        for phase in ("reproduction", "similarity", "followup"):
            config["phase"] = phase
            run_id = "standalone-" + phase
            if (args.output / "runs" / run_id).exists():
                raise SystemExit(f"Output already exists for {phase}; choose a fresh --output directory")
            os.environ["ORX_RUN_ID"] = run_id
            if phase == "followup":
                config["similarity_source_run"] = "standalone-similarity"
                source = args.output / "runs" / config["similarity_source_run"]
                files = [p for p in source.iterdir() if p.name.startswith(("embeddings-", "embedding-index-")) or p.name == "matched-controls.csv"]
                manifest_path.write_text(json.dumps({"run_id": config["similarity_source_run"], "files": [
                    {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in files]}, indent=2))
            config_path.write_text(json.dumps(config, indent=2))
            run()
    finally:
        config_path.write_text(original_config)
        manifest_path.write_text(original_manifest)


if __name__ == "__main__":
    main()
