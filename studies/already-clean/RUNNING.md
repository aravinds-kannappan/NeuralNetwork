# Standalone reproduction

Run in a fresh extracted folder with Python 3.12 and uv:

```sh
uv run --locked python reproduce_all.py --output ./outputs
```

This runs the three analyses in order, downloads the two pinned sentence encoders, and writes new results. It does not regenerate the original OLMo activations or private judging. The script restores the shipped config and input manifest after the run. GPU is not required. Original source and attribution are preserved under `already_clean_reproduction/inputs/`.
