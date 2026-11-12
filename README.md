# Borrowed geometry and robot recovery

A local CPU pilot linking questions about representation reuse to software-agent
fault tolerance. See PROTOCOL.md for registered scope, controls and limitations.

Run contract: `uv run --locked python -m borrowed_recovery.experiment`.
Use the project's experiment runner to execute a committed snapshot. Results are
written to outputs/ in the isolated run directory and emitted as JSON log records.

The controller is OGBench's packaged CubeMarkovOracle. It is scripted and uses
privileged simulator state. No Gemma, OLMo, BSF, MoE, or language-model agent is run.
