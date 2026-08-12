# Agent Catalog — Agent_IA_Server

Specialized agents and sub-agents that remediate the **SECURITY** and **AUDIO**
pipeline audits performed on 2026-08-12. Each agent is a self-contained
runbook: scope, files, approach, sub-agents, dependencies, acceptance.

## Layout

```
agents/
├── README.md                         # this file
├── sec-orchestrator.md               # security domain coordinator
├── audio-orchestrator.md             # audio pipeline coordinator
├── sec/                              # 14 sub-agents (one per finding)
│   ├── sec-001-bootstrap-credential.md
│   ├── sec-002-tenant-authorization.md
│   ├── sec-003-twilio-voice-signature.md
│   ├── sec-004-media-stream-binding.md
│   ├── sec-005-tool-ssrf.md
│   ├── sec-006-credential-storage.md
│   ├── sec-007-password-hashing.md
│   ├── sec-008-tool-policy.md
│   ├── sec-009-log-redaction.md
│   ├── sec-010-credential-reveal.md
│   ├── sec-011-call-status-signature.md
│   ├── sec-012-error-sanitization.md
│   ├── sec-013-dependency-sbom.md
│   └── sec-014-encryption-key.md
├── audio/                            # 12 sub-agents (one per finding)
│   ├── audio-001-queue-backpressure.md
│   ├── audio-002-echo-mute-buffer.md
│   ├── audio-003-bargein-correctness.md
│   ├── audio-004-tail-truncation.md
│   ├── audio-005-unbounded-speech.md
│   ├── audio-006-format-validation.md
│   ├── audio-007-resampling-quality.md
│   ├── audio-008-jitter-pacing.md
│   ├── audio-009-inworld-batching.md
│   ├── audio-010-vad-calibration.md
│   ├── audio-011-codec-regression.md
│   └── audio-012-observability.md
└── tools/                            # helper scripts invoked by agents
    ├── audio_metrics_collector.py
    ├── golden_audio_check.py
    └── queue_drop_probe.py
```

## Naming convention

- `sec-NNN-<slug>.md` — addresses one security finding.
- `audio-NNN-<slug>.md` — addresses one audio finding.
- `<domain>-orchestrator.md` — coordinates sub-agents within a domain.

## Status legend (carried from audits)

- `SAFE_CHANGE` — code/test modification is the primary deliverable.
- `NEEDS_TESTS_FIRST` — test suite / harness is the first deliverable; fix follows.
- `NEEDS_DESIGN_FIRST` — design + threat model must precede code changes.
- `NEEDS_MEASUREMENT_FIRST` — instrumentation + data must precede any tuning.

## Execution order (P0 → P2)

1. **P0** — sec-001, sec-002, sec-003, sec-004, sec-005, audio-001, audio-002, audio-003
2. **P1** — sec-006, sec-007, sec-008, audio-004, audio-005, audio-006, audio-007, audio-011
3. **P1/P2** — sec-013, audio-008, audio-009, audio-010
4. **P2** — sec-009, sec-010, sec-011, sec-012, sec-014, audio-012

The orchestrators enforce this order; sub-agents refuse to run if a blocking
dependency is not satisfied.

## Invocation

Each agent file is plain Markdown — readable, diffable, and executable by an
LLM-driven runner. To run one manually:

1. Read the agent file.
2. Verify its dependencies are met (acceptance criteria of upstream agents).
3. Apply the listed edits / tests.
4. Run the listed verification.
5. Mark the finding as resolved in the orchestrator's state file
   (`.agents-state.json`, optional).

## Verification baseline

```bash
# from Agent_IA_Server/
python -m pytest tests/ -q
python -c "from STT_server.services.audio_codec import ulaw2lin, lin2ulaw"
python agents/tools/queue_drop_probe.py --help
```

A sub-agent is "done" only when its acceptance block passes. Anything not
measurable today is marked **UNKNOWN — fixtures required** and parked, not
shipped.
