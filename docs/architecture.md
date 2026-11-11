# Architecture

Status: **Phases 1–8 implemented**, plus a read-only dashboard.

## Layers

```text
cli.py                      argument parsing, wiring, exit codes
web/server.py               read-only dashboard (reads output files only)
   │
orchestration/              run lifecycle, stage runner, manifest updates
   │   pipeline.py          Pipeline.run(PipelineRequest) -> RunSummary
   │   context.py           RunContext, SourceVideo, run IDs
   │   stages/              Stage ABC + one adapter module per phase
   │
domain modules              single responsibilities, no orchestration imports
   │   ingestion/           downloader, validator
   │   metadata/            ffprobe, gpmf_extractor, imu_parser
   │   stereo/              stereo_splitter, frame_sampler
   │   inference/           base (provider interface), ollama_client, prompts, qwen_classifier
   │   analysis/            temporal_filter (intervals/runs), hoi_analyzer (timeline)
   │   segmentation/        waste_detector, video_cutter
   │   synchronization/     imu_aligner
   │   metrics/             hoi_metrics
   │   reporting/           manifest
   │
core/                       infrastructure: config, env, logging, workspace, tools, hashing, io, errors
```

### Dependency rule

`cli → orchestration → domain modules → core`. The dashboard depends only on `core`
and on the artifact files.

* Domain modules never import `orchestration` or read configuration. They take
  plain values and return plain values.
* `orchestration/stages/*` are the adapters. They read config keys, load inputs from
  workspace files, call domain functions and write outputs.
* `Pipeline` depends only on the `Stage` interface. The stage list comes from
  `stages.build_stages()`.
* Inference: stages depend on `InferenceProvider` (inference/base.py).
  `stages/inference.py::create_provider` is the only place that names a concrete
  provider (`OllamaProvider`).

## Stage chain

```text
ingestion → validation → metadata → telemetry → sampling → inference → temporal → verification → segmentation → imu_sync → metrics → qc
  (P1)        (P1)        (P1)        (P2)        (P3)       (P4)        (P5)         (P5b)          (P6)          (P7)       (P8)    (QC gate)
```
Shared stage helpers live in `orchestration/stages/pairs.py` (`PairExtractor`:
windowed, byte-identical re-extraction; `PairClassifier`: cached classification;
the provider factory). QC logic lives in `qc/` (`checks.py`: pure decision checks;
`video_checks.py`: FFmpeg probes, decode and PSNR). Fine-grained verification
logic is in `segmentation/verification.py`.

Each stage reads only files written by earlier stages, so any stage can run
after a resume. See [pipeline.md](pipeline.md).

## Departures from the plan's directory tree

| Added | Why |
|---|---|
| `core/` | Cross-cutting infrastructure kept out of domain packages. |
| `orchestration/stages/` (package) | One adapter module per phase instead of a single growing file. |
| `orchestration/context.py` | Run-scoped state passed to stages. |
| `web/` + `scripts/serve_ui.py` | Requested dashboard. |
| `configs/nvenc.yaml` | Optional GPU-encoding override. |
| `docs/decisions.md`, `docs/dashboard.md` | Decision log, UI docs. |

See [decisions.md](decisions.md).
