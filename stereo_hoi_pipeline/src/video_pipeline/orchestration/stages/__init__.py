"""Pipeline stages, in execution order."""

from video_pipeline.orchestration.stages.base import Stage, StageOutcome
from video_pipeline.orchestration.stages.foundation import IngestionStage, InputValidationStage, MetadataExtractionStage
from video_pipeline.orchestration.stages.inference import InferenceStage
from video_pipeline.orchestration.stages.qc import QualityCheckStage
from video_pipeline.orchestration.stages.reporting import MetricsStage
from video_pipeline.orchestration.stages.sampling import SamplingStage
from video_pipeline.orchestration.stages.segmentation import SegmentationStage
from video_pipeline.orchestration.stages.synchronization import ImuSyncStage
from video_pipeline.orchestration.stages.telemetry import TelemetryStage
from video_pipeline.orchestration.stages.temporal import TemporalStage
from video_pipeline.orchestration.stages.verification import VerificationStage


def build_stages() -> list[Stage]:
    return [
        IngestionStage(),           # Phase 1
        InputValidationStage(),     # Phase 1
        MetadataExtractionStage(),  # Phase 1
        TelemetryStage(),           # Phase 2
        SamplingStage(),            # Phase 3
        InferenceStage(),           # Phase 4
        TemporalStage(),            # Phase 5: candidate waste sections
        VerificationStage(),        # Phase 5: fine re-check -> final removal decision
        SegmentationStage(),        # Phase 6
        ImuSyncStage(),             # Phase 7
        MetricsStage(),             # Phase 8
        QualityCheckStage(),        # QC gate (fails the run on failure)
    ]


STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in build_stages())

__all__ = ["Stage", "StageOutcome", "build_stages", "STAGE_NAMES"]
