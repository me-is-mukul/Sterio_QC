"""Exception hierarchy for the pipeline.

Every error raised deliberately by pipeline code derives from PipelineError so
the CLI can report it cleanly without a traceback.
"""


class PipelineError(Exception):
    """Base class for all expected pipeline failures."""


class ConfigError(PipelineError):
    """Configuration is missing, malformed or contains invalid values."""


class ToolNotFoundError(PipelineError):
    """A required external executable (ffmpeg, ffprobe, ...) is unavailable."""


class ToolExecutionError(PipelineError):
    """An external executable failed or timed out."""


class IngestionError(PipelineError):
    """The source video could not be acquired."""


class DownloadError(IngestionError):
    """Downloading a video from a URL failed."""


class FileConflictError(IngestionError):
    """A target file already exists and overwriting was not confirmed."""


class InputValidationError(PipelineError):
    """The source video failed validation."""


class MetadataExtractionError(PipelineError):
    """Metadata could not be extracted from the source video."""


class WorkspaceError(PipelineError):
    """The output workspace is inconsistent or cannot be created."""


class ResumeError(PipelineError):
    """A run cannot be resumed (missing manifest, different source, ...)."""


class SourceModifiedError(PipelineError):
    """The immutable source video changed while the pipeline was running."""


class TelemetryError(PipelineError):
    """Telemetry could not be extracted or parsed."""


class SamplingError(PipelineError):
    """Stereo frame sampling failed."""


class InferenceError(PipelineError):
    """The inference provider is unreachable, misconfigured or returned an unusable result."""


class SegmentationError(PipelineError):
    """Waste detection or video cutting failed."""


class SynchronizationError(PipelineError):
    """Telemetry could not be aligned with the processed video."""


class QualityCheckError(PipelineError):
    """One or more quality checks failed."""


class StageInputError(PipelineError):
    """A stage's required input artifact (from an earlier stage) is missing or invalid."""
