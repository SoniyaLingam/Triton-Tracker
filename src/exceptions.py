"""Project-specific exceptions for clear Tracker failure reporting."""


class TrackerError(Exception):
    """Base class for errors raised by Tracker components."""


class ConfigurationError(TrackerError):
    """Raised when pipeline configuration cannot be validated."""


class DataProcessingError(TrackerError):
    """Raised when CSV content cannot be processed safely."""


class ResourceError(TrackerError, OSError):
    """Raised when Tracker cannot access or clean up a file resource."""


class PipelineError(TrackerError, TypeError):
    """Raised when a pipeline cannot be constructed or execute a step."""


__all__ = [
    "ConfigurationError",
    "DataProcessingError",
    "PipelineError",
    "ResourceError",
    "TrackerError",
]
