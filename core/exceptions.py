"""Project-specific exception hierarchy.

Using dedicated exception types keeps error handling in the web layer clean:
the API can map ``ModelNotFoundError`` to an HTTP 503, validation problems to
400, and so on, instead of catching bare ``Exception``.
"""
from __future__ import annotations


class TrafficAIError(Exception):
    """Base class for all errors raised by this project."""


class ModelNotFoundError(TrafficAIError):
    """A required model file (YOLO/LSTM/XGBoost) is missing on disk."""


class ModelLoadError(TrafficAIError):
    """A model file exists but could not be loaded/deserialised."""


class StreamError(TrafficAIError):
    """A video/webcam/RTSP source could not be opened or read."""


class DatabaseError(TrafficAIError):
    """A database operation failed."""


class ValidationError(TrafficAIError):
    """User-supplied input failed validation."""
