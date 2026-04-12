class ServiceError(Exception):
    """Base class for service-layer failures."""


class ConfigurationError(ServiceError):
    """Raised when required application settings are missing or invalid."""


class DocumentProcessingError(ServiceError):
    """Raised when an uploaded document cannot be parsed or indexed."""


class DocumentNotFoundError(ServiceError):
    """Raised when a document id does not exist in the registry."""


class NoActiveDocumentError(ServiceError):
    """Raised when a chat request does not specify a document and none is active."""


class UpstreamServiceError(ServiceError):
    """Raised when an external model or vector store call fails."""
