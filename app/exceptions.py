from fastapi import HTTPException
from typing import Any, Dict, Optional

class APIError(Exception):
    """Base class for API exceptions."""
    def __init__(
        self,
        detail: str,
        code: str,
        status_code: int = 400,
        context: Optional[Dict[str, Any]] = None
    ):
        self.detail = detail
        self.code = code
        self.status_code = status_code
        self.context = context or {}
        super().__init__(self.detail)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detail": self.detail,
            "code": self.code,
            "context": self.context
        }

class ValidationError(APIError):
    """Raised for input validation failures."""
    def __init__(self, detail: str, code: str = "VALIDATION_ERROR", context: Optional[Dict] = None):
        super().__init__(detail, code, status_code=400, context=context)

class AuthenticationError(APIError):
    """Raised for authentication failures."""
    def __init__(self, detail: str = "Invalid API key", code: str = "INVALID_API_KEY", context: Optional[Dict] = None):
        super().__init__(detail, code, status_code=403, context=context)

class ProcessingError(APIError):
    """Raised for processing failures (e.g., PDF parsing, model errors)."""
    def __init__(self, detail: str, code: str, status_code: int = 400, context: Optional[Dict] = None):
        super().__init__(detail, code, status_code=status_code, context=context)

class ExternalServiceError(APIError):
    """Raised for external service failures (e.g., DeepSeek API)."""
    def __init__(self, detail: str, code: str = "EXTERNAL_SERVICE_ERROR", context: Optional[Dict] = None):
        super().__init__(detail, code, status_code=503, context=context)

class InternalServerError(APIError):
    """Raised for unexpected server errors."""
    def __init__(self, detail: str = "Internal server error", code: str = "INTERNAL_SERVER_ERROR", context: Optional[Dict] = None):
        super().__init__(detail, code, status_code=500, context=context)