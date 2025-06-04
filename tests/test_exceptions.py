
import pytest
from app.exceptions import APIError, ValidationError, AuthenticationError, ProcessingError, ExternalServiceError, InternalServerError

def test_api_error():
    """Test base APIError."""
    error = APIError(detail="Test error", code="TEST_ERROR", status_code=400, context={"key": "value"})
    assert error.detail == "Test error"
    assert error.code == "TEST_ERROR"
    assert error.status_code == 400
    assert error.context == {"key": "value"}
    assert error.to_dict() == {
        "detail": "Test error",
        "code": "TEST_ERROR",
        "context": {"key": "value"}
    }

def test_validation_error():
    """Test ValidationError."""
    error = ValidationError(detail="Invalid input", context={"field": "name"})
    assert error.status_code == 400
    assert error.code == "VALIDATION_ERROR"

def test_authentication_error():
    """Test AuthenticationError."""
    error = AuthenticationError()
    assert error.status_code == 403
    assert error.code == "INVALID_API_KEY"

def test_processing_error():
    """Test ProcessingError."""
    error = ProcessingError(detail="Processing failed", code="PROCESSING_ERROR", status_code=404)
    assert error.status_code == 404
    assert error.code == "PROCESSING_ERROR"

def test_external_service_error():
    """Test ExternalServiceError."""
    error = ExternalServiceError(detail="Service down")
    assert error.status_code == 503
    assert error.code == "EXTERNAL_SERVICE_ERROR"

def test_internal_server_error():
    """Test InternalServerError."""
    error = InternalServerError()
    assert error.status_code == 500
    assert error.code == "INTERNAL_SERVER_ERROR"
