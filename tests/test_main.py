import pytest
from fastapi.testclient import TestClient
from app.models import PDFRequest, ResumeInput
import base64
from app.exceptions import AuthenticationError, ValidationError, ProcessingError

pytestmark = pytest.mark.asyncio

async def test_predict_cv_valid_single_resume(client, valid_job_posting, valid_base64_pdf, mock_pdf_reader, mock_joblib, mock_deepseek_api):
    """Test valid single resume processing."""
    request = PDFRequest(
        api_key="1234",
        resumes=[ResumeInput(file_base64=valid_base64_pdf, job_posting=valid_job_posting)],
        job_posting=valid_job_posting
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["job_role"] == "Software Engineer"
    assert data["results"][0]["candidate_name"] == "John Doe"
    assert "llm_response" in data
    assert data.get("errors", []) == []

async def test_predict_cv_invalid_api_key(client, valid_job_posting, valid_base64_pdf):
    """Test invalid API key."""
    request = PDFRequest(
        api_key="wrong",
        resumes=[ResumeInput(file_base64=valid_base64_pdf, job_posting=valid_job_posting)]
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 403
    data = response.json()
    assert data["code"] == "INVALID_API_KEY"
    assert data["detail"] == "Invalid API key"

async def test_predict_cv_missing_resume_and_prompt(client):
    """Test missing resumes and prompt."""
    request = PDFRequest(api_key="1234", resumes=[])
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "VALIDATION_ERROR"
    assert data["detail"] == "At least one resume or a message prompt is required"

async def test_predict_cv_invalid_base64(client, valid_job_posting):
    """Test invalid base64 data."""
    request = PDFRequest(
        api_key="1234",
        resumes=[ResumeInput(file_base64="AB$CD==InvalidBase64String$$", job_posting=valid_job_posting)]
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "ALL_RESUMES_FAILED"
    assert any("INVALID_BASE64" in err["code"] for err in data["context"].get("errors", []))

async def test_predict_cv_partial_processing(client, valid_job_posting, valid_base64_pdf, mock_pdf_reader, mock_joblib, mock_deepseek_api):
    """Test partial processing with one invalid resume."""
    request = PDFRequest(
        api_key="1234",
        resumes=[
            ResumeInput(file_base64=valid_base64_pdf, job_posting=valid_job_posting),
            ResumeInput(file_base64="invalid_base64", job_posting=valid_job_posting)
        ]
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["code"] == "INVALID_BASE64"
    assert data["context"]["processed"] == 1
    assert data["context"]["failed"] == 1

async def test_predict_cv_missing_job_posting(client, valid_base64_pdf):
    """Test missing job posting."""
    request = PDFRequest(
        api_key="1234",
        resumes=[ResumeInput(file_base64=valid_base64_pdf)]
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "VALIDATION_ERROR"
    assert "job posting" in data["detail"].lower()

async def test_predict_cv_conversational_prompt(client, mock_deepseek_api):
    """Test conversational prompt without resumes."""
    request = PDFRequest(
        api_key="1234",
        message_prompt="How to improve my resume?"
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["llm_response"] == mock_deepseek_api
    assert data["results"] == []
    assert data["errors"] == []

async def test_predict_cv_empty_pdf(client, valid_job_posting, mocker):
    """Test empty PDF."""
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader = mocker.MagicMock()
    mock_reader.pages = [mock_page]
    mocker.patch("pypdf.PdfReader", return_value=mock_reader)
    request = PDFRequest(
        api_key="1234",
        resumes=[ResumeInput(file_base64=base64.b64encode(b"").decode(), job_posting=valid_job_posting)]
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "ALL_RESUMES_FAILED"
    assert any("EMPTY_PDF" in err["code"] for err in data["context"].get("errors", []))

async def test_predict_cv_missing_model(client, valid_base64_pdf, valid_job_posting, mocker, mock_pdf_reader):
    """Test missing model file."""
    mocker.patch("joblib.load", side_effect=FileNotFoundError("Model not found"))
    request = PDFRequest(
        api_key="1234",
        resumes=[ResumeInput(file_base64=valid_base64_pdf, job_posting=valid_job_posting)]
    )
    response = client.post("/cv/predict", json=request.model_dump())
    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "MODEL_NOT_FOUND"
