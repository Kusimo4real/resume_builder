import pytest
import base64
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import JobPosting, Resume, PDFRequest, ResumeInput
from app.utils import call_deepseek_api, process_single_resume
from unittest.mock import patch, MagicMock
import pandas as pd

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def valid_job_posting():
    """Valid job posting fixture."""
    return JobPosting(
        job_role="Software Engineer",
        department="Engineering",
        skills=["Python", "SQL"],
        experience=3.0,
        location="Remote",
        education="Bachelor",
        description="Develop software solutions."
    )

@pytest.fixture
def valid_resume_data():
    """Valid resume data fixture."""
    return {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "123-456-7890",
        "skills": ["Python", "SQL"],
        "education": ["Bachelor of Science"],
        "experience": 3.0
    }

@pytest.fixture
def valid_resume(valid_resume_data):
    """Valid resume fixture."""
    return Resume(**valid_resume_data)

@pytest.fixture
def sample_pdf_text():
    """Sample PDF text for mocking."""
    return """
    John Doe
    Email: john.doe@example.com
    Phone: 123-456-7890
    Skills: Python, SQL
    Education: Bachelor of Science
    Experience: Software Engineer, 3 years
    """

@pytest.fixture
def valid_base64_pdf(sample_pdf_text):
    """Valid base64-encoded PDF fixture."""
    return base64.b64encode(sample_pdf_text.encode()).decode()

@pytest.fixture
def mock_pdf_reader(mocker, sample_pdf_text):
    """Mock PyPDF2 PdfReader."""
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = sample_pdf_text
    mock_reader = mocker.MagicMock()
    mock_reader.pages = [mock_page]
    mocker.patch("pypdf.PdfReader", return_value=mock_reader)
    return mock_reader

@pytest.fixture
def mock_joblib(mocker):
    """Mock joblib for model/scaler loading."""
    mock_model = mocker.MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.6, 0.3]]  # weak, moderate, strong
    mock_model.classes_ = ["weak", "moderate", "strong"]
    mock_scaler = mocker.MagicMock()
    mock_scaler.transform.return_value = pd.DataFrame({
        'total_experience': [36],
        'degree_level': [2],
        'has_linkedin_profile': [1],
        'num_of_jobs': [2]
    })
    mocker.patch("joblib.load", side_effect=lambda path: mock_model if "model" in path else mock_scaler)
    return mock_model, mock_scaler

@pytest.fixture
def mock_deepseek_api(mocker):
    """Mock DeepSeek API call."""
    mock_response = "Mocked DeepSeek response"
    mocker.patch("app.utils.call_deepseek_api", return_value=mock_response)
    return mock_response