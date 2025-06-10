
import pytest
from app.utils import (
    parse_resume, calculate_experience, calculate_degree_level, calculate_resume_score,
    compute_skill_match, compute_experience_match, compute_match_score, match_score_to_label,
    process_single_resume, get_AI_feedback
)
from app.schemas  import Resume, JobPosting
from app.exceptions import ValidationError, ProcessingError
from unittest.mock import patch

def test_parse_resume(sample_pdf_text):
    """Test resume parsing."""
    result = parse_resume(sample_pdf_text)
    assert result["name"] == "John Doe"
    assert result["email"] == "john.doe@example.com"
    assert result["phone"] == "123-456-7890"
    assert result["skills"] == ["Python", "SQL"]
    assert result["education"] == ["Bachelor of Science"]
    assert result["experience"] == 1  # Assuming one experience entry

def test_calculate_experience_valid():
    """Test valid experience calculation."""
    assert calculate_experience("2.5") == 30  # 2.5 years = 30 months

def test_calculate_experience_invalid():
    """Test invalid experience."""
    with pytest.raises(ValidationError, match="Invalid experience format"):
        calculate_experience("invalid")

def test_calculate_degree_level():
    """Test degree level calculation."""
    assert calculate_degree_level("PhD in Computer Science") == 4
    assert calculate_degree_level("Master of Science") == 3
    assert calculate_degree_level("Bachelor of Arts") == 2
    assert calculate_degree_level("Diploma") == 1
    assert calculate_degree_level("High School") == 0.5
    assert calculate_degree_level("Unknown") == 0

def test_calculate_resume_score_valid():
    """Test resume score calculation."""
    score = calculate_resume_score(
        experience=24, num_of_skills=5, degree_level=3, has_linkedin=True, number_of_jobs=2
    )
    assert score == min((24 * 2) + (5 * 1.5) + (3 * 3) + (1 * 2) + (2 * 1.5), 100)

def test_calculate_resume_score_missing_input():
    """Test resume score with missing input."""
    with pytest.raises(ValidationError, match="Missing required inputs"):
        calculate_resume_score(None, 5, 3, True, 2)

def test_compute_skill_match():
    """Test skill match computation."""
    resume_skills = ["Python", "SQL", "Java"]
    required_skills = ["Python", "SQL"]
    assert compute_skill_match(resume_skills, required_skills) == 1.0
    assert compute_skill_match(resume_skills, ["Python", "C++"]) == 0.5
    assert compute_skill_match(resume_skills, []) == 0.0

def test_compute_experience_match():
    """Test experience match computation."""
    assert compute_experience_match(5.0, 3.0) == min(5.0 / (3.0 * 1.5), 1.0)
    assert compute_experience_match(1.0, 5.0) == 0.2  # Minimum score

def test_compute_match_score():
    """Test match score computation."""
    score = compute_match_score(skill_match_score=0.8, experience_match_score=0.6)
    assert score == (0.8 * 0.6 + 0.6 * 0.4) * 100

def test_match_score_to_label():
    """Test match label assignment."""
    assert match_score_to_label(70) == "strong match"
    assert match_score_to_label(50) == "moderate match"
    assert match_score_to_label(20) == "weak match"

def test_process_single_resume_valid(valid_resume, valid_job_posting, mock_joblib, mock_deepseek_api):
    """Test processing a single resume."""
    result = process_single_resume(valid_resume, valid_job_posting)
    assert result.job_role == "Software Engineer"
    assert result.candidate_name == "John Doe"
    assert result.match_score > 0
    assert result.match_label in ["weak match", "moderate match", "strong match"]

def test_process_single_resume_missing_model(valid_resume, valid_job_posting, mocker):
    """Test missing model."""
    mocker.patch("joblib.load", side_effect=FileNotFoundError)
    with pytest.raises(ProcessingError, match="Model or scaler"):
        process_single_resume(valid_resume, valid_job_posting)

def test_get_AI_feedback_valid_resume(valid_resume, valid_job_posting, mock_deepseek_api):
    """Test AI feedback with resume."""
    result = get_AI_feedback([(valid_resume, valid_job_posting)])
    assert len(result["results"]) == 1
    assert result["llm_response"] == mock_deepseek_api
    assert result["errors"] == []

def test_get_AI_feedback_conversational_prompt(mock_deepseek_api):
    """Test AI feedback with prompt."""
    result = get_AI_feedback([], message_prompt="How to improve my resume?")
    assert result["results"] == []
    assert result["llm_response"] == mock_deepseek_api
    assert result["errors"] == []