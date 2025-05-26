from pydantic import BaseModel
from typing import List


"""
 This module defines the data models used in the CV prediction API.
 It includes models for handling PDF requests, job postings,
 resumes, and match responses.
"""
class PDFRequest(BaseModel):
    api_key: str
    job_posting:object | str
    file_base64: str  # base64-encoded PDF

class JobPosting(BaseModel):
    api_key: str
    job_role: str
    department: str
    skills: List[str]
    experience: float
    location: str
    education: str
    description: str
    file_base64: str

class Resume(BaseModel):
    name: str
    email: str
    phone: str
    skills: List[str]
    education: List[str]
    experience: float

class MatchResponse(BaseModel):
    job_role: str
    candidate_name: str
    match_label: str
    match_score: float
    skill_match_score: float
    experience_match_score: float
    probabilities: dict
    features_used: dict
    contact_email: str
    contact_phone: str