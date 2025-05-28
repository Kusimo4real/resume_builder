from pydantic import BaseModel
from typing import List, Optional, Dict


"""
 This module defines the data models used in the CV prediction API.
 It includes models for handling PDF requests, job postings,
 resumes, and match responses.
"""

class JobPosting(BaseModel):
    job_role: str
    department: str
    skills: List[str]
    experience: float
    location: str
    education: str
    description: str
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
    probabilities: Dict[str, float]
    features_used: Dict
    contact_email: str
    contact_phone: str
    llm_response: str

class ResumeInput(BaseModel):
    file_base64: str
    job_posting: Optional[JobPosting] = None

class PDFRequest(BaseModel):
    api_key: str
    resumes: List[ResumeInput]
    job_posting: Optional[JobPosting] = None
    message_prompt: Optional[str] = None