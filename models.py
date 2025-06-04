from ast import mod
import base64
from pydantic import BaseModel, field_validator, EmailStr
from typing import List, Optional, Dict
import re

class JobPosting(BaseModel):
    job_role: str
    department: str
    skills: List[str]
    experience: float
    location: str
    education: str
    description: str

    @field_validator('job_role', 'department', 'location', 'education', mode='after')
    def non_empty_string(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('skills', mode='after')
    def non_empty_skills(cls, v):
        if not v:
            raise ValueError("At least one skill is required")
        return [skill.strip() for skill in v if skill.strip()]

    @field_validator('experience')
    def non_negative_experience(cls, v):
        if v < 0:
            raise ValueError("Experience cannot be negative")
        return v

class Resume(BaseModel):
    name: str
    email: EmailStr
    phone: str
    skills: List[str]
    education: List[str]
    experience: float

    @field_validator('name', mode='after')
    def non_empty_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator('phone', mode='after')
    def valid_phone(cls, v):
        # Basic phone validation
        if not re.match(r"(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", v):
            raise ValueError("Invalid phone number format")
        return v

    @field_validator('experience')
    def non_negative_experience(cls, v):
        if v < 0:
            raise ValueError("Experience cannot be negative")
        return v

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

    @field_validator('file_base64', mode='after')
    def valid_base64(cls, v):
        try:
            base64.b64decode(v, validate=True)
        except base64.binascii.Error:
            raise ValueError("Invalid base64 string")
        return v

class PDFRequest(BaseModel):
    api_key: str
    resumes: List[ResumeInput] = []
    job_posting: Optional[JobPosting] = None
    message_prompt: Optional[str] = None

    @field_validator('api_key')
    def non_empty_api_key(cls, v):
        if not v.strip():
            raise ValueError("API key cannot be empty")
        return v