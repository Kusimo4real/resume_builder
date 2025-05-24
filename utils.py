
from typing import List
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


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

"""
    _summary_
This module contains utility functions for data processing and model evaluation.
    _description_
    - `extract_experience`: Extracts years of experience from a given text.
    - `extract_skills`: Extracts and counts unique skills from a given text.
    - `has_linkedin`: Checks if a LinkedIn profile link is present in the text.
    - `extract_degree_level`: Extracts the degree level from the education section of the text.
    - `process_pdf`: Processes a PDF file to extract structured information including total experience, skills, LinkedIn presence, and degree level.
    - `extract_text_from_pdf`: Extracts text from a PDF file using PyPDF2.    
"""


def calculate_experience(text: str) -> int:
    """
    Extracts years of experience from a given text.
    returns experience in months as an integer.
    """
    try:
        return int(text * 12)
    except ValueError:
        raise ValueError("Invalid experience format. Please provide a valid number.")
    except Exception as e:
        raise Exception(f"Error calculating experience: {str(e)}")
    


def calculate_degree_level(text: str) -> int:
    '''
    calculates a score based on degree level
    '''
    try:
        program = str(text).lower()
        if "phd" or "doctor" in program:
            return 4
        elif "master" in program:
            return 3
        elif "bachelor" in program:
            return 2
        elif "diploma" in program:
            return 1
        elif "high school" in program:
            return 0.5
        else:
            return 0
    except Exception as e:
        raise Exception(f"Error calculating degree level: {str(e)}")
    
    
def calculate_resume_score(experience: int, num_of_skills: int, degree_level: int, has_linkedin: bool, number_of_jobs: int) -> float:
    """
    Calculate a resume score based on experience, skills, degree level, and LinkedIn presence.
    """
    try:
        score = (experience * 2) + (num_of_skills * 1.5) + (degree_level * 3) + (int(has_linkedin) * 2) + (number_of_jobs * 1.5)
        return min(score, 100)  # Cap the score at 100
    except ValueError:
        raise ValueError("Invalid input for resume score calculation. Please check the values provided.")
    except TypeError:
        raise TypeError("Invalid type for resume score calculation. Please ensure all inputs are of the correct type.")
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero error in resume score calculation. Please check the inputs.")
    except OverflowError:
        raise OverflowError("Overflow error in resume score calculation. Please check the inputs.")
    except Exception as e:
        raise Exception(f"Error calculating resume score: {str(e)}")
    
    
    
def compute_skill_match(resume_skills: List[str], required_skills: List[str]) -> float:
    """Compute skill match score as proportion of required skills present in resume."""
    required_skills = set(skill.lower().strip() for skill in required_skills)
    resume_skills = set(skill.lower().strip() for skill in resume_skills)
    total_required = len(required_skills)
    skill_matches = sum(1 for skill in required_skills if skill in resume_skills)
    return skill_matches / total_required if total_required > 0 else 0

def compute_experience_match(resume_experience_years: float, required_years: float) -> float:
    """Compute experience match score with linear scaling (notebook logic)."""
    total_years = resume_experience_years
    max_years = required_years * 1.5  # From Cell 9
    score = min(total_years / max_years, 1.0)
    return max(score, 0.2)  # Minimum score from Cell 9

def compute_match_score(skill_match_score: float, experience_match_score: float) -> float:
    """Compute weighted match score (notebook logic)."""
    skill_weight = 0.6
    exp_weight = 0.4
    return (skill_match_score * skill_weight + experience_match_score * exp_weight) * 100

def match_score_to_label(score: float) -> str:
    """Assign match label based on score (notebook thresholds)."""
    if score >= 65:
        return 'strong match'
    elif score >= 35:
        return 'moderate match'
    else:
        return 'weak match'
    

def match_resume(job_posting: JobPosting, resume: Resume):
    """Match a resume against a job posting and return match strength and probabilities."""
    # Validate api_key (basic check)
    # if not job_posting.api_key.strip():
    #     raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Extract relevant job posting fields
    print(job_posting)
    job_role = job_posting["job_role"].strip()
    required_skills = job_posting["skills"]
    required_years = job_posting["experience"]
    required_degree = job_posting["education"].lower().strip()

    # Load model and scaler
    model_path = Path(f"models/resume_match_model_{job_role.replace(' ', '_')}.pkl")
    scaler_path = Path(f"models/scaler_{job_role.replace(' ', '_')}.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model or scaler for job role '{job_role}' not found")

    # Compute skill_match_score
    skill_match_score = compute_skill_match(resume['skills'], required_skills)

    # Compute experience_match_score (convert resume experience to years)
    resume_experience_years = resume['experience']  # Assuming this is already in years
    experience_match_score = compute_experience_match(resume_experience_years, required_years)

    # Compute match_score and match_label
    match_score = compute_match_score(skill_match_score, experience_match_score)
    match_label = match_score_to_label(match_score)

    # Prepare features for model prediction
    # Notebook features: total_experience (months), degree_level, has_linkedin_profile, num_of_jobs
    degree_mapping = {
        'high school': 1,
        'bachelor': 2,
        'master': 3,
        'phd': 4,
        'doctorate': 4
    }
    resume_education = resume['education'][0].lower() if resume['education'] else ''
    degree_level = next((degree_mapping[deg] for deg in degree_mapping if deg in resume_education), 2)  # Default to Bachelor's

    # Assumptions for missing features
    total_experience_months = resume['experience'] * 12  # Convert years to months
    has_linkedin_profile = 1  # Assume yes (can be updated if resume provides this)
    num_of_jobs = 2  # Default (can be updated if resume provides job history)

    features = pd.DataFrame({
        'total_experience': [total_experience_months],
        'degree_level': [degree_level],
        'has_linkedin_profile': [has_linkedin_profile],
        'num_of_jobs': [num_of_jobs]
    })

    # Clip features (from Cell 11)
    features['total_experience'] = features['total_experience'].clip(upper=120)
    features['num_of_jobs'] = features['num_of_jobs'].clip(upper=6)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probabilities
    probabilities = model.predict_proba(features_scaled)[0]
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)}

    # Prepare response
    response = MatchResponse(
        job_role=job_role,
        candidate_name=resume['name'],
        match_label=match_label,
        match_score=round(match_score, 2),
        skill_match_score=round(skill_match_score, 4),
        experience_match_score=round(experience_match_score, 4),
        probabilities=prob_dict,
        features_used=features.to_dict(orient='records')[0],
        contact_email=resume["email"],
        contact_phone=resume["phone"]
    )

    return response
