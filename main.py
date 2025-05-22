from math import exp
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader
import base64
import io
import re
import pandas as pd
import numpy as np
import os
import joblib

from utils import calculate_degree_level, calculate_experience, calculate_resume_score

app = FastAPI()

VALID_API_KEY = "1234"

class PDFRequest(BaseModel):
    api_key: str
    file_base64: str  # base64-encoded PDF

@app.get("/")
def read_root():
    return {"message": "CV prediction API is running."}

@app.post("/cv/predict")
async def predict_cv(request: PDFRequest):
    # Validate API Key
    if request.api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        # Decode and read PDF
        decoded_pdf = base64.b64decode(request.file_base64)
        pdf_file = io.BytesIO(decoded_pdf)
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Extract experience
        def extract_experience(text: str) -> int:
            match = re.search(r"(\d+)\+?\s+years? (?:of )?experience", text, re.I)
            return int(match.group(1)) if match else 0

        # Extract number of skills
        def extract_skills(text: str):
            skill_sections = re.findall(r"(?:Technical Skills|Soft Skills|Other Skills):(.+?)(?:\n[A-Z][a-z]+:|$)", text, re.I | re.S)
            skills = set()
            for section in skill_sections:
                items = re.split(r"[,\n]", section)
                for item in items:
                    skill = item.strip()
                    if skill:
                        skills.add(skill)
            return sorted(skills)

        """
        def extract_skills(text: str) -> int:
            skill_sections = re.findall(r"(?:Technical Skills|Soft Skills|Other Skills):(.+?)(?:\n[A-Z][a-z]+:|$)", text, re.I | re.S)
            skills = set()
            for section in skill_sections:
                items = re.split(r"[,\\n]", section)
                for item in items:
                    skill = item.strip()
                    if skill:
                        skills.add(skill)
            return len(skills)
            """

        # Check for LinkedIn
        def has_linkedin(text: str) -> bool:
            return bool(re.search(r"https?://(www\.)?linkedin\.com", text, re.I))

        # Extract education
        def extract_degree_level(text: str) -> str:
            match = re.search(r"EDUCATION\s*(.+?)(?:\n[A-Z ]{3,}\n|\n\s*\n|$)", text, re.S | re.I)
            return match.group(1).strip() if match else ""
        
        degree = extract_degree_level(text)
        
        degree_score = calculate_degree_level(degree)

        experience = extract_experience(text)
        experience_score = calculate_experience(experience)
        
        skills = extract_skills(text)
        number_of_jobs = extract_number_of_jobs(None)
        
        # Calculate resume score
        resume_score = calculate_resume_score(experience_score, len(skills), degree_score, int(has_linkedin(text)), number_of_jobs)
        
        # build dataframe as model input
        df = pd.DataFrame([
            {
                "total_experience": float(calculate_experience(experience)),
                "num_of_skills": float(len(skills)),
                "degree_level": float(degree_score),
                "has_linkedin_profile": float(has_linkedin(text)),
                "num_of_jobs": float(number_of_jobs),
                "resume_score": resume_score
            }
        ])
        
        clf = joblib.load(r"resume_match_model.pkl")
        
        job_match_prediction = clf.predict(df)
        job_match_probabilities = clf.predict_proba(df)
        # Predict using the model
        

        # Build and return structured response
        return {
            "qualities" : {
                "total_experience": float(calculate_experience(experience)),
                "num_of_skills": float(len(skills)),
                "degree_level": float(degree_score),
                "has_linkedin_profile": float(has_linkedin(text)),
                "num_of_jobs": float(number_of_jobs),
                "resume_score": resume_score
            },
            "job_match_prediction": job_match_prediction[0],
            "job_match_probabilities": job_match_probabilities[0].tolist(),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")



def extract_number_of_jobs(text: str | None) -> int:
    """
    Extracts the number of jobs from the text.
    """
    try:
        if text is None:
            return 2
        # Use regex to find the number of jobs
        # match = re.search(r"(\d+)\s+years?\s+of\s+experience", text, re.I)
        # if match:
        #     return int(match.group(1))
        # else:
        #     return 0
    except Exception as e:
        raise Exception(f"Error extracting number of jobs: {str(e)}")
