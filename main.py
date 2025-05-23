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

from utils import calculate_degree_level, calculate_experience, calculate_resume_score, match_resume

app = FastAPI()

VALID_API_KEY = "1234"

class PDFRequest(BaseModel):
    api_key: str
    job_posting:object | str
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

    #Extract experience
        def extract_section(text: str, section_keywords) -> str:
            pattern = r"|".join([rf"{kw}[\s:]*" for kw in section_keywords])
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                return ""
            start = match.end()
            end = len(text)
            next_section = re.search(r"\n[A-Z][^\n]{1,50}\n", text[start:])
            if next_section:
                end = start + next_section.start()
            return text[start:end].strip()
            
        def extract_experience(text: str) -> list:
            section = extract_section(text, ["experience", "work history", "professional experience"])
            lines = section.split('\n')
            return [line.strip() for line in lines if line.strip()]

        # Extract number of skills
        def extract_skills(text: str) -> list:
            section = extract_section(text, ["skills", "technical skills"])
            items = re.split(r"[•\n,-]", section)
            return [item.strip() for item in items if len(item.strip()) > 1]

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
            #return bool(re.search(r"https?://(www\.)?linkedin\.com", text, re.I))
            return bool(re.search(r"(https?://)?(www\.)?linkedin\.com", text, re.I))

        # Extract education
        def extract_degree_level(text: str) -> str:
            match = re.search(r"EDUCATION\s*(.+?)(?:\n[A-Z ]{3,}\n|\n\s*\n|$)", text, re.S | re.I)
            return match.group(1).strip() if match else ""
        
        def extract_education(text: str) -> list:
            section = extract_section(text, ["education", "academic background", "qualifications"])
            lines = section.split('\n')
            return [line.strip() for line in lines if line.strip()]

        def extract_email(text: str) -> str:
            match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
            return match.group(0) if match else ""

        def extract_phone(text: str) -> str:
            match = re.search(r"(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text)
            return match.group(0) if match else ""
        
        def extract_name(text: str) -> str:
            lines = text.strip().split('\n')
            return lines[0].strip() if lines else ""
        
        def parse_resume(text: str) -> dict:
            return {
                "name": extract_name(text),
                "email": extract_email(text),
                "phone": extract_phone(text),
                "skills": extract_skills(text),
                "education": extract_education(text),
                "experience": len(extract_experience(text)),
            }
            
        resume_data = parse_resume(text)
        # print(resume_data)
        
        response = match_resume(request.job_posting, resume_data)
        if response is None:
            raise HTTPException(status_code=400, detail="Error matching resume with job posting")
        return response
        

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
