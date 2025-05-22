from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader
import base64
import io
import re

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

        # Build and return structured response
        return {
            "total_experience": extract_experience(text),
            "skills": extract_skills(text),
            "num_of_skills": len(extract_skills(text)),
            "has_linkedin": has_linkedin(text),
            "degree_level": extract_degree_level(text)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

