from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import PDFRequest
from pypdf import PdfReader
import base64
import io
from utils import get_AI_feedback, parse_resume

app = FastAPI()

VALID_API_KEY = "1234"

@app.get("/")
def read_root():
    return {"message": "CV prediction API is running."}

@app.post("/cv/predict")
async def predict_cv(request: PDFRequest):
    # Validate API Key
    if request.api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# try:
    # Validate input
    if not request.resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required.")

    # Check job posting consistency
    has_default_job = request.job_posting is not None
    has_per_resume_jobs = any(resume_input.job_posting is not None for resume_input in request.resumes)
    
    if not has_default_job and not has_per_resume_jobs:
        raise HTTPException(status_code=400, detail="A default job posting or per-resume job postings are required.")
    
    if has_default_job and has_per_resume_jobs:
        # Warn if both are provided; prioritize per-resume job postings
        for resume_input in request.resumes:
            if not resume_input.job_posting:
                resume_input.job_posting = request.job_posting
    elif has_default_job:
        # Apply default job posting to all resumes
        for resume_input in request.resumes:
            if not resume_input.job_posting:
                resume_input.job_posting = request.job_posting
    elif has_per_resume_jobs:
        # Ensure all resumes have a job posting
        for i, resume_input in enumerate(request.resumes, 1):
            if not resume_input.job_posting:
                raise HTTPException(status_code=400, detail=f"Resume {i} is missing a job posting.")

    # Parse resumes
    parsed_resumes = []
    for resume_input in request.resumes:
        decoded_pdf = base64.b64decode(resume_input.file_base64)
        pdf_file = io.BytesIO(decoded_pdf)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        resume_data = parse_resume(text)
        parsed_resumes.append((resume_data, resume_input.job_posting))

    # Call get_AI_feedback
    response = get_AI_feedback(
        resumes=parsed_resumes,
        default_job_posting=request.job_posting,
        message_prompt=request.message_prompt
    )
    # response = {"message": "This is a placeholder response. Implement get_AI_feedback to process resumes and job postings."}
    if response is None:
        raise HTTPException(status_code=400, detail="Error processing request")
    return JSONResponse(content=response)

# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")