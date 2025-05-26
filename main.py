from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import PDFRequest, JobPosting
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

    try:
        resume = None
        job_posting = None

        # Parse resume if file_base64 is provided
        if request.resume_file_base64:
            decoded_pdf = base64.b64decode(request.resume_file_base64)
            pdf_file = io.BytesIO(decoded_pdf)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            resume = parse_resume(text)

        # Parse job_posting if provided
        if request.job_posting:
            job_posting = JobPosting(**request.job_posting)

        # Call get_AI_feedback with appropriate inputs
        response = get_AI_feedback(job_posting, resume, request.message_prompt)
        if response is None:
            raise HTTPException(status_code=400, detail="Error processing request")
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")