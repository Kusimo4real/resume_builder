from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import PDFRequest
from pypdf import PdfReader
import base64
import io

from utils import match_resume, parse_resume

app = FastAPI()

VALID_API_KEY = "1234"



# home endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "CV prediction API is running."}

# Endpoint to handle CV prediction
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

        resume_data = parse_resume(text)
        # print(resume_data)
        
        response = match_resume(request.job_posting, resume_data)
        if response is None:
            raise HTTPException(status_code=400, detail="Error matching resume with job posting")
        return response
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")