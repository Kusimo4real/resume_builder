from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
from fastapi.responses import JSONResponse
from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt

app = FastAPI()

#Test APi key
VALID_API_KEY = "1234"

@app.get("/")
def read_root():
    return {"message": "CV prediction API is running."}

@app.post("/cv/predict")
async def predict_cv(
    file: UploadFile = File(...),
    api_key: str = Form(...),
):

    # Step 1: Validate File (Type, Size, API Key)
    #validate API Key
    if api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    #validate file type
    allowed_file_types = [ 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
    if file.content_type not in allowed_file_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    #Extract Text

    try:
        if file.content_type == 'application/pdf':
            extracted_text = await extract_text_from_pdf(file)
        elif file.content_type == 'application/vnd.opexmlformats-officedocument.wordprocessingml.document':
            extracted_text = await extract_text_from_docx(file)
        elif file.content_type == 'text/plain':
            extracted_text = await extract_text_from_txt(file)
        else:
            raise HTTPEXCEPTION(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    return JSONResponse({
        "message": "File and API key accepted.",
        "fiilename": file.filename,
        "content_type": file.content_type
        })
