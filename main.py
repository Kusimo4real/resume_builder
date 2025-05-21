from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
from fastapi.responses import JSONResponse
from pypdf import PdfReader

app = FastAPI()

#Test APi key - this is a test api to test the application
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
    #ensure file is a pdf
 
    # Ensure the file is a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        # Read file content into PdfReader
        contents = await file.read()
        with open("temp.pdf", "wb") as temp_file:
            temp_file.write(contents)
        reader = PdfReader("temp.pdf")
        num_pages = len(reader.pages)
        page = reader.pages[0]
        text = page.extract_text()
        return {
            "filename": file.filename,
            "total_pages": num_pages,
            "first_page_text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")
