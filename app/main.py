from email import errors
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.models import PDFRequest
from pypdf import PdfReader
from app.exceptions import APIError, ValidationError, AuthenticationError, ProcessingError, InternalServerError
import base64
import io
import logging
from app.utils import get_AI_feedback, parse_resume
import traceback
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

VALID_API_KEY = "1234"


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, APIError):
        error_dict = exc.to_dict()
        status_code = exc.status_code
    else:
        error_dict = {
            "detail": "Unexpected error occurred",
            "code": "UNEXPECTED_ERROR",
            "context": {"error_type": str(type(exc).__name__)}
        }
        status_code = 500
        logger.error(f"Unexpected error: {str(exc)}\n{traceback.format_exc()}")

    logger.error(f"Error: {error_dict['code']} - {error_dict['detail']}, Context: {error_dict['context']}")
    return JSONResponse(status_code=status_code, content=error_dict)


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "CV prediction API is running."}


@app.post("/cv/predict")
async def predict_cv(request: PDFRequest):
    start_time = time.time()
    logger.info(f"Received request with {len(request.resumes)} resumes and message_prompt: {request.message_prompt is not None}")    # Validate API Key
    if request.api_key != VALID_API_KEY:
        logger.error("Invalid API Key provided.")
        raise AuthenticationError()

# try:
    # Validate input
    if not request.resumes and not request.message_prompt:
        raise ValidationError(
            detail="At least one resume or a message prompt is required.",
            code="EMPTY_REQUEST"
        )
        
    # Check job posting consistency
    has_default_job = request.job_posting is not None
    
    # This implementation originally checked for a job posting attached to each resume
    # or a default job posting, but now we will only check for a single job posting for all resumes
    # I am leaving the original logic in place for now, but it can be simplified later
    has_per_resume_jobs = any(resume_input.job_posting is not None for resume_input in request.resumes)
    
    if not has_default_job and not has_per_resume_jobs and not request.message_prompt:
        raise ValidationError("A job posting or message prompt is required")
    if has_default_job and has_per_resume_jobs:
        logger.warning("Both default and per-resume job postings provided; prioritizing per-resume job postings")
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
                raise ValidationError(f"Resume {i} is missing a job posting", context={"resume_index": i})

    # Parse resumes
    parsed_resumes = []
    errors = []
    for i, resume_input in enumerate(request.resumes, 1):
        try:
            # Validate base64
            try:
                decoded_pdf = base64.b64decode(resume_input.file_base64, validate=True)
            except base64.binascii.Error:
                raise ProcessingError(
                    detail="Invalid base64 data",
                    code="INVALID_BASE64",
                    context={"resume_index": i}
                )

            pdf_file = io.BytesIO(decoded_pdf)
            try:
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if not text.strip():
                    raise ProcessingError(
                        detail="No text extracted from PDF",
                        code="EMPTY_PDF",
                        context={"resume_index": i}
                    )
            except Exception as e:
                raise ProcessingError(
                    detail=f"Failed to parse PDF: {str(e)}",
                    code="PDF_PARSING_ERROR",
                    context={"resume_index": i}
                )

            resume_data = parse_resume(text)
            parsed_resumes.append((resume_data, resume_input.job_posting, resume_input.file_base64))
        except APIError as e:
            errors.append(e.to_dict())
            logger.error(f"Resume {i} processing failed: {e.detail}, Code: {e.code}, Context: {e.context}")
        except Exception as e:
            error_dict = {
                "detail": f"Unexpected error processing resume: {str(e)}",
                "code": "UNEXPECTED_RESUME_ERROR",
                "context": {"resume_index": i}
            }
            errors.append(error_dict)
            logger.error(f"Resume {i} unexpected error: {str(e)}\n{traceback.format_exc()}")

    if not parsed_resumes and errors:
        raise ProcessingError(
            detail="Failed to process all resumes",
            code="ALL_RESUMES_FAILED",
            status_code=400,
            context={"error_count": len(errors)}
        )

    # Call get_AI_feedback
    try:
        response = await get_AI_feedback(
            resumes=parsed_resumes,
            default_job_posting=request.job_posting,
            message_prompt=request.message_prompt
        )
        if response is None:
            raise InternalServerError("Error processing request: get_AI_feedback returned None")
        
        # Add errors to response if any
        if errors:
            response["errors"] = errors
            response["context"] = {"processed": len(parsed_resumes), "failed": len(errors)}
    except APIError as e:
        raise e
    except Exception as e:
        raise InternalServerError(detail=f"Error processing request: {str(e)}")
    logger.info(f"Request processed in {time.time() - start_time:.2f} seconds")

    return JSONResponse(content=response)