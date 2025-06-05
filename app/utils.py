from typing import List, Dict, Optional, Tuple
import pandas as pd
from fastapi import HTTPException
import joblib
from pathlib import Path
import re
from app.models import JobPosting, Resume, MatchResponse
from openai import OpenAI
from app.config import settings
from app.exceptions import APIError, ValidationError, ProcessingError, ExternalServiceError, InternalServerError
import httpx
import time
import logging
import asyncio


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

# Cache for ML models and scalers
_model_cache: Dict[str, Tuple[object, object]] = {}


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


def calculate_experience(text: str | int | float) -> int:
    """Extracts years of experience in months."""
    try:
        return int(float(text) * 12)
    except ValueError:
        raise ValidationError(
            detail="Invalid experience format. Please provide a valid number.",
            code="INVALID_EXPERIENCE"
        )
    


def calculate_degree_level(text: str) -> int:
    """Calculates a score based on degree level."""
    try:
        program = str(text).lower()
        if any(x in program for x in ["phd", "doctor"]):
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
        raise ProcessingError(
            detail=f"Error calculating degree level: {str(e)}",
            code="DEGREE_CALCULATION_ERROR"
        )
    
    
def calculate_resume_score(experience: int, num_of_skills: int, degree_level: int, has_linkedin: bool, number_of_jobs: int) -> float:
    """Calculate a resume score."""
    try:
        if any(x is None for x in [experience, num_of_skills, degree_level, number_of_jobs]):
            raise ValidationError(
                detail="Missing required inputs for resume score calculation",
                code="MISSING_SCORE_INPUTS"
            )
        score = (experience * 2) + (num_of_skills * 1.5) + (degree_level * 3) + (int(has_linkedin) * 2) + (number_of_jobs * 1.5)
        return min(score, 100)
    except ValidationError:
        raise
    except Exception as e:
        raise ProcessingError(
            detail=f"Error calculating resume score: {str(e)}",
            code="RESUME_SCORE_ERROR"
        )
    
    
    
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
    

def extract_number_of_jobs(text: Optional[str]) -> int:
    """Extracts the number of jobs from the text."""
    try:
        if text is None:
            return 2
        # Use regex to find the number of jobs
        # match = re.search(r"(\d+)\s+years?\s+of\s+experience", text, re.I)
        # if match:
        #     return int(match.group(1))
        # else:
        #     return 0
        return 2
    except Exception as e:
        raise ProcessingError(
            detail=f"Error extracting number of jobs: {str(e)}",
            code="NUMBER_OF_JOBS_ERROR"
        )


# def call_deepseek_api(prompt: str, model: str = "deepseek/deepseek-r1:free", max_retries: int = 3) -> str:
#     """Call DeepSeek API with retries and timeout."""
#     deepseek_api_key = settings.deepseek_api_key_open_router
#     if not deepseek_api_key:
#         raise ExternalServiceError(
#             detail="DeepSeek API key not configured",
#             code="MISSING_API_KEY"
#         )
    
#     client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=deepseek_api_key)
#     payload = {
#         "model": model,
#         "messages": [{"role": "user", "content": prompt}],
#     }
    
#     for attempt in range(1, max_retries + 1):
#         try:
#             with httpx.Client(timeout=30.0) as client:
#                 response = client.post(
#                     url="https://openrouter.ai/api/v1/chat/completions",
#                     headers={"Authorization": f"Bearer {deepseek_api_key}"},
#                     json=payload
#                 )
#                 response.raise_for_status()
#                 return response.json()["choices"][0]["message"]["content"].strip()
#         except (httpx.RequestError, httpx.HTTPStatusError) as e:
#             if attempt == max_retries:
#                 raise ExternalServiceError(
#                     detail=f"Failed to call DeepSeek API after {max_retries} attempts: {str(e)}",
#                     code="DEEPSEEK_API_FAILURE",
#                     context={"attempts": max_retries}
#                 )
#             time.sleep(2 ** attempt)  # Exponential backoff
#     return ""  # Should never reach here

async def call_deepseek_api_async(prompt: str, model: str = "deepseek/deepseek-r1:free", max_retries: int = 2) -> str:
    """Call DeepSeek API asynchronously with retries and timeout."""
    deepseek_api_key = settings.deepseek_api_key_open_router
    if not deepseek_api_key:
        raise ExternalServiceError(
            detail="DeepSeek API key not configured",
            code="MISSING_API_KEY"
        )
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                response = await client.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {deepseek_api_key}"},
                    json=payload
                )
                response.raise_for_status()
                logger.info(f"DeepSeek API call took {time.time() - start_time:.2f} seconds")
                return response.json()["choices"][0]["message"]["content"].strip()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == max_retries:
                    raise ExternalServiceError(
                        detail=f"Failed to call DeepSeek API after {max_retries} attempts: {str(e)}",
                        code="DEEPSEEK_API_FAILURE",
                        context={"attempts": max_retries}
                    )
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return ""

def call_deepseek_api(prompt: str, model: str = "deepseek/deepseek-r1:free", max_retries: int = 2) -> str:
    """Synchronous wrapper for DeepSeek API call."""
    return asyncio.run(call_deepseek_api_async(prompt, model, max_retries))


def assess_prompt_scope(prompt: str) -> str:
    """Assess if a prompt is HR-relevant, general conversation, or out-of-scope."""
    scope_prompt = f"""
You are an HR assistant specializing in recruitment and resume evaluation.
Determine if the following prompt is:
- 'relevant': Related to HR (e.g., resume analysis, job matching, interview prep).
- 'general': General conversation (e.g., greetings, casual chat).
- 'out-of-scope': Unrelated to HR or inappropriate (e.g., sci-fi writing, complex non-HR tasks).
Prompt: "{prompt}"
Respond with only 'relevant', 'general', or 'out-of-scope'.
    """
    response = call_deepseek_api_async(scope_prompt)
    return response

async def process_single_resume_async(resume: Resume, job_posting: JobPosting, file_base64: str) -> MatchResponse:
    """Process a single resume asynchronously against a job posting."""
    try:
        start_time = time.time()
        job_role = job_posting.job_role.strip()
        required_skills = job_posting.skills
        required_years = job_posting.experience
        required_degree = job_posting.education.lower().strip()

        # Validate inputs
        if not required_skills:
            raise ValidationError(
                detail="Job posting must include required skills",
                code="MISSING_REQUIRED_SKILLS"
            )
        if required_years < 0:
            raise ValidationError(
                detail="Required experience cannot be negative",
                code="INVALID_EXPERIENCE"
            )

        # Load model and scaler from cache or disk
        cache_key = job_role.replace(' ', '_')
        if cache_key not in _model_cache:
            model_path = Path(f"models/resume_match_model_{cache_key}.pkl")
            scaler_path = Path(f"models/scaler_{cache_key}.pkl")
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                _model_cache[cache_key] = (model, scaler)
                logger.info(f"Loaded model and scaler for {job_role}")
            except FileNotFoundError:
                raise ProcessingError(
                    detail=f"Model or scaler for job role '{job_role}' not found",
                    code="MODEL_NOT_FOUND",
                    status_code=404
                )
        model, scaler = _model_cache[cache_key]

        # Compute scores
        skill_match_score = compute_skill_match(resume["skills"], required_skills)
        resume_experience_years = resume["experience"]
        experience_match_score = compute_experience_match(resume_experience_years, required_years)
        match_score = compute_match_score(skill_match_score, experience_match_score)
        match_label = match_score_to_label(match_score)

        # Prepare features
        degree_mapping = {
            'high school': 1,
            'bachelor': 2,
            'master': 3,
            'phd': 4,
            'doctorate': 4
        }
        resume_education = resume["education"][0].lower() if resume["education"] else ''
        degree_level = next((degree_mapping[deg] for deg in degree_mapping if deg in resume_education), 2)

        total_experience_months = resume["experience"] * 12
        has_linkedin_profile = 1
        num_of_jobs = 2

        features = pd.DataFrame({
            'total_experience': [total_experience_months],
            'degree_level': [degree_level],
            'has_linkedin_profile': [has_linkedin_profile],
            'num_of_jobs': [num_of_jobs]
        })

        try:
            features['total_experience'] = features['total_experience'].clip(upper=120)
            features['num_of_jobs'] = features['num_of_jobs'].clip(upper=6)
            features_scaled = scaler.transform(features)
        except Exception as e:
            raise ProcessingError(
                detail=f"Error scaling features: {str(e)}",
                code="FEATURE_SCALING_ERROR"
            )

        try:
            probabilities = model.predict_proba(features_scaled)[0]
            prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)}
        except Exception as e:
            raise ProcessingError(
                detail=f"Error predicting probabilities: {str(e)}",
                code="MODEL_PREDICTION_ERROR"
            )

        result = MatchResponse(
            job_role=job_role,
            candidate_name=resume["name"],
            match_label=match_label,
            match_score=round(match_score, 2),
            skill_match_score=round(skill_match_score, 4),
            experience_match_score=round(experience_match_score, 4),
            probabilities=prob_dict,
            features_used=features.to_dict(orient='records')[0],
            contact_email=resume["email"],
            contact_phone=resume["phone"],
            file_base64=file_base64,
            llm_response=""
        )
        logger.info(f"Processed resume for {resume['name']} in {time.time() - start_time:.2f} seconds")
        return result
    except APIError:
        raise
    except Exception as e:
        raise ProcessingError(
            detail=f"Error processing resume: {str(e)}",
            code="RESUME_PROCESSING_ERROR"
        )



async def get_llm_response_for_resume(resume: Resume, job_posting: JobPosting, match_response: Dict) -> str:
    """Generate LLM response for a single resume."""
    start_time = time.time()
    prompt = f"""
    You are a professional hiring manager reviewing a candidate for a {match_response['job_role']} position requiring {', '.join(job_posting.skills)}, {job_posting.experience} years of experience, and a {job_posting.education} degree.
    Candidate: {match_response['candidate_name']}
    Skills: {', '.join(resume['skills'][:5])}
    Experience: {resume['experience']} years
    Education: {resume['education'][0] if resume['education'] else 'Unknown'}
    Evaluation:
    - Skill Match Score: {match_response['skill_match_score']:.4f}
    - Experience Match Score: {match_response['experience_match_score']:.4f}
    - Overall Match Score: {match_response['match_score']:.2f}
    - Match Label: {match_response['match_label']}
    Write a concise, professional response (100-250 words) evaluating the candidate’s fit, highlighting strengths, noting gaps, and suggesting next steps. Use a positive tone.
    """
    response = await call_deepseek_api_async(prompt)
    logger.info(f"Generated LLM response for {resume['name']} in {time.time() - start_time:.2f} seconds")
    return response



async def get_AI_feedback(resumes: List[Tuple[Resume, Optional[JobPosting], str]], default_job_posting: Optional[JobPosting] = None, message_prompt: Optional[str] = None) -> Dict:
    """Generate AI feedback for single/multiple resumes or conversational prompts."""
    try:
        # Case 1: Custom prompt provided
        if message_prompt:
            scope = assess_prompt_scope(message_prompt)
            
            # Build context if resumes are provided
            context = []
            if resumes and resumes[0][0]:
                for resume, job_posting, _ in resumes:
                    job_posting = job_posting or default_job_posting
                    if job_posting:
                        context.append(f"Candidate: {resume['name']}, Role: {job_posting.job_role}, Skills: {', '.join(resume['skills'][:5])}, Experience: {resume['experience']} years")
                    else:
                        context.append(f"Candidate: {resume['name']}, Skills: {', '.join(resume['skills'][:5])}, Experience: {resume['experience']} years. No job posting provided.")
            # Handle prompt based on scope
            if scope == 'relevant':
                prompt = f"""
    You are a professional HR assistant specializing in recruitment and resume evaluation.
    Context: {'; '.join(context) if context else 'No resume/job context provided.'}
    User prompt: {message_prompt}
    Respond in a professional, HR-focused manner.
                """
            elif scope == 'general':
                prompt = f"""
    You are a friendly HR assistant specializing in recruitment and resume evaluation.
    Respond naturally and conversationally to the user's prompt, keeping a positive tone.
    You can engage in general chat but subtly steer toward HR-related topics when appropriate.
    {'Context: ' +  '; '.join(context) if context else ''}
    User prompt: {message_prompt}
                """
            else:  # out-of-scope
                prompt = f"""
    You are an HR assistant specializing in recruitment and resume evaluation.
    The user's prompt seems unrelated to HR tasks.
    Respond politely, explaining you're primarily an HR assistant, and suggest returning to HR topics.
    Keep the tone positive and professional.
    User prompt: {message_prompt}
    Example response: "I'm primarily an HR assistant focused on recruitment and resume evaluation. I can help with that or chat a bit, but let's keep it relevant!"
                """
            
            llm_response = await call_deepseek_api_async(prompt)
            results = []
            if resumes and resumes[0][0]:
            # Process resumes concurrently
                results_tasks = [
                    process_single_resume_async(resume, job_posting or default_job_posting, file_base64)
                    for resume, job_posting, file_base64 in resumes if job_posting or default_job_posting
                ]
                results = await asyncio.gather(*results_tasks, return_exceptions=True)
                results = [r.__dict__ for r in results if not isinstance(r, Exception)]
            
            # logger.info(f"Processed prompt with {len(results)} resumes in {time.time() - start_time:.2f} seconds")
            return {"results": results, "llm_response": llm_response, "errors": []}
        # Case 2: Resume evaluation
        if not resumes or not (resumes[0][1] or default_job_posting):
            raise ValidationError(
                detail="At least one resume and a job posting are required for evaluation",
                code="MISSING_RESUME_OR_JOB"
            )

        # Process resumes concurrently
        results_tasks = [
            process_single_resume_async(resume, job_posting or default_job_posting, file_base64)
            for resume, job_posting, file_base64 in resumes
        ]
        results = await asyncio.gather(*results_tasks, return_exceptions=True)
        
        # Handle errors and collect successful results
        errors = []
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_dict = {
                    "detail": str(result),
                    "code": getattr(result, "code", "RESUME_PROCESSING_ERROR"),
                    "context": {"resume_index": i + 1}
                }
                errors.append(error_dict)
                logger.error(f"Resume {i + 1} failed: {error_dict}")
            else:
                valid_results.append(result.__dict__)

        if not valid_results:
            raise ProcessingError(
                detail="Failed to process all resumes",
                code="ALL_RESUMES_FAILED",
                status_code=400,
                context={"error_count": len(errors)}
            )

        # Generate individual LLM responses concurrently
        llm_tasks = [
            get_llm_response_for_resume(resumes[i][0], resumes[i][1] or default_job_posting, result)
            for i, result in enumerate(valid_results)
        ]
        llm_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
        
        for i, llm_response in enumerate(llm_responses):
            if isinstance(llm_response, Exception):
                logger.error(f"LLM response for resume {i + 1} failed: {str(llm_response)}")
                valid_results[i]['llm_response'] = f"Error generating LLM response: {str(llm_response)}"
            else:
                valid_results[i]['llm_response'] = llm_response

        # Rank resumes by match_score
        valid_results = sorted(valid_results, key=lambda x: x['match_score'], reverse=True)

        # For now the overall LLM response is not generated, but can be uncommented if needed
        # Generate overall LLM response
        # if len(valid_results) == 1:
        #     llm_response = valid_results[0]['llm_response']
        # else:
        #     context = []
        #     for i, res in enumerate(valid_results, 1):
        #         job_role = res['job_role']
        #         context.append(f"Candidate {i}: {res['candidate_name']} for {job_role}, Match Score: {res['match_score']:.2f}, Skills: {', '.join(resumes[i-1][0]['skills'][:5])}, Experience: {resumes[i-1][0]['experience']} years")
        #     prompt = f"""
        #     You are a professional hiring manager reviewing multiple candidates.
        #     Context: {'; '.join(context)}.
        #     Ranked by match score, summarize the candidates’ fit for their respective roles (or shared role if applicable). Highlight top candidates’ strengths, note any common gaps, and suggest next steps (e.g., interviews, training). Keep the response comprehensive and professional.
        #     """
        #     llm_response = await call_deepseek_api_async(prompt)

        # response = {
        #     "results": valid_results,
        #     "llm_response": llm_response,
        #     "errors": errors,
        #     "context": {"processed": len(valid_results), "failed": len(errors)}
        # }
        response = {
            "results": valid_results,
            "llm_response": "",
            "errors": errors,
            "context": {"processed": len(valid_results), "failed": len(errors)}
        }
        # logger.info(f"Completed AI feedback for {len(resumes)} resumes in {time.time() - start_time:.2f} seconds")
        return response
    except APIError:
        raise
    except Exception as e:
        raise InternalServerError(detail=f"Error generating AI feedback: {str(e)}")
