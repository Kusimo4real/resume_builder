
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
from fastapi import HTTPException
import joblib
from pathlib import Path
import re
from app.schemas import JobPosting, Resume, MatchResponse
from openai import OpenAI
from app.config import settings
from app.exceptions import APIError, ValidationError, ProcessingError, ExternalServiceError, InternalServerError
import httpx
import time
import logging
import asyncio
from groq import Groq, AsyncGroq
import os
from collections import deque

GROQ_API_KEY = settings.groq_api_key

USERNAME = settings.ows_username
PASSWORD = settings.ows_password

OWS_BASE_URL = settings.ows_api_base_url

# SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '../ows_schema/hrms_schema_summary.txt')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '../ows_schema/hrms_schema.sql')

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


# Memory for last 5 interactions
_interaction_memory = deque(maxlen=5)  # Stores {user_query, response} pairs


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))

def fix_sql(sql: str) -> str:
    # Fix common LLM mistakes
    sql = sql.replace("FROM=", "FROM ")
    sql = sql.replace("JOIN=", "JOIN ")
    sql = sql.replace(";", "")  # Remove semicolons
    return sql.strip()

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


async def call_groq_api_async(
    prompt: str,
    # model: str = "deepseek-r1-distill-llama-70b",
    model: str = "openai/gpt-oss-120b",
    max_retries: int = 2,
    timeout: float = 5.0,
) -> str:
    """Call Groq chat completion API asynchronously with retries and timeout."""
    groq_api_key = GROQ_API_KEY
    if not groq_api_key:
        raise ExternalServiceError(
            detail="Groq API key not configured",
            code="MISSING_API_KEY",
        )

    client = AsyncGroq(api_key=groq_api_key)

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                timeout=timeout  # If SDK supports it
            )
            elapsed = time.time() - start_time
            logger.info(f"Groq API call took {elapsed:.2f} seconds (attempt {attempt})")
            return response.choices[0].message.content.strip()

        except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
            logger.warning(
                f"Groq API call attempt {attempt} failed: {e}"
            )
            if attempt == max_retries:
                raise ExternalServiceError(
                    detail=f"Failed to call Groq API after {max_retries} attempts: {str(e)}",
                    code="GROQ_API_FAILURE",
                    context={"attempts": max_retries},
                )
            backoff = 2 ** attempt
            logger.info(f"Retrying in {backoff} seconds...")
            await asyncio.sleep(backoff)

    # Safety fallback, though unreachable
    return ""

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
    return asyncio.run(call_groq_api_async(prompt, model, max_retries))


async def assess_prompt_scope(prompt: str) -> str:
    """Assess if a prompt is HR-relevant, general conversation, out-of-scope, or restricted (salary-related)."""
    # Check for salary-related keywords
    salary_keywords = r"\b(salary|salaries|compensation|pay|wage|wages|income|earnings)\b"
    if re.search(salary_keywords, prompt, re.IGNORECASE):
        return "restricted"
    
    scope_prompt = f"""
You are an HR assistant specializing in recruitment and resume evaluation.
Determine if the following prompt is:
- 'relevant': Related to HR (e.g., resume analysis, job matching, interview prep).
- 'general': General conversation (e.g., greetings, casual chat).
- 'out-of-scope': Unrelated to HR or inappropriate (e.g., sci-fi writing, complex non-HR tasks).
Prompt: "{prompt}"
Respond with only 'relevant', 'general', or 'out-of-scope'.
    """
    response = await call_groq_api_async(scope_prompt)
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
    You are a professional hiring manager for Huawei Technologies, Lagos reviewing a candidate for a {match_response['job_role']} position requiring {', '.join(job_posting.skills)}, {job_posting.experience} years of experience, and a {job_posting.education} degree.
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
    response = await call_groq_api_async(prompt)
    logger.info(f"Generated LLM response for {resume['name']} in {time.time() - start_time:.2f} seconds")
    return response

async def run_ows_tql_query(tql: str) -> Dict:
    url = f"{OWS_BASE_URL}/adc-intg/api/rest/v1/HR_RESUME_APP/HR_AI_API/hr_ai_api_hrms_data_query/hrms_data_query"
    auth = httpx.BasicAuth(USERNAME, PASSWORD)
    async with httpx.AsyncClient(auth=auth) as client:
        resp = await client.post(url, json={"tql": tql})
        resp.raise_for_status()
        return resp.json()
    
    
# async def handle_freeform_prompt(user_query: str) -> str:
#     # Step A: classify if DB lookup is needed
#     classify_prompt = f"""
#     You are a classifier. Question: "{user_query}"
#     Should this require querying the HR database (tables: hrms_dashb_employees, hrms_dashb_appraisal_data, hrms_dashb_appraisals, hrms_dashb_projects, hrms_dashb_departments, hrms_dashb_sub_departments, hrms_dashb_skills, hrms_dashb_vendors, hrms_dashb_level, hrms_dashb_roles, hrms_dashb_baseline, hrms_dashb_tenant, hrms_dashb_province_info)?
#     Answer only 'YES' or 'NO'.
#     """
#     classify_resp = await call_deepseek_api_async(classify_prompt)

#     if "NO" in classify_resp.upper():
#         # Just let LLM answer directly
#         return await call_deepseek_api_async(user_query)

#     # Step B: load schema
#     with open("../ows_schema/hrms_schema.txt") as f:
#         schema_text = f.read()

#     sql_prompt = f"""
#     You are a SQL generator.
#     Only SELECT is allowed.
#     Schema:
#     {schema_text}

#     User query: "{user_query}"
#     Return JSON with {{ "sql": "..." }} only.
#     """
#     sql_json = await call_deepseek_api_async(sql_prompt)
#     sql_query = json.loads(sql_json)["sql"]

#     # Step C: run SQL via API
#     try:
#         sql_results = await run_ows_tql_query(sql_query)
#     except Exception as e:
#         return f"I tried to run a query but failed: {str(e)}"

#     # Step D: final answer with results
#     answer_prompt = f"""
#     User asked: "{user_query}"
#     SQL executed: {sql_query}
#     Results: {sql_results}

#     Provide a natural, HR-assistant style answer using the results.
#     """
#     return await call_deepseek_api_async(answer_prompt)


def build_hr_response_prompt(context_text: str, user_prompt: str) -> str:
    """Your conditioning prompt applied consistently."""
    # Format recent interactions for context
    memory_context = ""
    if _interaction_memory:
        memory_context = "Recent Interactions:\n"
        for i, interaction in enumerate(_interaction_memory, 1):
            memory_context += f"{i}. User: {interaction['user_query']}\n   Response: {interaction['response'][:100]}...\n"
        memory_context += "\nUse this context to provide a more informed response, if relevant.\n"
    return f"""
You are a highly skilled HR assistant specializing in recruitment and resume evaluation for RNOC Department Huawei  Technologies Lagos Nigeria.

You have access to the the HRMS database and can query it as needed to assist with recruitment, employee management, and resume evaluation tasks.

**Your Task:**
1.  Analyze the 'User Prompt' in conjunction with the provided 'Resume/Job Context' and recent interactions.
2.  Determine the primary intent of the user's prompt:
    * **HR-Relevant:** If the prompt is directly about resumes, job postings, candidate evaluation, interview advice, or any recruitment-related task, considering the context.
    * **General Conversation:** If the prompt is a casual greeting, small talk, or asks for general non-HR information, but is still polite.
    * **Out-of-Scope/Inappropriate:** If the prompt is completely unrelated to HR, personal, or offensive.
3.  Based on your determination, respond following these guidelines:

**Response Guidelines:**

* **If HR-Relevant:** Provide a professional, detailed, and HR-focused response directly answering the user's prompt, leveraging the 'Resume/Job Context' as needed.
* **If General Conversation:** Respond naturally and conversationally, maintaining a positive and friendly HR assistant tone. You can engage in brief general chat but subtly steer back towards HR-related topics.
* **If Out-of-Scope/Inappropriate:** Politely explain that you are primarily an HR assistant and suggest returning to HR or recruitment topics. Keep the tone professional and helpful. Do NOT answer non-HR questions directly. Example: "I'm primarily an HR assistant focused on recruitment and resume evaluation. How can I assist you with your career or resume needs?"

---
**Memory Context:**
{memory_context if memory_context else 'No recent interactions.'}

**Resume/Job Context:**
{context_text if context_text else 'No specific resume or job posting context provided.'}

**User Prompt:**
{user_prompt}

---

**Your Response:**
""".strip()

def summarize_sql_results(payload: dict, max_chars: int = 2500) -> str:
    """
    Compact preview for the LLM context.
    Assumes your SQL API returns JSON. We just truncate the JSON string safely.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    return (text[:max_chars] + "…") if len(text) > max_chars else text

def is_safe_select(sql: str) -> bool:
    """Very simple safety gate: allow only single-statement SELECTs."""
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    # forbid multiple statements / known dangerous keywords
    forbidden = [";", " update ", " delete ", " insert ", " drop ", " alter ",
                 " truncate ", " grant ", " revoke ", " create ", " execute ",
                 " call ", " into "]
    return not any(tok in f" {s} " for tok in forbidden)

async def handle_freeform_prompt(user_query: str, call_deepseek_api_async) -> str:
    """
    Free-form path:
      1) classify if DB lookup needed
      2) if needed: generate SQL -> run -> inject results
      3) ALWAYS finish with your conditioning prompt
    """
    # A. Check for salary-related keywords first
    classify_resp = (await assess_prompt_scope(user_query)).strip().lower()
    if classify_resp == "restricted":
        logger.info("Prompt classified as restricted due to salary-related content.")
        final_prompt = build_hr_response_prompt(context_text="Queries about employee salaries are restricted due to company policy.", user_prompt=user_query)
        response = await call_deepseek_api_async(final_prompt)
        _interaction_memory.append({"user_query": user_query, "response": response})
        return response
    
    # B. Classify whether to query DB
    classify_prompt = f"""
You are a classifier. Question: "{user_query}"
Should this require querying the HR database (tables include employees, appraisals, projects, awards, departments, sub_departments, skills, vendors, roles, level, tenant)?
Answer only YES or NO.
""".strip()
    classify_resp = (await call_deepseek_api_async(classify_prompt)).strip().upper()
    logger.info(f"Classification response: {classify_resp}")
    # C. If NO DB needed: answer directly with conditioning
    if "NO" in classify_resp:
        logger.info("Skipping DB lookup as per classification.")
        final_prompt = build_hr_response_prompt(
            context_text="No specific resume or job posting context provided.",
            user_prompt=user_query
        )
        return await call_deepseek_api_async(final_prompt)

    # D. Load summarized schema (for SQL generation)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_text = f.read()
    except FileNotFoundError:
        logger.warning("Schema file not found, using fallback")
        schema_text = "hrms_dashb_employees(...), hrms_dashb_appraisals(...), hrms_dashb_awards(...)"  # tiny fallback

    # E. Ask LLM for SQL (JSON only)
    sql_prompt = f"""
    
You translate natural language to SQL for the given schema.

Schema:
{schema_text}

Rules:
- Generate a single-statement SELECT query only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, or multiple statements.
- Put all table names in double quotes and prepend with "/HRMS_Dashboard/HRMS_Dashboard/"
- Table names must appear as FROM "<table_name>" (with a space, never '=')
- Do not add a semicolon at the end.
- Use JOINs as needed, with proper ON clauses.
- Return only valid JSON with no extra text, no explanations, and no markdown.

Format: {{"tql":"..."}}

User query: "{user_query}"
""".strip()

    sql_prompt = f"""
    You are an expert SQL query generator for an HRMS database. Translate the user's natural language query into a single, valid SELECT query.

Schema:
{schema_text}

Key Relationships (Foreign Keys):
- hrms_dashb_employees.subdept_name → hrms_dashb_sub_departments.subdept_name
- hrms_dashb_sub_departments.dept_name → hrms_dashb_departments.dept_name
- hrms_dashb_appraisals.employee_id → hrms_dashb_employees.employee_id
- hrms_dashb_awards.employee_id → hrms_dashb_employees.employee_id
- hrms_dashb_skills.employee_id → hrms_dashb_employees.employee_id
(And so on for others—extract these from your full schema.sql)
- Departments and team names may be referred to interchangeably so recognize that and let your queries check both tables every time.
- The word autin refers to the AUTIN team within Huawei.

Rules:
- Generate ONLY a single SELECT statement. No INSERT/UPDATE/DELETE, no multiple statements, no semicolons.
- Always prepend table names with "/HRMS_Dashboard/HRMS_Dashboard/" and enclose in double quotes, e.g., FROM "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_employees".
- Use JOINs with ON clauses for multi-table queries. Use table aliases (e.g., AS e for employees).
- Handle aggregations (COUNT, SUM, AVG) if the query asks for totals, averages, etc.
- Avoid using COUNT(*): When counting rows, specify a particular column instead of using the wildcard. For example, use COUNT(column_name) instead of COUNT(*)
- For filters, use WHERE clauses. Use LIKE for partial matches.
- If the query is ambiguous, make reasonable assumptions but prioritize accuracy.
- When referencing a person's name, always apply proper casing by capitalizing the first letters of the name.
- For name searches: Use LIKE with wildcards (e.g., fullname LIKE '%John%') for partial or ambiguous names (e.g., 'John', 'Smith'). Use exact match (e.g., fullname = 'John Doe') for full names or when the query specifies an exact employee. If unsure, prefer LIKE for single names and = for full names.
- Think step-by-step: 1) Identify relevant tables. 2) Determine joins needed. 3) Add SELECT columns, WHERE filters, GROUP BY if aggregating. 4) Output the query.

Examples:
Query: "List all employees in the IT department."
Reasoning: Tables: hrms_dashb_employees (has department_two), hrms_dashb_sub_departments (links to dept_name). Join on subdept_name. Filter on dept_name = 'IT'.
SQL: SELECT e.fullname, e.employee_position FROM "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_employees" AS e JOIN "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_sub_departments" AS sd ON e.subdept_name = sd.subdept_name WHERE sd.dept_name = 'IT'

Query: "What is the average KPI score for employees with more than 5 years experience, grouped by department?"
Reasoning: Tables: hrms_dashb_employees (hire_date for experience), hrms_dashb_appraisal_data (kpi_score, department). Join on fullname or account_id (assuming match). Calculate experience as (CURRENT_DATE - hire_date) > 5 years. Aggregate AVG(kpi_score) GROUP BY department.
SQL: SELECT ad.department, AVG(ad.kpi_score) AS avg_kpi FROM "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_appraisal_data" AS ad JOIN "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_employees" AS e ON ad.fullname = e.fullname WHERE (CURRENT_DATE - e.hire_date) > INTERVAL '5 years' GROUP BY ad.department

Query: "Count awards per employee in Lagos location."
Reasoning: Tables: hrms_dashb_awards (awards), hrms_dashb_employees (location, fullname). Join on employee_id. Filter location = 'Lagos'. Aggregate COUNT GROUP BY employee.
SQL: SELECT e.fullname, COUNT(a.award_id) AS award_count FROM "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_employees" AS e JOIN "/HRMS_Dashboard/HRMS_Dashboard/hrms_dashb_awards" AS a ON e.employee_id = a.employee_id WHERE e.location = 'Lagos' GROUP BY e.fullname

User query: "{user_query}"

Return ONLY valid JSON: {{"tql": "your_sql_here"}}
    
    """

    sql_json_raw = await call_deepseek_api_async(sql_prompt)
    
    logger.info(f"Generated SQL JSON: {sql_json_raw}")

    # E. Parse & validate SQL
    try:
        sql_obj = extract_json(sql_json_raw)
        logger.info(f"Extracted SQL JSON object: {sql_obj}")
        sql_query = json.loads(sql_json_raw)["tql"]
    except Exception as e:
        logger.exception(f"Error parsing SQL JSON from LLM: {e}")
        # If the LLM didn't return valid JSON, fall back to conditioned answer without DB
        fallback_prompt = build_hr_response_prompt(
            context_text="No specific resume or job posting context provided.",
            user_prompt=user_query
        )
        return await call_deepseek_api_async(fallback_prompt)

    if not is_safe_select(sql_query):
        # refuse unsafe SQL; still answer, but without DB context
        fallback_prompt = build_hr_response_prompt(
            context_text="DB query was rejected for safety. Proceed without DB context.",
            user_prompt=user_query
        )
        return await call_deepseek_api_async(fallback_prompt)

    # F. Run SQL and summarize result
#     try:
#         sql_query = fix_sql(sql_query)
#         logger.info(f"Executing TQL: {sql_query}")
#         sql_results = await run_ows_tql_query(sql_query)
#         preview = summarize_sql_results(sql_results)
#         db_context = f"TQL executed: {sql_query}\nResults preview: {preview}"
#         logger.info(f"SQL executed successfully. Context length: {len(db_context)} chars")
#     except Exception as e:
#         # db_context = f"Attempted DB lookup but failed: {str(e)}"
#         db_context = """
# Database lookup failed. The HRMS system is currently unavailable.

# Rules for responding:
# - Do not show technical details (error codes, API URLs, stack traces).
# - Explain the issue in simple, non-technical terms.
# - Suggest practical next steps for the user (e.g., try again later, check HRMS portal, contact AUTIN team for support if it persists).
#         """
#         logger.error(db_context)

#     # G. Final answer with your conditioning prompt
#     final_prompt = build_hr_response_prompt(
#         context_text=db_context,
#         user_prompt=user_query
#     )
#     return await call_deepseek_api_async(final_prompt)

    max_retries = 3
    db_context = ""
    for attempt in range(1, max_retries + 1):
        try:
            sql_query = fix_sql(sql_query)
            logger.info(f"Executing TQL (attempt {attempt}): {sql_query}")
            sql_results = await run_ows_tql_query(sql_query)
            preview = summarize_sql_results(sql_results)
            db_context = f"TQL executed: {sql_query}\nResults preview: {preview}"
            logger.info(f"SQL executed successfully. Context length: {len(db_context)} chars")
            break  # Success, exit loop
        except Exception as e:
            error_message = str(e)
            logger.error(f"SQL execution failed (attempt {attempt}): {error_message}")
            if attempt == max_retries:
                db_context = """
Database lookup failed after retries. The HRMS system is currently unavailable.

Rules for responding:
- Do not show technical details (error codes, API URLs, stack traces).
- Explain the issue in simple, non-technical terms.
- Suggest they try again with a more specific query. e.g. "Employee name, department, location, project, appraisal, award, skill, vendor"
- Also suggest practical next steps for the user (e.g., try again later, check HRMS portal, contact AUTIN Technical team for support if it persists).
                """
                break
            # Generate correction
            correction_prompt = f"""
The previous SQL failed with error: {error_message}
Original query: {user_query}
Previous SQL: {sql_query}
Schema: {schema_text}
Fix the SQL step-by-step and return new JSON: {{"tql": "fixed_sql_here"}}
""".strip()
            correction_json_raw = await call_deepseek_api_async(correction_prompt)
            try:
                correction_obj = extract_json(correction_json_raw)
                sql_query = correction_obj["tql"]
                logger.info(f"Generated corrected SQL (attempt {attempt}): {sql_query}")
            except Exception as corr_e:
                logger.exception(f"Error parsing correction SQL: {corr_e}")
                db_context = "Failed to correct SQL after error."
                break

    # G. Final answer with your conditioning prompt
    final_prompt = build_hr_response_prompt(
        context_text=db_context,
        user_prompt=user_query
    )
    return await call_deepseek_api_async(final_prompt)



async def get_AI_feedback(resumes: List[Tuple[Resume, Optional[JobPosting], str]], default_job_posting: Optional[JobPosting] = None, message_prompt: Optional[str] = None) -> Dict:
    """Generate AI feedback for single/multiple resumes or conversational prompts."""
    try:
        start_time = time.time()
        # Case 1: Custom prompt provided
        if message_prompt:
            # scope = await assess_prompt_scope(message_prompt)
            if not resumes or not resumes[0][0]:   # no resumes provided
                overall_llm_response = await handle_freeform_prompt(user_query=message_prompt, call_deepseek_api_async=call_groq_api_async)
                return {"results": [], "llm_response": overall_llm_response, "errors": []}
            
            # Build context if resumes are provided
            context = []
            if resumes and resumes[0][0]:
                for resume, job_posting, _ in resumes:
                    job_posting = job_posting or default_job_posting
                    if job_posting:
                        context.append(f"Candidate: {resume['name']}, Role: {job_posting.job_role}, Skills: {', '.join(resume['skills'][:5])}, Experience: {resume['experience']} years")
                    else:
                        context.append(f"Candidate: {resume['name']}, Skills: {', '.join(resume['skills'][:5])}, Experience: {resume['experience']} years. No job posting provided.")
                        
            prompt = f"""
You are a highly skilled HR assistant specializing in recruitment, employee management, and resume evaluation for Huawei Technologies Nigeria.

**Your Task:**
1.  Analyze the 'User Prompt' in conjunction with the provided 'Resume/Job Context'.
2.  Determine the primary intent of the user's prompt:
    * **HR-Relevant:** If the prompt is directly about resumes, job postings, candidate evaluation, interview advice, or any HR related task, considering the context.
    * **General Conversation:** If the prompt is a casual greeting, small talk, or asks for general non-HR information, but is still polite.
    * **Out-of-Scope/Inappropriate:** If the prompt is completely unrelated to HR, personal, or offensive.
3.  Based on your determination, respond following these guidelines:

**Response Guidelines:**

* **If HR-Relevant:** Provide a professional, detailed, and HR-focused response directly answering the user's prompt, leveraging the 'Resume/Job Context/database' as needed.
* **If General Conversation:** Respond naturally and conversationally, maintaining a positive and friendly HR assistant tone. You can engage in brief general chat but subtly steer back towards HR-related topics.
* **If Out-of-Scope/Inappropriate:** Politely explain that you are primarily an HR assistant and suggest returning to HR or recruitment topics. Keep the tone professional and helpful. Do NOT answer non-HR questions directly. Example: "I'm primarily an HR assistant focused on recruitment and resume evaluation. How can I assist you with your career or resume needs?"

---

**Resume/Job Context:**
{'; '.join(context) if context else 'No specific resume or job posting context provided.'}

**User Prompt:**
{message_prompt}

---

**Your Response:**
"""

            # overall_llm_response = await call_deepseek_api_async(prompt)
            overall_llm_response_task = asyncio.create_task(call_groq_api_async(prompt))
            results = []
            if resumes and resumes[0][0]:
            # Process resumes concurrently
                results_tasks = [
                    process_single_resume_async(resume, job_posting or default_job_posting, file_base64)
                    for resume, job_posting, file_base64 in resumes if job_posting or default_job_posting
                ]
                results = await asyncio.gather(*results_tasks, return_exceptions=True)
                results = [r.__dict__ for r in results if not isinstance(r, Exception)]
                
                # get llm responses for each resume
                llm_tasks = [
                    get_llm_response_for_resume(resumes[i][0], resumes[i][1] or default_job_posting, results[i])
                    for i in range(len(results))
                ]
                llm_responses = await asyncio.gather(*llm_tasks, return_exceptions=True)
                for i, llm_response in enumerate(llm_responses):
                    if isinstance(llm_response, Exception):
                        logger.error(f"LLM response for resume {i + 1} failed: {str(llm_response)}")
                        results[i]['llm_response'] = f"Error generating LLM response: {str(llm_response)}"
                    else:
                        results[i]['llm_response'] = llm_response
            overall_llm_response = ""
            try:
                overall_llm_response = await overall_llm_response_task
            except Exception as e:
                logger.error(f"Failed to get overall LLM response: {str(e)}")
                overall_llm_response = "Something went wrong while responding. Please try again later."
            logger.info(f"Processed prompt with {len(results)} resumes in {time.time() - start_time:.2f} seconds")
            return {"results": results, "llm_response": overall_llm_response, "errors": []}
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
