
from typing import List, Dict, Optional, Tuple
import pandas as pd
from fastapi import HTTPException
import joblib
import pandas as pd
from pathlib import Path
import re
import requests
from models import JobPosting, Resume, MatchResponse
from openai import OpenAI
from config import settings



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

def calculate_experience(text: str) -> int:
    """
    Extracts years of experience from a given text.
    returns experience in months as an integer.
    """
    try:
        return int(text * 12)
    except ValueError:
        raise ValueError("Invalid experience format. Please provide a valid number.")
    except Exception as e:
        raise Exception(f"Error calculating experience: {str(e)}")
    


def calculate_degree_level(text: str) -> int:
    '''
    calculates a score based on degree level
    '''
    try:
        program = str(text).lower()
        if "phd" or "doctor" in program:
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
        raise Exception(f"Error calculating degree level: {str(e)}")
    
    
def calculate_resume_score(experience: int, num_of_skills: int, degree_level: int, has_linkedin: bool, number_of_jobs: int) -> float:
    """
    Calculate a resume score based on experience, skills, degree level, and LinkedIn presence.
    """
    try:
        score = (experience * 2) + (num_of_skills * 1.5) + (degree_level * 3) + (int(has_linkedin) * 2) + (number_of_jobs * 1.5)
        return min(score, 100)  # Cap the score at 100
    except ValueError:
        raise ValueError("Invalid input for resume score calculation. Please check the values provided.")
    except TypeError:
        raise TypeError("Invalid type for resume score calculation. Please ensure all inputs are of the correct type.")
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero error in resume score calculation. Please check the inputs.")
    except OverflowError:
        raise OverflowError("Overflow error in resume score calculation. Please check the inputs.")
    except Exception as e:
        raise Exception(f"Error calculating resume score: {str(e)}")
    
    
    
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



def call_deepseek_api(prompt: str, model: str = "deepseek/deepseek-r1:free", max_tokens: int = 200) -> str:
    deepseek_api_key = settings.deepseek_api_key_open_router
    if not deepseek_api_key:
        return "Error: DeepSeek API key not configured."
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=deepseek_api_key)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: Failed to generate response ({str(e)})"


def classify_prompt_intent(prompt: str) -> bool:
    classification_prompt = f"""
You are an HR assistant specializing in recruitment, resume evaluation, and candidate feedback. 
Determine if the following user prompt is related to HR operations (e.g., resume analysis, job matching, interview preparation, candidate evaluation, employee relations, or job description drafting).
Prompt: "{prompt}"
Respond with only "HR-related" or "non-HR" based on the prompt's intent.
    """
    response = call_deepseek_api(classification_prompt, max_tokens=10)
    return response == "HR-related"

def process_single_resume(resume: Resume, job_posting: JobPosting) -> MatchResponse:
    job_role = job_posting.job_role.strip()
    required_skills = job_posting.skills
    required_years = job_posting.experience
    required_degree = job_posting.education.lower().strip()

    # Load model and scaler
    model_path = Path(f"models/resume_match_model_{job_role.replace(' ', '_')}.pkl")
    scaler_path = Path(f"models/scaler_{job_role.replace(' ', '_')}.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model or scaler for job role '{job_role}' not found")

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

    features['total_experience'] = features['total_experience'].clip(upper=120)
    features['num_of_jobs'] = features['num_of_jobs'].clip(upper=6)
    features_scaled = scaler.transform(features)

    probabilities = model.predict_proba(features_scaled)[0]
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probabilities)}

    return MatchResponse(
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
        llm_response=""
    )

def get_AI_feedback(resumes: List[Tuple[Resume, Optional[JobPosting]]], default_job_posting: Optional[JobPosting] = None, message_prompt: Optional[str] = None) -> Dict:
    """Generate AI feedback for single or multiple resumes or custom HR prompt."""
    # Case 1: Custom prompt provided
    if message_prompt:
        if not classify_prompt_intent(message_prompt):
            return {"llm_response": "This AI is restricted to HR-related tasks, such as resume evaluation, job matching, interview preparation, or recruitment advice."}
        
        # Include resume/job context if provided
        if resumes and resumes[0][0]:
            context = []
            for resume, job_posting in resumes:
                job_posting = job_posting or default_job_posting
                if job_posting:
                    context.append(f"Candidate: {resume['name']}, Role: {job_posting.job_role}, Skills: {', '.join(resume['skills'][:5])}, Experience: {resume['experience']} years")
            prompt = f"""
You are a professional HR assistant specializing in recruitment and resume evaluation. 
Context: {'; '.join(context) if context else 'No resume/job context provided.'}
User prompt: {message_prompt}
Respond concisely in a professional, HR-focused manner.
            """
        else:
            # Standalone HR prompt without resume context
            prompt = f"""
You are a professional HR assistant specializing in recruitment and resume evaluation. 
User prompt: {message_prompt}
Respond in a professional, HR-focused manner.
            """
        llm_response = call_deepseek_api(prompt)
        results = []
        if resumes and resumes[0][0]:
            for resume, job_posting in resumes:
                job_posting = job_posting or default_job_posting
                if job_posting:
                    match_response = process_single_resume(resume, job_posting)
                    results.append(match_response.__dict__)
        return {"results": results, "llm_response": llm_response}

    # Case 2: Resume evaluation (single or multiple)
    if not resumes or not (resumes[0][1] or default_job_posting):
        return {"llm_response": "At least one resume and a job posting are required for evaluation."}

    # Process resumes
    results = []
    for resume, job_posting in resumes:
        job_posting = job_posting or default_job_posting
        if job_posting:
            match_response = process_single_resume(resume, job_posting)
            results.append(match_response.__dict__)

    # Rank resumes by match_score
    results = sorted(results, key=lambda x: x['match_score'], reverse=True)

    # Generate single LLM response
    if len(results) == 1:
        res = results[0]
        job_posting_for_single_resume = default_job_posting if default_job_posting else resumes[0][1]
        print(type(res))
        prompt = f"""
You are a professional hiring manager reviewing a candidate for a {res['job_role']} position requiring {', '.join(job_posting_for_single_resume.skills)}, {job_posting_for_single_resume.experience} years of experience, and a {job_posting_for_single_resume.education} degree. 
Candidate: {res['candidate_name']}
Skills: {', '.join(resumes[0][0]['skills'][:5])}
Experience: {resumes[0][0]['experience']} years
Education: {resumes[0][0]['education'][0] if resumes[0][0]['education'] else 'Unknown'}
Evaluation:
- Skill Match Score: {res['skill_match_score']:.4f}
- Experience Match Score: {res['experience_match_score']:.4f}
- Overall Match Score: {res['match_score']:.2f}
- Match Label: {res['match_label']}
Write a professional response evaluating the candidate’s fit, highlighting strengths, noting gaps, and suggesting next steps. Use a positive tone.
        """
    else:
        # Multiple resumes
        context = []
        for i, res in enumerate(results, 1):
            job_role = res['job_role']
            context.append(f"Candidate {i}: {res['candidate_name']} for {job_role}, Match Score: {res['match_score']:.2f}, Skills: {', '.join(resumes[i-1][0]['skills'][:5])}, Experience: {resumes[i-1][0]['experience']} years")
        prompt = f"""
You are a professional hiring manager reviewing multiple candidates. 
Context: {'; '.join(context)}.
Ranked by match score, summarize the candidates’ fit for their respective roles (or shared role if applicable). Highlight top candidates’ strengths, note any common gaps, and suggest next steps (e.g., interviews, training). Keep the response comprehensive and professional.
        """
    llm_response = call_deepseek_api(prompt)

    return {"results": results, "llm_response": llm_response}