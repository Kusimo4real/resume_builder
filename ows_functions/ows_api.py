from app.config import settings
import requests


OWS_BASE_URL = settings.ows_base_url
OWS_API_KEY = settings.ows_api_key
OWS_API_VERSION = settings.ows_api_version



def get_cv_link(job_title: str) -> str:
    auth = (settings.ows_username, settings.ows_password)
    # return a URL to the CV template based on the job title
    response = requests.get(
        f"{OWS_BASE_URL}/adc-intg/api/rest/v1/HR_RESUME_APP/HR_AI_API/hr_ai_api_recommend_candidate/recommend_candidate/recommend_candidate",
        params={"job_title": job_title, "api_key": OWS_API_KEY, "api_version": OWS_API_VERSION}, auth=auth
    )
    return response
