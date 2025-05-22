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