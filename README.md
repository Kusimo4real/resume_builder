# 🧠 CV Quality Prediction API

This is a FastAPI-based backend that accepts CV files (PDF, DOCX, or TXT), extracts the content, and uses a machine learning model to classify the CV as **"Good"** or **"Bad"** based on its content.

---

## 🚀 Features

- `/cv/predict` endpoint to:
  - Accept PDF, DOCX, or TXT files
  - Validate file type and API key
  - Extract text content from the uploaded file
  - Predict CV quality using a scikit-learn model
- Basic API Key protection
- Preview of extracted text
- Score with prediction confidence

---

## 🧰 Technologies Used

- Python 3.9+
- FastAPI
- pdfplumber (for PDF processing)
- python-docx (for DOCX processing)
- scikit-learn + joblib (for model)
- Uvicorn (for local development server)

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Kusimo4real/resume_builder.git
cd resume_builder
```

### 2. install dependencie
```pip install -r requirements.txt```

### 3. Run the API
```
uvicorn main:app --reload
```
