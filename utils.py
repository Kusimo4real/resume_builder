import pdfplumber
import docx
from fastapi import UploadFile
import io

async def extract_text_from_pdf(file: UploadFile) -> str:
    text = ""
    #async with file as f:
    contents = await fle.read()
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


async def extract_text_from_txt(file: UploadFile) -> str:
    async with file as f:
        contents = await f.read()
        return contents.decode("utf-8").strip()

async def extract_text_from_docx(file:UploadFile) -> str:
    async with file as f:
        contents = await f.read()
        doc = docx.Document(io.BytesIO(contents))
        return "\n".join([para.text for pata in doc.paragraphs]).strip()
