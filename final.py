import asyncio
import json
import os

import google.generativeai as genai
import pdfplumber
from dotenv import load_dotenv

_ = load_dotenv()

model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=os.getenv("LEAKED_API_KEY"))


async def extract(pdf_path: str) -> str:
    """Extracts text from a PDF asynchronously."""
    return await asyncio.to_thread(sync_extract, pdf_path)


def sync_extract(pdf_path: str) -> str:
    """Synchronous function for extracting text from PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def getPrompt(resume_text: str) -> str:
    return f"""
    Extract the following structured data from the resume:
    - Name
    - Skills
    - Years of Experience
    - Degree
    - Education
    - Projects
    - Previous Companies
    Return the response as a JSON object. Dont use any markdown in response.

    Resume: {resume_text}
    """


async def getResponse(prompt: str):
    response = await model.generate_content_async(getPrompt(prompt))
    print(response.text, flush=True)
    return response


async def main():
    files = [
        "./uploads/Ashish_Resume_ATS.pdf",
        "./uploads/Vienna-Modern-Resume-Template.pdf",
        "./uploads/Dublin-Resume-Template-Modern.pdf",
        "./uploads/Sydney-Resume-Template-Modern.pdf",
        "./uploads/Amsterdam-Modern-Resume-Template.pdf",
        "./uploads/Stockholm-Resume-Template-Simple.pdf",
        "./uploads/London-Resume-Template-Professional.pdf",
    ]
    print("Extracting text", flush=True)
    extracted_texts = await asyncio.gather(*[extract(file) for file in files])

    print("Gathering...", flush=True)
    _ = await asyncio.gather(*[getResponse(text) for text in extracted_texts])


asyncio.run(main())
