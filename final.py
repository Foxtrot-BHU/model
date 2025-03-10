import asyncio
import json
import os

import google.generativeai as genai
import pdfplumber
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

_ = load_dotenv()

# Configure Gemini API
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=os.getenv("LEAKED_API_KEY"))


# PDF Text Extraction Functions
async def extract(pdf_path: str) -> str:
    """Extracts text from a PDF asynchronously."""
    try:
        return await asyncio.to_thread(sync_extract, pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def sync_extract(pdf_path: str) -> str:
    """Synchronous function for extracting text from PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# Gemini API Functions
def get_prompt(resume_text: str) -> str:
    """Creates a prompt for extracting structured data from resume text."""
    return f"""
    Extract the following structured data from the resume:
    - Name
    - Skills
    - Years of Experience
    - Degree
    - Projects
    - Previous Companies
    Return the response as a JSON object. Dont use any markdown in response.

    Resume: {resume_text}
    """


async def get_response(resume_text: str):
    """Gets structured data from resume text using Gemini API."""
    response = await model.generate_content_async(get_prompt(resume_text))
    print(response.text, flush=True)
    try:
        parsed_response = json.loads(response.text)
        print("Parsed Response:", parsed_response)
    except json.decoder.JsonDecodeError as e:
        print(f"Error parsing JSON: {e}")
    return response


# Vectorization and Similarity Functions
def convert_to_tfidf_vectors(resume_texts, job_description):
    """Converts resume texts and job description to TF-IDF vectors."""
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    
    # Combine resumes and job description for consistent vocabulary
    all_texts = resume_texts + [job_description]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split into resume vectors and job vector
    resume_vectors = tfidf_matrix[:-1]  # All but last row
    job_vector = tfidf_matrix[-1]       # Last row
    
    # Cache (optional)
    joblib.dump(resume_vectors, "resume_vectors.pkl")
    joblib.dump(job_vector, "job_vector.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    
    return resume_vectors, job_vector, vectorizer


def compute_cosine_similarity(resume_vectors, job_vector, candidate_names, k=5):
    """Computes cosine similarity and returns top K matching resumes."""
    # Cosine similarity between job vector and all resume vectors
    similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]
    
    # Get top K indices using argpartition
    top_k_indices = np.argpartition(similarity_scores, -k)[-k:]
    # Sort by similarity in descending order
    top_k_indices = top_k_indices[np.argsort(-similarity_scores[top_k_indices])]
    top_k_scores = similarity_scores[top_k_indices] * 100  # Convert to percentage
    top_k_names = [candidate_names[i] for i in top_k_indices]
    
    # Combine names and scores
    top_k_results = [f"{name}: {score:.1f}%" for name, score in zip(top_k_names, top_k_scores)]
    
    return top_k_results, similarity_scores


def display_results(top_k_results, similarity_scores, resume_vectors):
    """Displays the matching results."""
    print("\nTop 5 Matching Resumes:")
    for result in top_k_results:
        print(result)
    print(f"\nAverage Similarity Score: {np.mean(similarity_scores) * 100:.1f}%")
    print(f"TF-IDF Matrix Shape: {resume_vectors.shape}")


# Integrated Main Function
async def main():
    # Define PDF file paths
    files = [
        "./uploads/Ashish_Resume_ATS.pdf",
        "./uploads/Vienna-Modern-Resume-Template.pdf",
        "./uploads/Dublin-Resume-Template-Modern.pdf",
        "./uploads/Sydney-Resume-Template-Modern.pdf",
        "./uploads/Amsterdam-Modern-Resume-Template.pdf",
        "./uploads/Stockholm-Resume-Template-Simple.pdf",
        "./uploads/London-Resume-Template-Professional.pdf",
    ]
    
    # Get job description from user FIRST
    job_description = input("Enter the job description to find the best matching resumes:\n")
    
    # Extract text from PDFs
    print("Extracting text from PDFs...", flush=True)
    extracted_texts = await asyncio.gather(*[extract(file) for file in files])
    
    # Process with Gemini API
    print("Processing resumes with Gemini API...", flush=True)
    responses = await asyncio.gather(*[get_response(text) for text in extracted_texts])
    
    # Extract names from responses for candidate identification
    # This is a simplified approach; in practice you might want more robust parsing
    candidate_names = [f"Candidate from {os.path.basename(files[i])}" for i in range(len(files))]
    
    # TF-IDF vectorization
    print("Vectorizing resumes and job description...", flush=True)
    resume_vectors, job_vector, vectorizer = convert_to_tfidf_vectors(extracted_texts, job_description)
    
    # Compute similarity and find top matches
    print("Computing similarity scores...", flush=True)
    top_k_results, similarity_scores = compute_cosine_similarity(
        resume_vectors, job_vector, candidate_names, k=5
    )
    
    # Display results
    display_results(top_k_results, similarity_scores, resume_vectors)


if __name__ == "__main__":
    asyncio.run(main())