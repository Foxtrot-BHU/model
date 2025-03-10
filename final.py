import asyncio
import json
import os
from ctypes import sizeof

import google.generativeai as genai
import joblib
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_ = load_dotenv()

# Configure Gemini API
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=os.getenv("LEAKED_API_KEY"))


# PDF Text Extraction Functions
async def extract(pdf_path: str) -> str:
    try:
        return await asyncio.to_thread(sync_extract, pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def sync_extract(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# Gemini API Functions
def get_prompt(description: str, isJD=False) -> str:
    if isJD:
        return f"""
        Extract the following structured data from the job description:
        - Job Title
        - Skills
        - Years of Experience
        - Projects
        - Previous Companies
        Return the response as a JSON object. Dont use any markdown in response.

        Job Description: {description}
        """
    # return f"""
    # Extract and return string of the following structured data from the resume:
    # Dont use any markdown in response.
    # Name, Job Title: list[str], Skills: list[str]
    # - Years of Experience: str
    # - Projects: list[str]
    # - Previous Companies: list[str]
    # Return the response as a JSON object.
    #
    # Resume: {description}
    # """
    return f"""
    Extract the following structured data from the resume:
    Dont use any markdown in response.
    - Name: str
    - Job Title: list[str]
    - Skills: list[str]
    - Years of Experience: str
    - Projects: list[str]
    - Previous Companies: list[str]
    Return the response as a JSON object.

    Resume: {description}
    """


async def get_response(prompt: str):
    response = await model.generate_content_async(prompt)
    print(response.text, flush=True)
    try:
        parsed_response = json.loads(response.text)
    except json.decoder.JsonDecodeError as e:
        parsed_response = None
        print(f"Error parsing JSON: {e}")
    return parsed_response


def json2text(responses, isJD=False) -> list[str]:
    extracted_texts = []
    if isJD:
        text_parts = [
            f"{responses.get('Years of Experience', '') or "experienced"} years of experience in",
            f"{', '.join(responses.get('Skills', []) or  [])}",
            # f"Projects: {', '.join(responses.get('Projects', []) or [])}",
            # f"Previous companies: {' '.join(responses.get('Previous Companies') or [])}",
        ]
        extracted_texts.append(
            " ".join(filter(None, text_parts))
        )  # Join non-empty fields
        return extracted_texts

    for response in responses:
        text_parts = [
            response.get("Name", ""),
            f"{response.get('Job Title', '') or ""},",
            f"{response.get('Years of Experience', '') or "experienced"} years of experience in",
            f"{', '.join(response.get('Skills', []))}",
            # f"Projects: {' '.join(response.get('Projects', []))}",
            # f"Previous companies: {' '.join(response.get('Previous Companies') or [])}",
        ]
        extracted_texts.append(
            " ".join(filter(None, text_parts))
        )  # Join non-empty fields
    return extracted_texts


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
    job_vector = tfidf_matrix[-1]  # Last row

    # Cache (optional)
    joblib.dump(resume_vectors, "resume_vectors.pkl")
    joblib.dump(job_vector, "job_vector.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return resume_vectors, job_vector, vectorizer


def weighted_similarity(
    resume_embeddings, job_embedding, candidate_names, weights=[0.3, 0.7], k=5
):
    """
    Compute similarity with higher weight for skills match.
    weights[0] = weight for general experience
    weights[1] = weight for skills
    """
    base_similarity = cosine_similarity(job_embedding, resume_embeddings)[0]

    # Simulate a "skills-only" embedding (vector slicing would be better)
    skills_similarity = base_similarity.copy()
    skills_similarity *= 1.5  # Boost skills component artificially

    top_k_indices = np.argpartition(skills_similarity, -k)[-k:]
    # Sort by similarity in descending order
    top_k_indices = top_k_indices[np.argsort(-skills_similarity[top_k_indices])]
    top_k_scores = skills_similarity[top_k_indices] * 100  # Convert to percentage
    top_k_names = [candidate_names[i] for i in top_k_indices]

    # Combine names and scores
    top_k_results = [
        f"{name}: {score:.1f}%" for name, score in zip(top_k_names, top_k_scores)
    ]
    return top_k_results, (weights[0] * base_similarity) + (
        weights[1] * skills_similarity
    )


def compute_cosine_similarity(resume_vectors, job_vector, candidate_names, k=5):
    """Computes cosine similarity and returns top K matching resumes."""
    # Cosine similarity between job vector and all resume vectors
    similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]
    similarity_scores *= 1.5  # Boost skills component artificially

    # Get top K indices using argpartition
    top_k_indices = np.argpartition(similarity_scores, -k)[-k:]
    # Sort by similarity in descending order
    top_k_indices = top_k_indices[np.argsort(-similarity_scores[top_k_indices])]
    top_k_scores = similarity_scores[top_k_indices] * 100  # Convert to percentage
    top_k_names = [candidate_names[i] for i in top_k_indices]

    # Combine names and scores
    top_k_results = [
        f"{name}: {score:.1f}%" for name, score in zip(top_k_names, top_k_scores)
    ]

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
    job_description = input(
        "Enter the job description to find the best matching resumes:\n"
    )

    # Extract text from PDFs
    print("Extracting text from PDFs...", flush=True)
    extracted_texts = await asyncio.gather(*[extract(file) for file in files])

    # Process with Gemini API
    print("Processing resumes with Gemini API...", flush=True)
    responses = await asyncio.gather(
        *[get_response(get_prompt(text)) for text in extracted_texts]
    )

    # Extract names from responses for candidate identification
    # This is a simplified approach; in practice you might want more robust parsing
    candidate_names = [
        f"Candidate from {os.path.basename(files[i])}" for i in range(len(files))
    ]

    print("Processing job description w/ Gemini API...", flush=True)
    jd = await get_response(get_prompt(job_description, isJD=True))
    # responses.append(jd)

    parsed_res = json2text(responses)
    parsed_job = json2text(jd, isJD=True)

    print(parsed_res)
    print(parsed_job)

    # TF-IDF vectorization
    print("Vectorizing resumes and job description...", flush=True)
    resume_vectors, job_vector, vectorizer = convert_to_tfidf_vectors(
        parsed_res, parsed_job[0]
    )

    # Compute similarity and find top matches
    print("Computing similarity scores...", flush=True)
    top_k_results, similarity_scores = weighted_similarity(
        resume_vectors, job_vector, candidate_names, k=5
    )

    # Display results
    display_results(top_k_results, similarity_scores, resume_vectors)


if __name__ == "__main__":
    asyncio.run(main())
