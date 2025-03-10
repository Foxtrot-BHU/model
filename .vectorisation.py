from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

# Step 1: Convert resume texts and job description to TF-IDF vectors
def convert_to_tfidf_vectors(resume_texts, job_description):
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

# Step 2: Compute cosine similarity and get top K resumes
def compute_cosine_similarity(resume_vectors, job_vector, candidate_names, k=5):
    # Cosine similarity between job vector and all resume vectors
    similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]
    
    # Get top K indices using argpartition
    top_k_indices = np.argpartition(similarity_scores, -k)[-k:]
    top_k_scores = similarity_scores[top_k_indices] * 100  # Convert to percentage
    top_k_names = [candidate_names[i] for i in top_k_indices]
    
    # Combine names and scores
    top_k_results = [f"{name}: {score:.1f}%" for name, score in zip(top_k_names, top_k_scores)]
    
    return top_k_results, similarity_scores

# Step 3: Display results
def display_results(top_k_results, similarity_scores):
    print("Top 5 Matching Resumes:")
    for result in top_k_results:
        print(result)
    print(f"\nAverage Similarity Score: {np.mean(similarity_scores) * 100:.1f}%")
    print(f"TF-IDF Matrix Shape: {resume_vectors.shape}")

if __name__ == "__main__":
    # Dummy data: Assume 100 extracted resume texts and names
    # Replace with your actual extracted data
    resume_texts = [
        "ashish kushwaha python flask fastapi sql 4 years software engineer",
        "john doe java c++ 2 years developer",
        "jane smith python flask 3 years software developer management",
        "jack sparrow devops engineer 6 years backend and proficient in python",
        "loremi inpusmus html css developer 10 years best with AI",
        # ... 97 more resumes in similar format ...
    ] * 100  # Simulate 100 resumes
    candidate_names = [f"candidate_{i}" for i in range(100)]  # Placeholder names
    
    # Job description
    job_description = "I need a software developer having 3 years of experience and having skills in Flask, Python"
    
    # TF-IDF vectorization
    resume_vectors, job_vector, vectorizer = convert_to_tfidf_vectors(resume_texts, job_description)
    
    # Cosine similarity and top K
    top_k_results, similarity_scores = compute_cosine_similarity(resume_vectors, job_vector, candidate_names, k=5)
    
    # Display results
    display_results(top_k_results, similarity_scores)
    