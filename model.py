import asyncio
import glob
import hashlib
import json
import os
import random as rnd

import google.generativeai as genai
import numpy as np
import pdfplumber
import redis
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_ = load_dotenv()

CACHE_TTL = 86400
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=os.getenv("LEAKED_API_KEY"))
app = FastAPI()
rclient: redis.Redis = redis.Redis(
    host="localhost", port=6379, db=0, decode_responses=True
)
wsocks: list[int] = []
wsocks_metadata: dict[int, str] = {}
wsocks_processed: list[int] = []


origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JDRequest(BaseModel):
    jd: str


def get_file_hash(file_path: str, chunk_size: int = 4096) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@app.get("/")
async def root():
    return HTMLResponse("<p style='padding-left: 2rem;'>listenin</p>")


@app.post("/extract")
async def extractEndpoint(data: JDRequest):
    id = rnd.randint(5000, 7000)
    if id in wsocks:
        return JSONResponse(
            status_code=200,
            content={"id": id, "JD": data.jd},
        )
    wsocks.append(id)
    wsocks_metadata[id] = data.jd
    return JSONResponse(
        status_code=200,
        content={"id": id},
    )


@app.get("/analysis/{id}")
async def fetch_analysis(id: int, pgindex: int = 0, pgsize: int = 0):
    if id in wsocks:
        return JSONResponse(
            status_code=404,
            content={"datail": f"the proccess with id={id} is under processing"},
        )
    if id not in wsocks_processed:
        return JSONResponse(
            status_code=404,
            content={"datail": f"the proccess with id={id} does not exist"},
        )
    try:
        obj = json.loads(rclient.get(f"resumes:id:{id}"))
    except:
        return JSONResponse(
            status_code=404,
            content={"detail": f"The content for process id={id} does not exist"},
        )
    retOBJ = []
    pgstart = pgindex * pgsize
    pgend = (pgindex + 1) * pgsize
    if pgindex == pgsize == 0:
        pgstart = 0
        pgend = len(obj)

    if pgstart > len(obj):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Invalid index"},
        )
    count = -1
    for item in obj:
        count += 1
        if count < pgstart:
            continue
        if count >= pgend:
            break

        file_hash = item["metadata"][1]
        file_data = json.loads(rclient.get(f"resumes:{file_hash}"))
        retOBJ.append(
            {
                "Name": file_data["Name"],
                "Email": file_data["Email"],
                "Phone": file_data["Phone"],
                "Skills": file_data["Skills"],
                "Years of Experience": file_data["Years of Experience"],
                "Projects": file_data["Projects"],
                "File": file_data["File"],
                "Score": item["score"],
            }
        )

    return retOBJ


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(full_path: str):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Route '/{full_path}' not found"},
    )


@app.websocket("/extract/{id}")
async def websocket_rank(websocket: WebSocket, id: int):
    await websocket.accept()
    if id not in wsocks:
        await websocket.send_json({"error": "Invalid ID"})
        await websocket.close()
        return
    await websocket.send_json({"status": "Connected", "message": "Extraction started"})

    job_description = wsocks_metadata[id]
    await websocket.send_json(
        {"status": "Connected", "message": "Parsing job description"}
    )
    jd = await get_response(get_prompt(job_description, isJD=True))

    await websocket.send_json({"status": "Connected", "message": "Fetching files..."})
    files = glob.glob("../ui/uploads/*.pdf")
    if files == []:
        await websocket.send_json({"status": "Error", "message": "No file to proccess"})
        await websocket.close()
        return

    await websocket.send_json(
        {"status": "Connected", "message": "Munching on PDF Data"}
    )

    file_in_order = []
    cached_responses = []
    files_to_process = []
    for file in files:
        file_hash = get_file_hash(file)
        cached_data = rclient.get(f"resumes:{file_hash}")
        if cached_data:
            file_in_order.append([file, file_hash])
            cached_responses.append(json.loads(cached_data))
        else:
            files_to_process.append(file)

    if not files_to_process:
        await websocket.send_json(
            {
                "status": "Completed",
                "message": "Cached resumes found",
            }
        )

    try:
        extracted_texts = await asyncio.gather(
            *[extract(file) for file in files_to_process]
        )
    except Exception as e:
        await websocket.send_json({"status": "Error", "message": str(e)})
        await websocket.close()
        return

    await websocket.send_json(
        {"status": "Connected", "message": "Making Sense of Data"}
    )

    try:
        new_responses = await asyncio.gather(
            *[get_response(get_prompt(text)) for text in extracted_texts]
        )
    except Exception as e:
        await websocket.send_json({"status": "Error", "message": str(e)})
        await websocket.close()
        return

    for file, response in zip(files_to_process, new_responses):
        file_hash = get_file_hash(file)
        response["File"] = file
        file_in_order.append([file, file_hash])
        _ = rclient.setex(f"resumes:{file_hash}", CACHE_TTL, json.dumps(response))

    responses = cached_responses + new_responses

    await websocket.send_json(
        {"status": "Connected", "message": "Organizing Collected Data"}
    )
    try:
        parsed_res = json2text(responses)
        parsed_job = json2text(jd, isJD=True)
    except Exception as e:
        await websocket.send_json({"status": "Error", "message": str(e)})
        await websocket.close()
        return

    await websocket.send_json({"status": "Connected", "message": "Creating Vectors"})
    try:
        resume_vectors, job_vector, vectorizer = convert_to_tfidf_vectors(
            parsed_res, parsed_job[0]
        )
    except Exception as e:
        await websocket.send_json({"status": "Error", "message": e})
        await websocket.close()
        return

    await websocket.send_json(
        {"status": "Connected", "message": "Computing Similarity"}
    )
    candidate_names = [response["Name"] for response in responses]
    try:
        top_k_results, similarity_scores = weighted_similarity(
            resume_vectors, job_vector, file_in_order
        )
    except Exception as e:
        await websocket.send_json({"status": "Error", "message": str(e)})
        await websocket.close()
        return

    await websocket.send_json(
        {"status": "Connected", "message": "Hold on tight... Fetching Analysis just4u"}
    )

    await websocket.send_json({"status": "Complete", "message": None})
    await websocket.close()

    del wsocks_metadata[id]
    wsocks.remove(id)
    wsocks_processed.append(id)
    _ = rclient.setex(f"resumes:id:{id}", CACHE_TTL, json.dumps(top_k_results))


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


async def extract(pdf_path: str) -> str:
    try:
        return await asyncio.to_thread(sync_extract, pdf_path)
    except Exception as e:
        return ""


def sync_extract(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def get_prompt(description: str, isJD=False) -> str:
    if isJD:
        return f"""
        Extract the following structured data from the job description:
        Dont use any markdown in response.
        - Job Title
        - Skills
        - Years of Experience
        - Projects
        - Previous Companies
        Return the response as a JSON object. Dont use any markdown in response.

        Job Description: {description}
        """
    return f"""
    Extract the following structured data from the resume:
    Dont use any markdown in response.
    - Name: str
    - Email: list[str]
    - Phone: list[str]
    - Job Title: list[str] (Remark: Most significant max 5)
    - Skills: list[str] (Remark: Most significant max 5)
    - Years of Experience: str
    - Projects: list[str]
    - Previous Companies: list[str]
    Return the response as a JSON object.
    Dont use any markdown in response.

    Resume: {description}
    """


async def get_response(prompt: str):
    response = await model.generate_content_async(prompt)
    try:
        parsed_response = json.loads(response.text)
    except json.decoder.JsonDecodeError as e:
        parsed_response = None
    return parsed_response


def json2text(responses, isJD: bool = False) -> list[str]:
    extracted_texts: list[str] = []
    if isJD:
        text_parts = [
            f"{responses.get('Years of Experience', '') or "experienced"} years of experience in",
            f"{', '.join(responses.get('Skills', []) or  [])}",
        ]
        extracted_texts.append(" ".join(filter(None, text_parts)))
        return extracted_texts

    for response in responses:
        text_parts: list[str] = [
            response.get("Name", ""),
            f"{response.get('Job Title', '') or ""},",
            f"{response.get('Years of Experience', '') or "experienced"} years of experience in",
            f"{', '.join(response.get('Skills', []))}",
        ]
        extracted_texts.append(" ".join(filter(None, text_parts)))
    return extracted_texts


# Vectorization and Similarity Functions
def convert_to_tfidf_vectors(resume_texts: list[str], job_description: str):
    """Converts resume texts and job description to TF-IDF vectors."""
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

    # Combine resumes and job description for consistent vocabulary
    all_texts: list[str] = resume_texts + [job_description]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split into resume vectors and job vector
    resume_vectors = tfidf_matrix[:-1]  # All but last row
    job_vector = tfidf_matrix[-1]  # Last row

    return resume_vectors, job_vector, vectorizer


def weighted_similarity(
    resume_embeddings,
    job_embedding,
    candidate_names: list[str],
    weights: list[float] = [0.3, 0.7],
    k: int = -1,
):
    """
    Compute similarity with higher weight for skills match.
    weights[0] = weight for general experience
    weights[1] = weight for skills
    """
    if k == -1:
        k = len(candidate_names)
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
        {"metadata": name, "score": f"{score:.1f}"}
        for name, score in zip(top_k_names, top_k_scores)
    ]
    return top_k_results, (weights[0] * base_similarity) + (
        weights[1] * skills_similarity
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
