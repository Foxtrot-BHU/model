import asyncio
import json
import os

import google.generativeai as genai
import pdfplumber
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from google.ai.generativelanguage_v1beta.types import content

_ = load_dotenv()
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=os.getenv("LEAKED_API_KEY"))
app = FastAPI()
wsocks: list[int] = []
wsocks_metadata = {}


@app.get("/")
async def root():
    return HTMLResponse("<p style='padding-left: 2rem;'>listenin</p>")


@app.get("/extract")
async def extractEndpoint(id: int, jd: str):
    if id in wsocks:
        return JSONResponse(
            status_code=200,
            content={"id": id, "JD": f"{jd}"},
        )
    wsocks.append(id)
    wsocks_metadata[id] = jd
    return JSONResponse(
        status_code=200,
        content={"id": id, "JD": f"{jd}"},
    )


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
    await websocket.send_json(
        {"status": "Connected", "message": "Extraction started..."}
    )

    jd = wsocks_metadata[id]
    files = [
        "./uploads/Ashish_Resume_ATS.pdf",
        "./uploads/Vienna-Modern-Resume-Template.pdf",
        "./uploads/Dublin-Resume-Template-Modern.pdf",
        "./uploads/Sydney-Resume-Template-Modern.pdf",
        "./uploads/Amsterdam-Modern-Resume-Template.pdf",
        "./uploads/Stockholm-Resume-Template-Simple.pdf",
        "./uploads/London-Resume-Template-Professional.pdf",
    ]
    await websocket.send_json({"status": "Connected", "message": "Extracting Text..."})
    print("Extracting text", flush=True)
    extracted_texts = await asyncio.gather(*[extract(file) for file in files])

    await websocket.send_json({"status": "Connected", "message": "Gathering..."})
    print("Gathering...", flush=True)
    response = await asyncio.gather(*[getResponse(text) for text in extracted_texts])
    print(response)

    await websocket.send_json({"status": "Ranking started", "message": None})

    await websocket.send_json({"status": "Complete", "message": None})
    await websocket.close()

    del wsocks_metadata[id]
    wsocks.remove(id)


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
    # print(response.text, flush=True)
    return response.text


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
