# Resume Ranking API

## Overview
This FastAPI-based project provides a system for ranking resumes based on a job description. It uses Google's Gemini AI for text processing, TF-IDF vectorization, and cosine similarity to match resumes against job descriptions. Redis is used for caching, and WebSockets enable real-time communication.

## Features
- Extracts structured information from job descriptions and resumes using AI.
- Stores and retrieves resume data efficiently with Redis.
- Implements WebSockets for real-time resume ranking updates.
- Uses TF-IDF vectorization and cosine similarity to rank candidates.
- Provides RESTful endpoints for job description submission and result retrieval.

## Technologies Used
- **FastAPI** (for API development)
- **Redis** (for caching)
- **pdfplumber** (for PDF text extraction)
- **Google Generative AI (Gemini)** (for text analysis)
- **scikit-learn** (for vectorization and similarity computation)
- **WebSockets** (for real-time ranking updates)
- **Uvicorn** (for ASGI server)

## Installation
### Prerequisites
- Python 3.9+
- Redis server running on `localhost:6379`

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/Foxtrot-BHU/model
   cd model
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add your API key:
   ```
   LEAKED_API_KEY=your_google_gemini_api_key
   ```
5. Run the application:
   ```sh
   python model.py
   ```

## API Endpoints
### 1. Root Endpoint
**GET `/`**
- Returns a simple HTML response.

### 2. Extract Job Description
**POST `/extract`**
- **Request Body:**
  ```json
  {
    "jd": "Job description text here"
  }
  ```
- **Response:**
  ```json
  {
    "id": 1234
  }
  ```

### 3. Get Resume Analysis
**GET `/analysis/{id}`**
- Fetches resume ranking results based on a job description.
- Returns paginated results if `pgindex` and `pgsize` are provided.

### 4. WebSocket for Real-time Updates
**WebSocket `/extract/{id}`**
- Clients can connect and receive updates about the resume ranking process.

## License
This project is open-source under the MIT License.
