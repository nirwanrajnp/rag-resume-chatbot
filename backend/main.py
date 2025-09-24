from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from ollama_embeddings import get_ollama_embeddings
import requests
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:1.5b")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# Initialize components
logger.info("Initializing Ollama bge-m3 embeddings...")
embedding_model = get_ollama_embeddings("bge-m3:latest")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection("resume_knowledge")
    logger.info(f"Loaded existing ChromaDB collection with {collection.count()} documents")
except Exception as e:
    logger.error(f"Failed to load ChromaDB collection: {e}")
    logger.info("Creating new empty ChromaDB collection - you may need to run ingestion script")
    collection = chroma_client.create_collection("resume_knowledge")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

def get_relevant_context(question: str, top_k: int = 2) -> tuple[str, list]:
    query_embedding = embedding_model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding,  # Already a list from Ollama
        n_results=top_k
    )

    if not results['documents'] or not results['documents'][0]:
        return "No relevant context found.", []

    context_parts = []
    sources = []

    for i, doc in enumerate(results['documents'][0]):
        # Use more context but still limit for performance
        truncated_doc = doc[:500] + "..." if len(doc) > 500 else doc
        context_parts.append(truncated_doc)
        if results['metadatas'] and results['metadatas'][0]:
            sources.append(results['metadatas'][0][i].get('source', 'Resume'))

    return "\n\n".join(context_parts), sources

def query_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.3,
                    "max_tokens": 200
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return "Sorry, I'm having trouble processing your request right now."
            
    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        return "Sorry, I'm having trouble connecting to the AI service."

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health():
    try:
        # Test Ollama connection
        response = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        ollama_status = "unhealthy"
    
    # Test ChromaDB
    try:
        collection.count()
        chroma_status = "healthy"
    except:
        chroma_status = "unhealthy"
    
    return {
        "status": "healthy" if ollama_status == "healthy" and chroma_status == "healthy" else "unhealthy",
        "ollama": ollama_status,
        "chromadb": chroma_status,
        "collection_count": collection.count() if chroma_status == "healthy" else 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Handle simple greetings directly
        greetings = ['hi', 'hello', 'hey', 'howdy']
        clean_question = request.question.lower().strip().rstrip('!?.')
        if clean_question in greetings:
            return ChatResponse(
                answer="Hello! I'm Nirwan Raj Nagpal's resume assistant. I can help you learn about his work experience, skills, education, projects, and certifications. What would you like to know?",
                sources=[]
            )

        # Get relevant context from resume - use more results for company queries and prioritize current employment
        top_k = 6 if any(word in request.question.lower() for word in ['current', 'currently', 'now', 'present']) else 4 if any(word in request.question.lower() for word in ['company', 'companies', 'work', 'employer', 'experience']) else 2
        context, sources = get_relevant_context(request.question, top_k)

        # Generate answer using Ollama with the retrieved context
        if not context or context == "No relevant context found.":
            return ChatResponse(
                answer="I don't have information about that in Nirwan's resume. Please ask about his experience, skills, education, or projects.",
                sources=[]
            )

        prompt = f"""You are a professional resume assistant. Answer ONLY resume and career questions using the provided resume data.

CRITICAL: If asked for jokes, stories, or casual conversation, respond EXACTLY: "I can only help with resume and career questions."

DO NOT create content using resume data for non-professional purposes.

Question: {request.question}
Resume Data: {context}

Professional Answer:"""

        # Debug logging
        logger.info(f"Query: {request.question}")
        logger.info(f"Context length: {len(context)}")
        logger.info(f"Number of sources: {len(sources)}")
        logger.info(f"Sources: {sources}")
        logger.info(f"Context preview: {context[:500]}...")

        answer = query_ollama(prompt)

        # Debug: Log the raw answer to see if it contains think tags
        logger.info(f"Raw answer length: {len(answer)}")
        if '<think>' in answer:
            logger.info("Answer contains <think> tags")
            # Log a snippet of the think content
            think_start = answer.find('<think>')
            think_end = answer.find('</think>') + 8
            if think_start != -1 and think_end != -1:
                think_snippet = answer[think_start:min(think_start + 100, think_end)]
                logger.info(f"Think content preview: {think_snippet}...")
        else:
            logger.info("Answer does NOT contain <think> tags")

        logger.info(f"Sending to frontend - Answer preview: {answer[:200]}...")

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)