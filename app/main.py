from fastapi import FastAPI
from app.controllers import chat_router, api_router
from app.utils import logger

# Initialize FastAPI App
app = FastAPI(
    title="RAG Chatbot Assignment API",
    description="A dual RAG application for PDF content (GPT-based) and Live API data (Offline Model).",
    version="1.0.0"
)

# Include the API router
app.include_router(chat_router.router)
app.include_router(api_router.router)

@app.on_event("startup")
async def startup_event():
    """Application startup logging."""
    logger.info("RAG Chatbot API is starting up...")

@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "RAG Chatbot API is running. Access /docs for endpoints."}

# To run the application:
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000