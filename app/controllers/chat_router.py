import os
from fastapi import APIRouter, HTTPException, Depends, FastAPI, File, UploadFile
from app.models.chat_models import ChatQuery, ChatResponse
from app.services.pdf_rag_service import PDFRAGService
from app.utils import logger
from starlette.background import BackgroundTasks
import shutil

# --- Initialization of Dependencies (Global Singleton/Factory Pattern) ---
# In a robust MVC, these would be injected via FastAPI's Depends, but initialized here for simplicity

router = APIRouter(prefix="/chat", tags=["Chatbot RAG"])


# --- 1. Global State ---
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
pdf_rag_service = PDFRAGService()  # Initialize the core service empty


# --- 2. New API Endpoint for Upload and Indexing ---

# Handles the upload of a PDF file, saves it to disk, and then initializes the 
# PDFRAGService's index with the file's content.
@router.post("/upload/pdf")
async def upload_and_index_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"status": "error", "message": "Only PDF files are accepted."}
    
    # Save the file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger indexing (Refactored call)
        pdf_rag_service.initialize_index(pdf_file_path=file_path)
        
        return {
            "status": "success", 
            "message": f"PDF '{file.filename}' uploaded and RAG index successfully built."
        }
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}")
        return {"status": "error", "message": f"Failed to process and index PDF: {e}"}

# Receives a chat query and processes it against the currently indexed PDF document
# by calling the PDFRAGService to retrieve context and generate a response.
@router.post("/query/pdf", response_model=ChatResponse)
async def query_pdf_rag(query_data: ChatQuery):
    """
    Endpoint for GPT-based RAG Chatbot (PDF Source).
    """
    try:
        if not pdf_rag_service or not pdf_rag_service.vector_store:
            raise HTTPException(status_code=503, detail="PDF RAG service is offline or failed initialization.")

        logger.info(f"Controller: Received PDF query: {query_data.query}")
        
        response = await pdf_rag_service.get_pdf_rag_response(query_data.query)
        
        return ChatResponse(response=response, source_type="PDF")
        
    except HTTPException:
        # Re-raise explicit HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query_pdf_rag controller: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during PDF RAG query.")

