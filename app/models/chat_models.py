from pydantic import BaseModel, Field

class ChatQuery(BaseModel):
    """Model for user query input."""
    query: str = Field(..., description="The user's question to the chatbot.")

class ChatResponse(BaseModel):
    """Model for the chatbot response output."""
    response: str = Field(..., description="The RAG-generated answer.")
    source_type: str = Field(..., description="Source of the answer: 'PDF' or 'API'.")