

import json
import httpx 
from fastapi import APIRouter, HTTPException
from app.models.chat_models import ChatQuery, ChatResponse
from app.services.offline_rag_service import OfflineRAGService
from app.utils import logger


router = APIRouter(prefix="/api", tags=["Chatbot Metrics (Offline)"])


offline_rag_service = OfflineRAGService()


METRICS_API_URL = "http://127.0.0.1:8000/api/realtime_metrics_source"

@router.get("/realtime_metrics_source")
def realtime_metrics_source():
    """
    Simulates the external API endpoint. 
    It returns the current host system metrics (CPU, Memory, Disk) 
    in the requested legacy JSON structure, dynamically generated via psutil.
    """
    try:
        from app.data.system_metrics import get_realtime_data_in_target_format
        data = get_realtime_data_in_target_format()
        logger.info("Controller: Successfully fetched real-time system metrics.")
        return data
    except Exception as e:
        logger.error(f"Error fetching real-time metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get host system metrics via psutil.")


@router.post("/query/api", response_model=ChatResponse)
async def query_api_rag(query_data: ChatQuery):
    """
    Endpoint for Offline Model RAG Chatbot (Real-Time Metrics Source).
    Makes an HTTP call to /realtime_metrics_source to simulate external API consumption.
    """
    logger.info(f"Controller: Received API query: {query_data.query}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(METRICS_API_URL)
            response.raise_for_status()
            metrics_data = response.json()
            
        context_str = json.dumps(metrics_data)
        logger.info(f"Controller: Successfully retrieved context via HTTP from {METRICS_API_URL}.")
        
        response = await offline_rag_service.get_api_rag_response(
            query=query_data.query,
            context=context_str
        )
        
        return ChatResponse(response=response, source_type="API")
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Metrics API call failed with status {e.response.status_code}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Metrics source API unavailable or returned error: {e.response.status_code}")
    
    except Exception as e:
        logger.error(f"Unexpected error in query_api_rag controller: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during API RAG query.")