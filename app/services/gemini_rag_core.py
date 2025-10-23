import os
from app.utils import logger
from google import genai 
from google.genai.errors import APIError
from google.genai.types import Content, Part
from dotenv import load_dotenv
import json
load_dotenv() 

class GeminiRAG:
    """
    Handles all interactions with the Gemini model for RAG.
    This class is the GEN-module, taking context and generating the answer via the live API.
    """
    # Initializes the GeminiRAG core, sets up the model name, the constrained system prompt, 
    # and initializes the live Gemini client using the GEMINI_API_KEY environment variable.
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.logger = logger
        self.model_name = model_name
        
        self.system_prompt = (
            "Act Like a sales agent and respond to the answer query."
            "You are a highly constrained system. Your only goal is to answer the user's question "
            "If the relavent information is present in the CONTEXT. instead of saying 'no' directly. if no information is present then say no "
            
            "If the specific requested information (e.g., support for a product or a version) is NOT explicitly mentioned in the context, "
            "the answer is **'No'**. **NEVER** use phrases like 'not listed,' 'not available,' or 'I apologize.' "
        )
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("GEMINI_API_KEY is not set. Client initialization failed.")
            raise ValueError("GEMINI_API_KEY environment variable is required for live invocation.")
            
        self.client = genai.Client(api_key=api_key)
        self.logger.info(f"GeminiRAG core initialized for model: {self.model_name}. Ready for live API calls.")
        
    # Generates a response by combining the system prompt, the retrieved context, and the user query, 
    # then calls the Gemini API to produce the final RAG answer.
    async def generate_rag_response(self, query: str, context: str, source_type: str) -> str:
        """
        Generates a response by combining the prompt, context, and calling the LLM.
        """
        try:
            self.logger.info(f"GeminiRAG: Preparing prompt for {source_type} query.")

            full_prompt = (
                f"{self.system_prompt}\n\n"
                f"**CONTEXT ({source_type}):**\n"
                f"---START CONTEXT---\n{context}\n---END CONTEXT---\n\n"
                f"**USER QUESTION:** {query}"
            )

            contents = [
                Content(role="user", parts=[Part.from_text(text=full_prompt)])
            ]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )

            if not response.candidates:
                return "The Gemini model returned an empty response. Context may have been insufficient or unsafe."

            self.logger.info("Gemini API call succeeded. Response received.")
            return response.text

        except APIError as e: 
            self.logger.error(f"Gemini API Error: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API invocation failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected Error during generation: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate response using the Gemini model: {e}")