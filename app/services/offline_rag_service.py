# app/services/offline_rag_service.py

import os
from app.utils import logger
import json
from dotenv import load_dotenv
import asyncio

# --- HUGGING FACE DOWNLOAD DEPENDENCY ---
# Install this with: pip install huggingface-hub
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    logger.warning("huggingface-hub not installed. Cannot auto-download model.")
    HF_HUB_AVAILABLE = False


load_dotenv() 

# --- OFFLINE LLM DEPENDENCIES ---
# Install this with: pip install llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python not installed. Using placeholder inference.")
    LLAMA_CPP_AVAILABLE = False


class OfflineRAGService:
    """
    Handles the RAG process and invocation for the Offline LLM (Llama-CPP).
    The model is downloaded from Hugging Face directly to the models folder.
    """
    # --- Hugging Face Model Constants ---
    REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Define the local directory for the model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\services', '')
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # The expected final path after download
    MODEL_PATH = os.path.join(MODEL_DIR, FILENAME)
    
    # Placeholder name used for logging
    MODEL_NAME = FILENAME 

    def __init__(self):
        self.logger = logger
        self.llm = None
        # We no longer track the path for deletion since the user now wants to keep it.
        # self._downloaded_model_path = None 

        local_model_path = self.MODEL_PATH

        # 1. DOWNLOAD THE MODEL DIRECTLY TO THE TARGET FOLDER
        if HF_HUB_AVAILABLE:
            self.logger.info(f"Checking for and downloading model '{self.FILENAME}' to {self.MODEL_DIR}...")
            try:
                # hf_hub_download downloads the file (or uses the cache if the file is identical)
                # and places it directly in the specified local_dir.
                local_model_path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=self.FILENAME,
                    local_dir=self.MODEL_DIR, # Target directory
                    local_dir_use_symlinks=False # Forces the actual file to be moved/copied here
                )
                self.logger.info(f"Model file confirmed/downloaded at: {local_model_path}")
            except Exception as e:
                self.logger.error(f"FATAL: Failed to download model from Hugging Face: {e}", exc_info=True)
                local_model_path = None
        
        # 2. Initialize Llama-CPP with the model path
        if LLAMA_CPP_AVAILABLE and local_model_path and os.path.exists(local_model_path):
            try:
                # Initialize the Llama model for local inference
                self.llm = Llama(
                    model_path=local_model_path,
                    n_ctx=2048, 
                    n_threads=os.cpu_count() // 2 or 1,
                    n_gpu_layers=-1 
                )
                self.logger.info(f"OfflineRAGService initialized for local model: {local_model_path}. Ready for RAG.")
                
            except Exception as e:
                self.logger.error(f"FATAL: Failed to initialize Llama model from {local_model_path}: {e}", exc_info=True)
                self.llm = None
        else:
            self.logger.error(f"OfflineRAGService: Initialization failed. Model not found or dependencies missing.")
    
    # --- Cleanup Methods Removed ---
    # The __del__ and cleanup_model methods were removed since you want to KEEP the model.
    
    # Generates a response using the offline model, constrained by the API data context.
    async def get_api_rag_response(self, query: str, context: str) -> str:
        """
        Executes the full RAG pipeline using the offline model.
        """
        if not self.llm:
            return "Error: Local LLM is not loaded. Check model path and llama-cpp-python installation."

        system_prompt = (
            "You are a system metrics monitor. Your task is to provide a concise answer. "
            "Use the CONTEXT JSON to find the required metric. **DO NOT** repeat or re-paste the context JSON in your response. "
            "If the metric is present, state the value, unit, and priority status. Format the numerical value and unit in bold. "
            "If the specific requested metric is not explicitly present in the JSON, the answer is **'No'**."
        )

        full_prompt = (
            f"**CONTEXT (System Metrics JSON):**\n"
            f"---START CONTEXT---\n{context}\n---END CONTEXT---\n\n"
            f"**USER QUESTION:** {query}"
        )
        
        self.logger.info("Offline RAG: Preparing prompt for metrics query.")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            
            generated_text = response['choices'][0]['message']['content'].strip()
            self.logger.info("Offline LLM RAG generation succeeded.")
            return generated_text

        except Exception as e:
            self.logger.error(f"Offline model (Llama-CPP) failed to process context: {e}", exc_info=True)
            return f"RAG Generation Failed: Local LLM invocation failed. Error: {e}"
