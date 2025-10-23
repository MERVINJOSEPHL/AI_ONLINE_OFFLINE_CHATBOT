import os
import numpy as np
import faiss
from app.utils import logger
from PyPDF2 import PdfReader 
import pdfplumber
from sentence_transformers import SentenceTransformer
from app.services.gemini_rag_core import GeminiRAG
import re 
import json

import os
import numpy as np
import faiss
from app.utils import logger
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import re 
try:
    sent_tokenize("test sentence.")
except LookupError:
    nltk.download('punkt')


class VectorStore:
    """
    Implements document processing, embedding (using Sentence-Transformers), and FAISS indexing for RAG.
    Uses Semantic Chunking.
    """
    MODEL_NAME = 'all-MiniLM-L6-v2' 
    CHUNK_THRESHOLD = 0.55  
    CHUNK_SIZE_LIMIT = 512  

    # Initializes the VectorStore with full text, chunks it semantically, and builds the FAISS index.
    def __init__(self, full_text: str):
        self.logger = logger
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.documents = self._semantic_chunk(full_text)
        self.index = self._build_faiss_index()
        self.logger.info(f"VectorStore ready. Indexed {len(self.documents)} text chunks.")

    # Helper to clean up text by normalizing whitespace and removing backslashes.
    def _clean_text(self, text: str) -> str:
        """Helper to clean up text."""
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\\', '', text) 
        return text.strip()

    # Processes text using semantic chunking based on sentence similarity.
    def _semantic_chunk(self, text: str) -> list:
        """
        Processes text using semantic chunking based on sentence similarity.
        """
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return []

    
        sentences = sent_tokenize(cleaned_text)
        self.logger.info(f"Tokenized document into {len(sentences)} sentences for semantic analysis.")
        
    
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        
        similarities = []
        for i in range(len(sentences) - 1):
          
            vec1 = sentence_embeddings[i] / np.linalg.norm(sentence_embeddings[i])
            vec2 = sentence_embeddings[i+1] / np.linalg.norm(sentence_embeddings[i+1])
            similarity = np.dot(vec1, vec2)
            similarities.append(similarity)

        
        split_points = [i + 1 for i, sim in enumerate(similarities) if sim < self.CHUNK_THRESHOLD]
        
        chunks = []
        current_chunk = []
        start_index = 0

        for end_index in split_points:
          
            chunk_sentences = sentences[start_index:end_index]
            chunk_text = " ".join(chunk_sentences)
            
            if len(chunk_text) > self.CHUNK_SIZE_LIMIT:
                 self.logger.warning(f"Semantic chunk exceeded character limit: {len(chunk_text)} chars.")
            
            chunks.append(chunk_text)
            start_index = end_index

        final_chunk_text = " ".join(sentences[start_index:])
        if final_chunk_text:
            chunks.append(final_chunk_text)
      
        return [c.strip() for c in chunks if c.strip()][:100]

    # Encodes the text chunks and builds a FAISS IndexFlatIP index for fast similarity search.
    def _build_faiss_index(self) -> faiss.IndexFlatIP:
        try:
            self.logger.info(f"Building FAISS index for {len(self.documents)} documents...")
            if not self.documents:
                 raise ValueError("No documents to index.")
                 
            embeddings = self.model.encode(self.documents, convert_to_numpy=True)
            embeddings = np.array(embeddings).astype('float32')
            

            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            self.logger.info("FAISS Index build complete.")
            return index
        except Exception as e:
            self.logger.error(f"FATAL: Failed to build FAISS index: {e}", exc_info=True)
            raise RuntimeError("FAISS Index creation failed.")

    # Takes a query, converts it to an embedding, searches the FAISS index, and returns the top k relevant text chunks.
    def retrieve(self, query: str, k: int = 5) -> str:
        try:
            query_vector = self.model.encode([query], convert_to_numpy=True)
            query_vector = np.array(query_vector).astype('float32')
            faiss.normalize_L2(query_vector)
            
            
            D, I = self.index.search(query_vector, k) 
            
            retrieved_chunks = [self.documents[i] for i in I[0]]
            
            context = "\n---\n".join(retrieved_chunks)
            self.logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks from index.")
            return context
        except Exception as e:
            self.logger.error(f"Error during FAISS retrieval: {e}", exc_info=True)
            return "RAG Retrieval Failed: An error occurred during vector search."

class PDFRAGService:
    """Handles PDF extraction, RAG process, and communication with Gemini."""
    
    # Initializes the PDFRAGService, setting up the logger and the GeminiRAG core.
    def __init__(self):
        self.logger = logger
        self.gemini_rag = GeminiRAG(model_name='gemini-2.5-flash')
        self.pdf_path = None

    # Extracts text from the specified PDF and initializes the VectorStore with the content.
    def initialize_index(self, pdf_file_path: str):
        """
        NEW METHOD: Extracts text and initializes the FAISS vector store
        dynamically when the API is triggered.
        """
        try:
            self.pdf_path = pdf_file_path
            full_text = self._extract_text_from_pdf(pdf_file_path)
            self.vector_store = VectorStore(full_text)
            self.logger.info(f"PDF RAG Service successfully indexed: {os.path.basename(pdf_file_path)}")
        except Exception as e:
            self.logger.error(f"PDF RAG Service failed to initialize index: {e}")
            self.vector_store = None 
            raise


    # Extracts text content from the PDF file using the pdfplumber library.
    def _extract_text_from_pdf(self,pdf_file_path) -> str:
        """
        Extracts text from the PDF file using pdfplumber for better reliability
        with complex PDF structures.
        """
        pdf_text = []
        total_pages = 0
        try:
            self.logger.info(f"Starting PDF text extraction from: {pdf_file_path}")
            if not os.path.exists(pdf_file_path):
                raise FileNotFoundError(f"PDF file not found at: {pdf_file_path}")

            with pdfplumber.open(pdf_file_path) as pdf:
                total_pages = len(pdf.pages)

                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if page_text:
                        pdf_text.append(page_text)

            text = "\n\n".join(pdf_text) 

            self.logger.info(f"Text extraction complete. Total pages: {total_pages}")


            if not text.strip():
                self.logger.warning("Extracted text is empty or only whitespace even with pdfplumber. "
                                     "The PDF may be a scanned image.")

            print("Extracted Text (first 500 chars):", text[:500])  
            return text

        except Exception as e:
            self.logger.error(f"Error during PDF text extraction: {e}", exc_info=True)
            raise RuntimeError("Failed to extract content from PDF for RAG.")

    # Executes the full RAG pipeline: retrieves context from the vector store and generates a response using Gemini.
    async def get_pdf_rag_response(self, query: str) -> str:
        """Executes the full RAG pipeline: Retrieve -> Augment -> Generate."""
        try:
            if not self.vector_store:
                return "Error: RAG indexing failed during startup."
                
            self.logger.info(f"PDF RAG: Executing Retrieve phase for query: {query}")
            

            context = self.vector_store.retrieve(query)
            
            response = await self.gemini_rag.generate_rag_response(query, context, 'PDF')
            
            self.logger.info("PDF RAG response successfully generated via Gemini.")
            return response
            
        except RuntimeError as e:
            self.logger.error(f"Gemini error in PDF RAG: {e}", exc_info=True)
            return f"Apologies, the RAG system failed at the generation step: {e}"
        except Exception as e:
            self.logger.error(f"Error generating PDF RAG response: {e}", exc_info=True)
            return "Apologies, an unexpected error occurred while processing your PDF question."