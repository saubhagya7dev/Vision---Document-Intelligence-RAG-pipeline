from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image

from vision_rag.retrieval.retriever import Retriever
from vision_rag.core.generator import BaseVLMGenerator
from vision_rag.ingest.document_loader import DocumentLoader

class RAGSynthesizer:
    """End-to-End Orchestrator for the Vision RAG Retrieval and Generation."""
    
    def __init__(self, retriever: Retriever, generator: BaseVLMGenerator, document_loader: DocumentLoader):
        """Initialize the synthesizer.
        
        Args:
            retriever: The retrieval component to find relevant document chunks/pages.
            generator: The VLM generator to synthesize the answer.
            document_loader: The document loader to read the original images from disk 
                             based on retrieval payload.
        """
        self.retriever = retriever
        self.generator = generator
        self.document_loader = document_loader
        
    def query(self, user_query: str, top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Perform a full RAG query.
        
        1. Retrieve relevant pages from the vector database.
        2. Load the actual images for those pages from the source.
        3. Pass the query and images to the VLM.
        
        Args:
            user_query: The text query.
            top_k: Number of pages to retrieve.
            
        Returns:
            A tuple of (Generated Text Response, List of Retrieved Source Metadata).
        """
        
        # 1. Retrieval
        retrieved_results = self.retriever.retrieve(query=user_query, top_k=top_k)
        
        if not retrieved_results:
            return "No relevant documents found.", []
            
        # 2. Extract Source Information and Load Images
        # In a production system, images might be stored in S3/GCS.
        # Here we extract the source path and page number from the payload and load it on the fly.
        
        images_to_pass = []
        source_metadata = []
        
        # To avoid loading the same PDF multiple times, we could cache the loaded PDFs
        pdf_cache: Dict[str, List[Image.Image]] = {}
        
        for result in retrieved_results:
            payload = result.get("payload", {})
            source_path = payload.get("source")
            page_num = payload.get("page_number")
            
            source_metadata.append({
                "score": result.get("score"),
                "source": source_path,
                "page": page_num
            })
            
            if source_path and page_num is not None:
                # Load PDF if not in cache
                if source_path not in pdf_cache:
                    pdf_cache[source_path] = self.document_loader.load_pdf(source_path)
                    
                # page_num is usually 1-indexed, so we subtract 1 for the 0-indexed list
                idx = page_num - 1
                if 0 <= idx < len(pdf_cache[source_path]):
                    images_to_pass.append(pdf_cache[source_path][idx])
                    
        # 3. Generation
        if not images_to_pass:
            return "Could not load the images for the retrieved documents.", source_metadata
            
        final_answer = self.generator.generate(query=user_query, images=images_to_pass)
        
        return final_answer, source_metadata
