import uuid
from pathlib import Path
from typing import Dict, Any

from vision_rag.ingest.document_loader import DocumentLoader
from vision_rag.core.embeddings import BaseEmbeddingModel
from vision_rag.core.vector_store import BaseVectorStore

class IngestionPipeline:
    """Orchestrates the process of loading, embedding, and storing documents."""
    
    def __init__(
        self, 
        loader: DocumentLoader, 
        embedder: BaseEmbeddingModel, 
        vector_store: BaseVectorStore
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            loader: Component to load PDFs into images.
            embedder: Component to generate embeddings from images.
            vector_store: Component to store embeddings and metadata.
        """
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest_document(self, file_path: str | Path, metadata: Dict[str, Any] | None = None) -> None:
        """Process a document and store its embeddings.
        
        Args:
            file_path: Path to the PDF document.
            metadata: Optional base metadata to attach to each page.
        """
        path = str(file_path)
        print(f"Starting ingestion for: {path}")
        
        # 1. Load document (convert pages to images)
        print("Converting PDF pages to images...")
        images = self.loader.load_pdf(path)
        print(f"Converted {len(images)} pages.")
        
        # 2. Generate embeddings
        print("Generating embeddings for pages...")
        embeddings = self.embedder.encode_images(images)
        
        # 3. Prepare payloads and IDs
        if metadata is None:
            metadata = {}
            
        ids = []
        payloads = []
        
        for i, _ in enumerate(images):
            page_num = i + 1
            # Generate a deterministic or random UUID for the page
            page_id = str(uuid.uuid4())
            ids.append(page_id)
            
            payload = metadata.copy()
            payload.update({
                "source": path,
                "page_number": page_num
            })
            payloads.append(payload)
            
        # 4. Upsert to Vector Store
        print(f"Upserting {len(embeddings)} vectors into the database...")
        self.vector_store.upsert(ids=ids, vectors=embeddings, payloads=payloads)
        print("Ingestion complete.")
