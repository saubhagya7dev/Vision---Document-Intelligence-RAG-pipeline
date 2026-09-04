from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import shutil
from pathlib import Path
import tempfile
import traceback
from typing import List, Dict, Any

from vision_rag.config import settings
from vision_rag.ingest.document_loader import DocumentLoader
from vision_rag.models.colpali_embedder import ColPaliEmbeddingModel
from vision_rag.vector_stores.qdrant_store import QdrantVectorStore
from vision_rag.ingest.pipeline import IngestionPipeline
from vision_rag.models.gemini_generator import GeminiGenerator
from vision_rag.retrieval.retriever import Retriever
from vision_rag.generate.synthesizer import RAGSynthesizer

router = APIRouter()

# --- Dependency Injection / Singletons ---
# In a real app, these would be managed by FastAPI's DI system or lifespan events
# to prevent reloading models on every request.
_pipeline = None
_synthesizer = None

def get_components():
    global _pipeline, _synthesizer
    if _pipeline is None or _synthesizer is None:
        loader = DocumentLoader()
        embedder = ColPaliEmbeddingModel(model_name=settings.embedding_model_name)
        vdb = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            in_memory=settings.qdrant_in_memory,
        )
        generator = GeminiGenerator(model_name=settings.generation_model_name, api_key=settings.google_api_key)
        
        _pipeline = IngestionPipeline(loader, embedder, vdb)
        retriever = Retriever(embedder, vdb)
        _synthesizer = RAGSynthesizer(retriever, generator, loader)
        
    return _pipeline, _synthesizer


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document into the Vision RAG pipeline."""
    filename = file.filename or ""
    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    pipeline, _ = get_components()
    
    # Save uploaded file to persistent data/ directory so it can be re-read at query time
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        save_path = data_dir / filename
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Ingest the document
        pipeline.ingest_document(str(save_path), metadata={"original_filename": filename})
        
        return {"status": "success", "message": f"Successfully ingested {filename}"}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """Query the Vision RAG pipeline."""
    _, synthesizer = get_components()
    
    try:
        answer, sources = synthesizer.query(user_query=request.query, top_k=request.top_k)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
