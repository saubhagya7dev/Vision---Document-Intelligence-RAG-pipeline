from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil
from pathlib import Path
import tempfile
import traceback
import io
import os
from typing import List, Dict, Any
from datetime import datetime

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
_vdb = None
_loader = None

def get_components():
    global _pipeline, _synthesizer, _vdb, _loader
    if _pipeline is None or _synthesizer is None:
        _loader = DocumentLoader()
        embedder = ColPaliEmbeddingModel(model_name=settings.embedding_model_name)
        _vdb = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            in_memory=settings.qdrant_in_memory,
        )
        generator = GeminiGenerator(model_name=settings.generation_model_name, api_key=settings.google_api_key)
        
        _pipeline = IngestionPipeline(_loader, embedder, _vdb)
        retriever = Retriever(embedder, _vdb)
        _synthesizer = RAGSynthesizer(retriever, generator, _loader)
        
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


# --- New Endpoints for the UI ---

@router.get("/documents")
async def list_documents():
    """List all uploaded PDF documents in the data/ directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    documents = []
    for pdf_file in sorted(data_dir.glob("*.pdf")):
        stat = pdf_file.stat()
        # Count pages by loading via document loader (cached if possible)
        try:
            loader = DocumentLoader()
            images = loader.load_pdf(pdf_file)
            page_count = len(images)
        except Exception:
            page_count = 0
        
        documents.append({
            "filename": pdf_file.name,
            "size_bytes": stat.st_size,
            "page_count": page_count,
            "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    return documents


@router.get("/documents/{filename}/pages/{page_num}")
async def get_page_image(filename: str, page_num: int):
    """Serve a specific page of a PDF as a JPEG image."""
    file_path = Path("data") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    
    try:
        loader = DocumentLoader()
        images = loader.load_pdf(file_path)
        
        idx = page_num - 1  # page_num is 1-indexed
        if idx < 0 or idx >= len(images):
            raise HTTPException(status_code=404, detail=f"Page {page_num} not found. Document has {len(images)} pages.")
        
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        images[idx].save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get dashboard statistics."""
    # Count documents in data/ directory
    data_dir = Path("data")
    doc_count = len(list(data_dir.glob("*.pdf"))) if data_dir.exists() else 0
    
    # Get vector store info if initialized
    collection_info = {"vectors_count": 0, "points_count": 0, "status": "not_initialized"}
    if _vdb is not None:
        collection_info = _vdb.get_collection_info()
    
    return {
        "total_documents": doc_count,
        "total_vectors": collection_info.get("points_count", 0),
        "collection_status": collection_info.get("status", "unknown"),
    }


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its associated vectors."""
    file_path = Path("data") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    
    try:
        # Delete vectors from Qdrant that match this source file
        if _vdb is not None:
            _vdb.delete_by_payload_filter("source", str(file_path))
        
        # Delete the PDF file
        file_path.unlink()
        
        return {"status": "success", "message": f"Deleted {filename} and its vectors."}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
