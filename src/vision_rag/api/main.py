from fastapi import FastAPI
import uvicorn

from vision_rag.api.routes import router

app = FastAPI(
    title="Vision-Native RAG API",
    description="OCR-free document intelligence pipeline using Vision-Language Models.",
    version="0.1.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok"}

def start():
    """Entry point for the application."""
    uvicorn.run("vision_rag.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
