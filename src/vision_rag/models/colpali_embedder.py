import torch
from typing import List, Any
from PIL import Image
from transformers import AutoProcessor, AutoModel

from vision_rag.core.embeddings import BaseEmbeddingModel

class ColPaliEmbeddingModel(BaseEmbeddingModel):
    """Implementation of BaseEmbeddingModel using ColPali.
    
    ColPali converts images into multi-vector representations, allowing for
    late interaction retrieval which is highly effective for visual documents.
    """
    
    def __init__(self, model_name: str = "vidore/colpali-v1.2", device: str | None = None):
        """Initialize the ColPali model.
        
        Args:
            model_name: HuggingFace model name for ColPali.
            device: Device to load the model on ('cuda', 'mps', 'cpu').
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading ColPali model '{model_name}' on {self.device}...")
        # Note: In a real production setup, we might load with bfloat16 or 8-bit depending on hardware
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print("ColPali model loaded successfully.")

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> List[Any]:
        """Encode a list of PIL Images into multi-vector embeddings."""
        if not images:
            return []
            
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # ColPali outputs multi-vector embeddings (batch_size, num_patches, hidden_dim)
        embeddings = self.model(**inputs).last_hidden_state
        return embeddings.cpu().tolist()
        
    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> List[Any]:
        """Encode a list of text queries into multi-vector embeddings."""
        if not queries:
            return []
            
        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        embeddings = self.model(**inputs).last_hidden_state
        return embeddings.cpu().tolist()
