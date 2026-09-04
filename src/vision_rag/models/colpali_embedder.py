import torch
from typing import List, Any, cast
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

from vision_rag.core.embeddings import BaseEmbeddingModel


class ColPaliEmbeddingModel(BaseEmbeddingModel):
    """Implementation of BaseEmbeddingModel using ColPali.

    ColPali converts images into multi-vector patch representations, allowing for
    late-interaction retrieval which is highly effective for visual documents.
    Uses the dedicated `colpali-engine` library for correct model loading.
    """

    def __init__(self, model_name: str = "vidore/colpali-v1.3", device: str | None = None):
        """Initialize the ColPali model.

        Args:
            model_name: HuggingFace model name for ColPali (e.g. 'vidore/colpali-v1.3').
            device: Device to load the model on ('cuda', 'mps', 'cpu'). Auto-detected if None.
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

        # Use float16 on GPU for efficiency; float32 on CPU
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        print(f"Loading ColPali model '{model_name}' on {self.device} ({dtype})...")
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        print("ColPali model loaded successfully.")

    def encode_images(self, images: List[Image.Image]) -> List[Any]:
        """Encode a list of PIL Images into single-vector embeddings via mean pooling.

        ColPali natively produces multi-vector patch embeddings (num_patches × 128).
        We mean-pool across the patch dimension to get one 128-dim vector per image,
        making it compatible with standard single-vector Qdrant collections.
        """
        if not images:
            return []

        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            # Shape: (batch_size, num_patches, 128)
            embeddings = self.model(**batch)
            # Mean-pool across patches → (batch_size, 128)
            embeddings = embeddings.mean(dim=1)
            return cast(List[Any], embeddings.cpu().tolist())

    def encode_queries(self, queries: List[str]) -> List[Any]:
        """Encode a list of text queries into single-vector embeddings via mean pooling."""
        if not queries:
            return []

        with torch.no_grad():
            batch = self.processor.process_queries(queries).to(self.device)
            # Shape: (batch_size, num_tokens, 128)
            embeddings = self.model(**batch)
            # Mean-pool across tokens → (batch_size, 128)
            embeddings = embeddings.mean(dim=1)
            return cast(List[Any], embeddings.cpu().tolist())
