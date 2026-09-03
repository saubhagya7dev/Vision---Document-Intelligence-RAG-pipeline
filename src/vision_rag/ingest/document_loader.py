import os
from pathlib import Path
from typing import List
from PIL import Image
from pdf2image import convert_from_path

class DocumentLoader:
    """Handles the ingestion and conversion of documents (PDFs) into images."""
    
    def __init__(self, dpi: int = 300, fmt: str = "jpeg"):
        """Initialize the document loader.
        
        Args:
            dpi: Resolution for PDF to image conversion. Higher is better for VLMs.
            fmt: Output image format.
        """
        self.dpi = dpi
        self.fmt = fmt

    def load_pdf(self, file_path: str | Path) -> List[Image.Image]:
        """Convert a PDF document into a list of PIL Images (one per page).
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of PIL Image objects.
            
        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at: {path}")
            
        # Convert PDF pages to images
        # Note: poppler must be installed on the system for pdf2image to work
        images = convert_from_path(
            pdf_path=str(path),
            dpi=self.dpi,
            fmt=self.fmt,
            thread_count=4  # Speed up conversion using multiple threads
        )
        
        return images
