import os
from typing import List, Any
from PIL import Image
from google import genai

from vision_rag.core.generator import BaseVLMGenerator

class GeminiGenerator(BaseVLMGenerator):
    """Implementation of BaseVLMGenerator using Google's Gemini Models."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: str | None = None):
        """Initialize the Gemini Generator.
        
        Args:
            model_name: The Gemini model to use (e.g., 'gemini-1.5-pro' or 'gemini-1.5-flash').
            api_key: Google API Key. If None, it will try to read from the GOOGLE_API_KEY env variable.
        """
        self.model_name = model_name
        
        # Determine API key
        resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("Google API key must be provided or set as GOOGLE_API_KEY environment variable.")
            
        self.client = genai.Client(api_key=resolved_api_key)

    def generate(self, query: str, images: List[Image.Image]) -> str:
        """Generate a response using Gemini, passing the query and the retrieved images."""
        
        # Prepare the prompt contents
        # Gemini API accepts a list of text and images
        contents: List[Any] = [
            "You are a helpful assistant. Answer the user's question based strictly on the provided document pages (images).",
            f"Question: {query}",
            "Document Pages:"
        ]
        
        # Append all PIL images directly to the contents list
        contents.extend(images)
        
        print(f"Sending query and {len(images)} images to {self.model_name}...")
        
        # Generate the response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        
        return str(response.text)
