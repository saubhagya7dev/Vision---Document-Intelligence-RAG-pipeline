from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[Any], payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        """Upsert vectors and their metadata into the store.
        
        Args:
            ids: List of unique identifiers.
            vectors: List of embeddings.
            payloads: Optional list of metadata dictionaries.
        """
        pass
        
    @abstractmethod
    def search(self, query_vector: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for the most similar vectors.
        
        Args:
            query_vector: The query embedding.
            limit: Number of results to return.
            
        Returns:
            List of dictionaries containing matching documents (id, score, payload).
        """
        pass
