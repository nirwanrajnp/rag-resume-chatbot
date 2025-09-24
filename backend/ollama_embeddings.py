import requests
import os
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    def __init__(self, model_name: str = "bge-m3:latest", ollama_url: str = None):
        self.model_name = model_name
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text(s) using Ollama bge-m3 model

        Args:
            texts: Single text string or list of text strings

        Returns:
            List of embedding vectors (list of floats for each text)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    embeddings.append(embedding)
                    logger.debug(f"Generated embedding for text (length: {len(text)}, embedding dim: {len(embedding)})")
                else:
                    logger.error(f"Ollama embedding API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API error: {response.status_code}")

            except requests.RequestException as e:
                logger.error(f"Request error while generating embeddings: {e}")
                raise Exception(f"Ollama connection error: {str(e)}")

        logger.info(f"Generated {len(embeddings)} embeddings using {self.model_name}")
        return embeddings

    def test_connection(self) -> bool:
        """Test if Ollama API and bge-m3 model are available"""
        try:
            # Test with a simple text
            result = self.encode(["test"])
            return len(result) > 0 and len(result[0]) > 0
        except Exception as e:
            logger.error(f"Ollama embedding test failed: {e}")
            return False

def get_ollama_embeddings(model_name: str = "bge-m3:latest") -> OllamaEmbeddings:
    """
    Factory function to create OllamaEmbeddings instance
    """
    return OllamaEmbeddings(model_name=model_name)

# Test the connection when module is imported
if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)

    embedder = get_ollama_embeddings()

    if embedder.test_connection():
        print("Ollama bge-m3 embeddings working correctly")

        # Test with sample texts
        test_texts = [
            "What companies has the person worked for?",
            "Technical skills and programming languages",
            "Current employment and job history"
        ]

        embeddings = embedder.encode(test_texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {len(embeddings[0])}")

    else:
        print("Ollama bge-m3 embeddings not working")
        print("Make sure Ollama is running and bge-m3:latest model is available")