import os
import config as cfg

print("Config file loaded from:")
print(cfg.__file__)
class Config:
    """
    Configuration class to manage file paths, embeddings, and model settings.
    """

    # PDF Configuration
    PDF_SOURCE_DIRECTORY = "data"
    CHROMA_PERSIST_DIRECTORY = "docs/chroma"

    # Embedding Configuration
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 100

    # Chat Model Configuration
    CHAT_MODEL_NAME = "llama-3.1-8b-instant"
    MAX_TOKENS = 400
    TEMPERATURE = 0.3

    # Environment Variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Global instance
config = Config()
