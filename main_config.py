import os

class Config:
    """
    Configuration class for the ClearQuote RAG application.
    """

    # PDF Configuration
    PDF_SOURCE_DIRECTORY = "data"
    CHROMA_PERSIST_DIRECTORY = "docs/chroma"

    # Embedding Configuration
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 100

    # Chat Model Configuration
    CHAT_MODEL_NAME = "openai/gpt-oss-20b"
    MAX_TOKENS = 400
    TEMPERATURE = 0.3

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)


# Global configuration instance
config = Config()
