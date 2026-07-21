import os
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pdf_content(pdf_path: str):
    """Load a single PDF and return pages."""
    logger.info(f"Loading PDF document: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading {pdf_path}: {e}")
        return []


def load_pdfs_from_directory(pdf_directory: str):
    """Loads all PDFs from a given directory (including subfolders)."""
    if not os.path.exists(pdf_directory):
        logger.warning(f"PDF directory '{pdf_directory}' not found.")
        return []

    all_docs = []
    for root, _, files in os.walk(pdf_directory):
        for filename in files:
            if filename.endswith(".pdf"):
                filepath = os.path.join(root, filename)
                all_docs.extend(load_pdf_content(filepath))

    if all_docs:
        logger.info(f"Loaded {len(all_docs)} pages from '{pdf_directory}'.")
    else:
        logger.warning(f"No PDF documents found in '{pdf_directory}'.")
    return all_docs


def ingest_documents(pdf_paths: list, persist_directory: str = "docs/chroma"):
    """Ingests a list of PDF files or directories into ChromaDB."""
    logger.info("--- Starting PDF ingestion ---")

    all_docs = []
    for path in pdf_paths:
        if os.path.isdir(path):
            all_docs.extend(load_pdfs_from_directory(path))
        elif os.path.isfile(path) and path.endswith(".pdf"):
            all_docs.extend(load_pdf_content(path))
        else:
            logger.warning(f"Skipping invalid path: {path}")

    if not all_docs:
        logger.error("No documents found. Exiting ingestion.")
        return

    logger.info(f"Total loaded pages: {len(all_docs)}")

    # Split into chunks
    logger.info(
        f"Splitting documents (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})..."
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunked_docs = splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunked_docs)} chunks.")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # Vector DB
    logger.info(f"Creating/Updating Chroma DB at '{persist_directory}'...")
    os.makedirs(persist_directory, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    logger.info(f"✅ Ingestion complete! {len(chunked_docs)} chunks stored in ChromaDB.")


if __name__ == "__main__":
    ingest_documents(
        pdf_paths=[
            "data/8.4 Connector and Integration Guides",
            "data/identityiq-doc-8.4"
        ],
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )