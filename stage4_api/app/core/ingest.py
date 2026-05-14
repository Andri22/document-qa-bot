from app.logger import get_logger

# Load PDF
from langchain_community.document_loaders import PyPDFLoader

# Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embed and store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.helper import (
    get_chroma_path,
    get_chunk_settings,
    get_embedding_model,
)

logger = get_logger(__name__)


def ingest_document(file_path: str, session_id: str) -> int:
    try:
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        logger.info("Splitting into chunks")
        settings = get_chunk_settings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings["chunk_size"],  # chunk size (characters)
            chunk_overlap=settings["chunk_overlap"],  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model=get_embedding_model())

        vector_store = Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=embeddings,
            persist_directory=get_chroma_path(),
        )

        ids = [f"{file_path}_chunk_{i}" for i in range(len(all_splits))]

        logger.info(f"Storing {len(all_splits)} chunks in ChromaDB")
        vector_store.add_documents(documents=all_splits, ids=ids)

        logger.info(f"Ingestion complete: {len(all_splits)} chunks from {file_path}")
        return len(all_splits)
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        return 0
