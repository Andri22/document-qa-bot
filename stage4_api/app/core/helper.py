import os
from dotenv import load_dotenv

load_dotenv()


def get_chroma_path() -> str:
    return os.getenv("CHROMA_PATH", "./vectorstore")


def get_collection_name() -> str:
    return os.getenv("COLLECTION_NAME", "example_collection")


def get_chunk_settings() -> dict:
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
    }


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")
