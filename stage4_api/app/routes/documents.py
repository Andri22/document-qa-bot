from fastapi import APIRouter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.helper import (
    get_chroma_path,
    get_collection_name,
    get_embedding_model,
)

router = APIRouter()


@router.get("/documents")
def list_documents():
    collection = get_collection_name()
    embeddings = OpenAIEmbeddings(model=get_embedding_model())
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=get_chroma_path(),
    )
    data = vector_store.get()
    sources = sorted(set(m.get("source", "unknown") for m in data.get("metadatas", [])))
    return {"documents": sources}
