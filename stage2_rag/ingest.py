import os
from dotenv import load_dotenv

# Load PDF
from langchain_community.document_loaders import PyPDFLoader

# Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embed and store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

file_path = "./data/example1.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_PATH"),
)

# Example: "example.pdf_chunk_0", "example.pdf_chunk_1" etc
ids = [f"{file_path}_chunk_{i}" for i in range(len(all_splits))]
vector_store.add_documents(documents=all_splits, ids=ids)

print(f"✅ Ingested {len(all_splits)} chunks from {file_path} into ChromaDB")
