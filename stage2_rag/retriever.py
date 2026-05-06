import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_PATH"),
)

reterivier = vector_store.as_retriever(search_kwargs={"k": 3})

question = "What the title of this document"
results = reterivier.invoke(question)

for doc in results:
    print(doc.page_content)
    print("--")
