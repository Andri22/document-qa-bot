import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Initialize the model (it automatically uses the loaded OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_PATH"),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

template = """You are a document assistant. Answer only from the 
provided context. If the answer is not in the context, 
say you don't know. Always cite your sources.

Context: {context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


question = "What is the purpose of the Governance Agent?"
results = chain.invoke(question)
print("Answer:", results.content)
print("---")
print("Sources:")

source = retriever.invoke(question)
for doc in source:
    print(doc.metadata["source"])
