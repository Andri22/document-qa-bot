from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.logger import get_logger
from app.core.helper import (
    get_chroma_path,
    get_embedding_model,
    get_llm_model,
)
from app.utils import format_docs, extract_sources

logger = get_logger(__name__)


class RAGEngine:
    def __init__(self, session_id: str):
        logger.info(f"RAG engine initializing for session: {session_id}")
        # initialize llm, embeddings, chromadb, retriever, chain
        llm = ChatOpenAI(model=get_llm_model())

        embeddings = OpenAIEmbeddings(model=get_embedding_model())

        vector_store = Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=embeddings,
            persist_directory=get_chroma_path(),
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        template = """You are a document assistant. Answer only from the 
        provided context. If the answer is not in the context, 
        say you don't know. Always cite your sources.

        Context: {context}
        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        logger.info("RAG engine ready")
        self.chain = chain
        self.retriever = retriever

    def query(self, question: str) -> dict:
        logger.info(f"Processing question: {question}")
        answer = self.chain.invoke(question).content
        source_docs = self.retriever.invoke(question)
        logger.info(f"Retrieved {len(source_docs)} sources")
        sources = extract_sources(source_docs)
        logger.info("Query complete")
        return {"answer": answer, "sources": sources}
