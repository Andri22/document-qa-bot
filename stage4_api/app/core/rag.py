from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.logger import get_logger
from app.core.helper import (
    get_chroma_path,
    get_collection_name,
    get_embedding_model,
    get_llm_model,
)
from app.utils import format_docs, extract_sources

logger = get_logger(__name__)


class RAGEngine:
    def __init__(self):
        collection = get_collection_name()
        logger.info(f"RAG engine initializing for collection: {collection}")
        self.llm = ChatOpenAI(model=get_llm_model())

        self.embeddings = OpenAIEmbeddings(model=get_embedding_model())

        self.vector_store = Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=get_chroma_path(),
        )

        template = """You are a document assistant. Answer only from the 
        provided context. If the answer is not in the context, 
        say you don't know. Always cite your sources.

        Context: {context}
        Question: {question}"""

        self.prompt = ChatPromptTemplate.from_template(template)

        logger.info("RAG engine ready")

    def query(self, question: str, sources: list[str] | None = None) -> dict:
        logger.info(f"Processing question: {question}")
        retriever = self._build_retriever(sources)
        chain = self._build_chain(retriever)
        answer = chain.invoke(question).content
        source_docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(source_docs)} sources")
        cited_files = extract_sources(source_docs)
        logger.info("Query complete")
        return {"answer": answer, "sources": cited_files}

    def _build_retriever(self, sources: list[str] | None):
        search_kwargs = {"k": 3}
        if sources:
            if len(sources) == 1:
                search_kwargs["filter"] = {"source": sources[0]}
            else:
                search_kwargs["filter"] = {"source": {"$in": sources}}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def _build_chain(self, retriever):
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
        return chain
