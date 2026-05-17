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
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter

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

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """YYou are a document assistant. 
                    Answer from the provided context when available. 
                    Use the conversation history for follow-up questions. 
                    If the answer is in neither, say you don't know. 
                    Always cite your sources.
                    
                    Context: {context}
                    """,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        logger.info("RAG engine ready")

    def query(
        self,
        question: str,
        history: ChatMessageHistory,
        sources: list[str] | None = None,
    ) -> dict:
        logger.info(f"Processing question: {question}")
        retriever = self._build_retriever(sources)
        chain = self._build_chain(retriever)
        answer = chain.invoke(
            {
                "question": question,
                "history": history.messages,  # Pass messages here
            }
        ).content
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
            RunnablePassthrough.assign(
                # Extract "question" for the retriever, but keep "history" and "question" in the final dict
                context=itemgetter("question") | retriever | format_docs
            )
            | self.prompt
            | self.llm
        )
        return chain
