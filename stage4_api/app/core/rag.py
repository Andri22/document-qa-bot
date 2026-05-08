import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class RAGEngine:
    def __init__(self):
        # initialize llm, embeddings, chromadb, retriever, chain
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
        self.chain = chain
        self.retriever = retriever

    def query(self, question: str) -> dict:
        # run chain, get answer and sources
        # get answer
        answer = self.chain.invoke(question).content

        # get sources separately
        source_docs = self.retriever.invoke(question)
        sources = list(
            set([doc.metadata.get("source", "unknown") for doc in source_docs])
        )

        # return {"answer": answer, "sources": sources}
        return {"answer": answer, "sources": sources}
