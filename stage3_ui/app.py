import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import tempfile
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


# Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Page config — always first line
st.set_page_config(page_title="Document Q&A Bot", layout="wide")


# format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Cache expensive resources
@st.cache_resource
def load_resources():
    # everything needed to initialize retriever and llm
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

    rag_prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
    )

    return retriever, llm, chain


retriever, llm, chain = load_resources()

st.title("📄 Document Q&A Bot")


def ingest_document(file_path):
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
    ids = [f"{file_path}_chunk_{i}" for i in range(len(all_splits))]

    vector_store.add_documents(documents=all_splits, ids=ids)

    return len(all_splits)


with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Ingest Document"):
        # save uploaded file to data/ folder
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # run ingestion
        count = ingest_document(tmp_path)

        # show success message
        st.success(f"Ingested {count} chunks successfully")
        pass

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if prompt := st.chat_input("Ask a question about your document"):
    with st.chat_message("user"):  # ← correct
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chain.invoke(prompt)
    answer = response.content

    with st.chat_message("assistant"):  # ← correct
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
