import streamlit as st
import requests

# docker service name
FASTAPI_URL = "http://fastapi:8000"

# Page config — always first line
st.set_page_config(page_title="Document Q&A Bot", layout="wide")
st.title("📄 Document Q&A Bot - AI Powered")


with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Ingest Document"):
        # send file to FastAPI
        response = requests.post(
            f"{FASTAPI_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Ingested {result['chunks']} chunks successfully")
        else:
            error = response.json()
            st.error(f"Upload failed: {error['detail']}")

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

    # send question to FastAPI
    response = requests.post(f"{FASTAPI_URL}/chat", json={"question": prompt})

    if response.status_code == 200:
        result = response.json()
        answer = result["answer"]
    else:
        answer = "Service temporarily unavailable. Please try again."

    with st.chat_message("assistant"):  # ← correct
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
