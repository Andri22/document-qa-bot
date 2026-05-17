import streamlit as st
import requests

# docker service name
FASTAPI_URL = "http://fastapi:8000"

# Page config — always first line
st.set_page_config(page_title="Document Q&A Bot", layout="wide")
st.title("📄 Document Q&A Bot - AI Powered")

with st.sidebar:
    st.header("Upload Document")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True, key="uploaded_files")

    if uploaded_files and st.button("Ingest Document"):
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
        response = requests.post(f"{FASTAPI_URL}/upload", files=files)
        if response.status_code == 200:
            result = response.json()
            if result.get("succeeded", 0) > 0:
                st.success(
                    f"Ingested {result['total_chunks']} chunks from "
                    f"{result['succeeded']} file(s)"
                )
                if "document_list" not in st.session_state:
                    st.session_state.document_list = []

                for item in result["results"]:
                    if item["status"] == "ok" and item["filename"] not in st.session_state.document_list:
                        st.session_state.document_list.append(item["filename"])
            else:
                st.error("No files were ingested.")
            if result.get("failed"):
                for item in result["results"]:
                    if item["status"] == "error":
                        st.error(f"{item['filename']}: {item['detail']}")
        else:
            error = response.json()
            st.error(f"Upload failed: {error['detail']}")
    options = ["All documents"] + st.session_state.get("document_list", [])
    selected = st.selectbox("Answer from:", options)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    payload = {"question": prompt}
    if selected != "All documents":
        payload["sources"] = [selected]

    response = requests.post(f"{FASTAPI_URL}/chat", json=payload)

    if response.status_code == 200:
        result = response.json()
        answer = result["answer"]
    else:
        answer = "Service temporarily unavailable. Please try again."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
