from fastapi import FastAPI
from app.core.rag import RAGEngine
from app.routes import chat, upload

app = FastAPI(title="Document Q&A Bot API")
app.state.engine = RAGEngine()
# register routes here
app.include_router(chat.router)
app.include_router(upload.router)


@app.get("/health")
def health():
    return {"status": "ok"}
