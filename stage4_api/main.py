from fastapi import FastAPI
from app.routes import chat, upload
from app.logger import get_logger
import os

logger = get_logger(__name__)

app = FastAPI(title="Document Q&A Bot API")

# register routes here
app.include_router(chat.router)
app.include_router(upload.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    if not os.getenv("CHROMA_PATH"):
        raise RuntimeError("CHROMA_PATH is required")
    logger.info("Environment validated successfully")

    app.state.engine = None
    logger.info("RAG engine slot initialized")
