from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_community.chat_message_histories import SQLChatMessageHistory


from app.core.rag import RAGEngine
from app.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    session_id: str
    sources: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///chat_history.db"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Ask a question about uploaded documents.
    Returns answer and source citations.
    """
    try:
        logger.info(f"Received question: {request.question}")
        if req.app.state.engine is None:
            req.app.state.engine = RAGEngine()
        history = get_session_history(request.session_id)
        result = req.app.state.engine.query(
            request.question, history=history, sources=request.sources
        )
        history.add_user_message(request.question)
        history.add_ai_message(result["answer"])
        logger.info("Query complete")
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
