from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi import Request
from app.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Ask a question about uploaded documents.
    Returns answer and source citations.
    """
    try:
        logger.info(f"Received question: {request.question}")
        engine = req.app.state.engine
        result = engine.query(request.question)
        logger.info("Query complete")
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
