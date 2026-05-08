from fastapi import APIRouter
from pydantic import BaseModel
from fastapi import Request

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    engine = req.app.state.engine
    result = engine.query(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])
