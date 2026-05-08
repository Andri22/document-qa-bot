from fastapi import APIRouter, UploadFile, File
from app.core.ingest import ingest_document
import tempfile

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. save uploaded file to temp path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    content = await file.read()
    tmp.write(content)
    tmp.close()
    # 2. call ingest_document(tmp_path)
    count = ingest_document(tmp.name)
    return {"filename": file.filename, "chunks": count}
