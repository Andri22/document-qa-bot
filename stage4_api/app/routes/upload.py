from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ingest import ingest_document
import tempfile
from app.logger import get_logger
from app.utils import validate_pdf

logger = get_logger(__name__)

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        if not validate_pdf(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return {"error": "Only PDF files are accepted"}
        # 1. save uploaded file to temp path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # 2. call ingest_document(tmp_path)
        count = ingest_document(tmp.name)
        logger.info(f"Ingestion complete: {count} chunks")
        return {"filename": file.filename, "chunks": count}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
