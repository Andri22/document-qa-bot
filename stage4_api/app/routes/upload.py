from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ingest import ingest_document
import tempfile
from app.logger import get_logger
from app.utils import validate_pdf
from pydantic import BaseModel
import os

logger = get_logger(__name__)

router = APIRouter()

class FileUploadResult(BaseModel):
    filename: str
    chunks: int = 0
    status: str  # "ok" | "error"
    detail: str | None = None

class UploadResponse(BaseModel):
    results: list[FileUploadResult]
    total_chunks: int
    succeeded: int
    failed: int

async def _process_one_file(file: UploadFile) -> FileUploadResult:
    name = file.filename or "unknown"
    logger.info(f"Received file: {name}")
    if not validate_pdf(name):
        logger.warning(f"Invalid file type: {name}")
        return FileUploadResult(filename=name or "unknown", status="error", detail="Only PDF files are accepted")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        count = ingest_document(tmp_path, source_name=file.filename)
        if count == 0:
            return FileUploadResult(filename=name, status="error", detail="Ingestion failed")
        return FileUploadResult(filename=name, chunks=count, status="ok")
    except Exception as e:
        logger.error(f"Failed to process {name}: {e}")
        return FileUploadResult(filename=name, status="error", detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Upload a PDF document for ingestion into the knowledge base.
    Returns filename and number of chunks created.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Too many files")

    results : list[FileUploadResult] = []
    for file in files:
        results.append(await _process_one_file(file))
    
    total_chunks = sum(r.chunks for r in results if r.status == "ok")
    succeeded = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "error")
    return UploadResponse(results=results, total_chunks=total_chunks, succeeded=succeeded, failed=failed)

