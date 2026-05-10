"""
FastAPI server for RAG-powered document Q&A.
Endpoints: /upload, /ask, /sessions/<id>, /health, /
Supported file types: PDF, TXT, CSV
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict
from pathlib import Path
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Q&A",
    description="Document question-answering using RAG and Google Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10MB default
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}

# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipeline()
    logger.info("RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    raise


# ============================================================================
# Request/Response Models
# ============================================================================

class AskRequest(BaseModel):
    """Request model for asking a question."""
    session_id: str
    question: str


class UploadResponse(BaseModel):
    """Response model for file upload."""
    session_id: str
    file_name: str
    chunk_count: int
    message: str


class AskResponse(BaseModel):
    """Response model for Q&A."""
    answer: str
    sources: list


class SessionMetadata(BaseModel):
    """Session metadata response."""
    session_id: str
    file_name: str
    chunk_count: int
    created_at: str
    file_type: str


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(html_path, media_type="text/html")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, or CSV) and ingest it into the RAG pipeline.

    Parameters:
    - file: The document file (PDF, TXT, or CSV)

    Returns:
    - session_id: Unique identifier for this document session
    - file_name: Name of the uploaded file
    - chunk_count: Number of chunks created
    - message: Status message
    """
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Read file into temporary location
        contents = await file.read()

        # Validate file size
        if len(contents) > MAX_UPLOAD_SIZE:
            size_mb = MAX_UPLOAD_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {size_mb:.1f}MB"
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Determine file type
            file_type = file_ext[1:]  # Remove leading dot

            # Generate session ID
            session_id = str(uuid.uuid4())
            collection_name = f"doc_{session_id.replace('-', '')}"

            # Process document through RAG pipeline
            vectorstore, metadata = rag_pipeline.process_document(tmp_path, file_type, collection_name=collection_name)

            # Store vectorstore and metadata
            sessions[session_id] = {
                "vectorstore": vectorstore,
                "file_name": metadata["file_name"],
                "file_type": metadata["file_type"],
                "chunk_count": metadata["chunk_count"],
                "created_at": datetime.utcnow().isoformat(),
                "total_documents": metadata["total_documents"]
            }

            logger.info(f"Document uploaded and processed. Session: {session_id}, File: {file.filename}, Chunks: {metadata['chunk_count']}")

            return UploadResponse(
                session_id=session_id,
                file_name=metadata["file_name"],
                chunk_count=metadata["chunk_count"],
                message=f"Document processed successfully. Created {metadata['chunk_count']} chunks."
            )

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error during upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing document")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about the uploaded document.

    Parameters:
    - session_id: Session ID from upload endpoint
    - question: Natural language question about the document

    Returns:
    - answer: Answer grounded in the document
    - sources: List of source chunks used to answer
    """
    try:
        # Validate session exists
        if request.session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {request.session_id}"
            )

        # Validate question
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Limit question length to prevent token overflow
        question = request.question.strip()[:1000]

        # Get vectorstore from session
        session = sessions[request.session_id]
        vectorstore = session["vectorstore"]

        # Generate answer
        result = rag_pipeline.answer_question(vectorstore, question)

        logger.info(f"Question answered. Session: {request.session_id}, Sources: {len(result['sources'])}")

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during question answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating answer")


@app.get("/sessions/{session_id}", response_model=SessionMetadata)
async def get_session_metadata(session_id: str):
    """
    Get metadata for a session (used to validate session persistence).

    Parameters:
    - session_id: Session ID from upload endpoint

    Returns:
    - Session metadata including file name, chunk count, creation time
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    session = sessions[session_id]
    return SessionMetadata(
        session_id=session_id,
        file_name=session["file_name"],
        chunk_count=session["chunk_count"],
        created_at=session["created_at"],
        file_type=session["file_type"]
    )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with appropriate error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
