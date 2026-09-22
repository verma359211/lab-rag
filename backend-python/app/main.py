"""Expose the RAG features as a small HTTP API.

This module deals only with web concerns: routes, uploaded files, request
validation, and HTTP errors. The actual RAG work is delegated to the ingestion
and answering modules. This separation lets us understand or change the RAG
pipeline without mixing it with FastAPI details.
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_frontend_origins
from app.documents import delete_document, get_document_chunks, list_documents
from app.ingestion import ingest_pdf
from app.rag import answer_question
from app.schemas import ChatRequest


app = FastAPI(title="LangChain RAG API")

# The React app now calls FastAPI directly using its complete backend URL.
# Because ports 5173 and 3000 are different browser origins, FastAPI must
# explicitly allow the frontend origin to read its responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Confirm that the FastAPI process is running.

    This deliberately stays lightweight. It does not call Gemini, Groq, or the
    database, so it quickly tells us whether the web server started correctly.
    """

    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Accept one PDF and send it through the ingestion pipeline.

    Upload reading is asynchronous because the data arrives through an HTTP
    request. After reading it, the ingestion module handles PDF parsing,
    splitting, embedding, and database storage.
    """

    filename = file.filename or "document.pdf"

    # Some clients provide the PDF MIME type and others only provide a useful
    # filename. Accept either signal, but reject clearly unsupported files.
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty")

    try:
        chunk_count = ingest_pdf(pdf_bytes, filename)
        return {"message": f'Ingested {chunk_count} chunks from "{filename}"'}
    except Exception as error:
        # FastAPI turns this exception into a JSON response with status 500.
        # Keeping the original exception as the cause preserves its traceback.
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat")
def chat(request: ChatRequest):
    """Answer one question using context retrieved from uploaded PDFs."""

    # Stripping prevents spaces-only input from causing a database and model
    # request that cannot produce a useful answer.
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        return answer_question(question)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/documents")
def documents():
    """List stored PDFs by grouping vector chunks with the same source."""

    try:
        return {"documents": list_documents()}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/documents/chunks")
def document_chunks(source: str):
    """Show the chunk text currently stored for one source filename."""

    clean_source = source.strip()

    if not clean_source:
        raise HTTPException(status_code=400, detail="source is required")

    try:
        chunks = get_document_chunks(clean_source)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if not chunks:
        raise HTTPException(status_code=404, detail="Document was not found")

    return {"source": clean_source, "chunks": chunks}


@app.delete("/documents")
def remove_document(source: str):
    """Delete every stored chunk currently grouped under one source name."""

    clean_source = source.strip()

    if not clean_source:
        raise HTTPException(status_code=400, detail="source is required")

    try:
        deleted_count = delete_document(clean_source)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document was not found")

    return {
        "message": f'Deleted "{clean_source}" and its {deleted_count} chunks',
        "deletedChunks": deleted_count,
    }
