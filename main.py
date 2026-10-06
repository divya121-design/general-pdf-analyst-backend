import logging
import os
import tempfile
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="General PDF Analyst (Groq + HF Embeddings)")


# Force OpenAPI 3.0.3 schema so Swagger UI renders real file pickers
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version="3.0.3",
        routes=app.routes,
    )

    # Force binary file input UI schema for Swagger /upload
    try:
        if "/upload" in openapi_schema["paths"]:
            openapi_schema["paths"]["/upload"]["post"]["requestBody"] = {
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "binary"},
                                    "description": "Select PDF files to upload",
                                }
                            },
                            "required": ["files"],
                        }
                    }
                },
                "required": True,
            }
    except Exception as e:
        logger.error(f"Failed to patch OpenAPI schema: {e}")

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

GLOBAL_DOCS = []
GLOBAL_VECTORSTORE = None
GLOBAL_RETRIEVER = None

groq_api_key = os.getenv("GROQ_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

LLM = (
    ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )
    if groq_api_key
    else None
)

EMBEDDINGS = (
    HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=hf_api_key,
    )
    if hf_api_key
    else None
)


def join_unique_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return "\n\n".join(unique)


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 6


class QueryResponse(BaseModel):
    answer: str
    quote: Optional[str] = None
    confidence: str
    context: Optional[str] = None


@app.post("/upload")
async def upload_pdfs(
    files: Annotated[
        list[UploadFile],
        File(description="Select one or more PDF files to analyze"),
    ],
):
    global GLOBAL_DOCS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER

    if not EMBEDDINGS:
        raise HTTPException(
            status_code=500,
            detail="HF_API_KEY is missing or invalid in environment variables.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"File '{f.filename}' is not a PDF."
            )

        content = await f.read()
        if not content:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
        except Exception as e:
            logger.error(f"Error parsing PDF {f.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from {f.filename}: {str(e)}",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_docs:
        raise HTTPException(
            status_code=400, detail="No readable text found in uploaded PDF(s)."
        )

    try:
        GLOBAL_DOCS = all_docs
        GLOBAL_VECTORSTORE = FAISS.from_documents(GLOBAL_DOCS, EMBEDDINGS)
        GLOBAL_RETRIEVER = GLOBAL_VECTORSTORE.as_retriever(search_kwargs={"k": 6})

    except Exception as e:
        logger.error(f"Error building vectorstore: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document embeddings: {str(e)}",
        )

    return {"status": "indexed", "documents": len(GLOBAL_DOCS)}


@app.post("/query", response_model=QueryResponse)
def query_pdfs(req: QueryRequest):
    global GLOBAL_RETRIEVER, LLM

    if not LLM:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing in environment variables.",
        )

    if GLOBAL_RETRIEVER is None:
        raise HTTPException(
            status_code=400,
            detail="No PDFs indexed yet. Please upload PDFs via /upload first.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        docs = GLOBAL_RETRIEVER.invoke(question)
        context = join_unique_docs(docs)
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error querying vectorstore: {str(e)}"
        )

    if not context:
        return QueryResponse(
            answer="I could not find relevant information in the provided PDF context.",
            quote=None,
            confidence="Low",
            context=None,
        )

    prompt = f"""
Use ONLY the CONTEXT below to answer the QUESTION.
If the context does not contain the answer, say so clearly.
Always quote the exact sentence from the context if relevant.

CONTEXT:
{context}

QUESTION:
{question}
"""

    llm_resp = LLM.invoke(prompt)
    answer = llm_resp.content

    low_confidence = any(
        p in answer.lower()
        for p in [
            "could not find",
            "no information",
            "insufficient",
            "does not contain",
        ]
    )
    confidence = "Low" if low_confidence else "High"

    quote = None
    if '"' in answer:
        try:
            quote = answer.split('"')[1]
        except Exception:
            quote = None

    return QueryResponse(
        answer=answer, quote=quote, confidence=confidence, context=context
    )


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


@app.get("/status")
def status():
    return {
        "status": "online",
        "docs_indexed": len(GLOBAL_DOCS),
        "has_vectorstore": GLOBAL_VECTORSTORE is not None,
        "keys_configured": bool(groq_api_key and hf_api_key),
    }