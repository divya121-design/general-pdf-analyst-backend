import io
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = FastAPI(title="General PDF Analyst (Agentic RAG)")

GLOBAL_DOCS = []
GLOBAL_VECTORSTORE = None
GLOBAL_RETRIEVER = None

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def join_unique_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return "\n\n".join(unique)


def call_llm_safe(prompt: str) -> str:
    # Placeholder: replace with real LLM call later
    return f"[LLM placeholder] Answer based on context:\n\n{prompt[:1000]}"


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 6


class QueryResponse(BaseModel):
    answer: str
    quote: Optional[str] = None
    confidence: str
    context: Optional[str] = None


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global GLOBAL_DOCS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {f.filename} is not a PDF.")

        content = await f.read()
        pdf_bytes = io.BytesIO(content)

        loader = PyPDFLoader(pdf_bytes)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No text extracted from PDFs.")

    GLOBAL_DOCS = all_docs

    GLOBAL_VECTORSTORE = Chroma.from_documents(
        documents=GLOBAL_DOCS,
        embedding=EMBEDDINGS,
        persist_directory="./uploaded_db"
    )
    GLOBAL_RETRIEVER = GLOBAL_VECTORSTORE.as_retriever(search_kwargs={"k": 6})

    return {"status": "indexed", "documents": len(GLOBAL_DOCS)}


@app.post("/query", response_model=QueryResponse)
def query_pdfs(req: QueryRequest):
    global GLOBAL_RETRIEVER

    if GLOBAL_RETRIEVER is None:
        raise HTTPException(status_code=400, detail="No PDFs indexed yet. Upload PDFs first via /upload.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = GLOBAL_RETRIEVER.invoke(question)
    context = join_unique_docs(docs)

    if not context:
        answer = "I could not find relevant information in the provided PDF context."
        return QueryResponse(
            answer=answer,
            quote=None,
            confidence="Low",
            context=None
        )

    prompt = f"""
You are a general-purpose analyst. Use ONLY the CONTEXT below (extracted from user-provided PDFs) to answer the QUESTION.
If the context does not contain the answer, say "I could not find relevant information in the provided PDF context."
Always quote the exact sentence or short excerpt from the CONTEXT that supports your answer.

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely, then provide the supporting quote (if any).
"""

    answer = call_llm_safe(prompt)

    low_confidence_phrases = [
        "no information",
        "could not find",
        "not enough information",
        "no relevant data",
        "i could not find",
        "insufficient information",
    ]
    confidence = "High"
    if any(p in answer.lower() for p in low_confidence_phrases):
        confidence = "Low"

    quote = None
    if '"' in answer:
        try:
            quote = answer.split('"')[1]
        except Exception:
            quote = None

    return QueryResponse(
        answer=answer,
        quote=quote,
        confidence=confidence,
        context=context
    )


@app.get("/status")
def status():
    return {
        "status": "online",
        "docs_indexed": len(GLOBAL_DOCS),
        "has_vectorstore": GLOBAL_VECTORSTORE is not None
    }
