# main.py
import io
import os
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="General PDF Analyst (Groq + HF Embeddings)")

GLOBAL_DOCS = []
GLOBAL_VECTORSTORE = None
GLOBAL_RETRIEVER = None

# Groq LLM
LLM = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
)

# HuggingFace Inference API embeddings (Render-safe)
EMBEDDINGS = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_API_KEY"),
    model_name="BAAI/bge-small-en-v1.5"
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
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global GLOBAL_DOCS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")

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
        raise HTTPException(status_code=400, detail="No PDFs indexed yet.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = GLOBAL_RETRIEVER.invoke(question)
    context = join_unique_docs(docs)

    if not context:
        answer = "I could not find relevant information in the provided PDF context."
        return QueryResponse(answer=answer, quote=None, confidence="Low", context=None)

    prompt = f"""
Use ONLY the CONTEXT below to answer the QUESTION.
If the context does not contain the answer, say so.
Always quote the exact sentence from the context.

CONTEXT:
{context}

QUESTION:
{question}
"""

    llm_resp = LLM.invoke(prompt)
    answer = llm_resp.content

    low_confidence = any(
        p in answer.lower()
        for p in ["could not find", "no information", "insufficient"]
    )
    confidence = "Low" if low_confidence else "High"

    quote = None
    if '"' in answer:
        try:
            quote = answer.split('"')[1]
        except:
            quote = None

    return QueryResponse(answer=answer, quote=quote, confidence=confidence, context=context)


@app.get("/status")
def status():
    return {
        "status": "online",
        "docs_indexed": len(GLOBAL_DOCS),
        "has_vectorstore": GLOBAL_VECTORSTORE is not None
    }
