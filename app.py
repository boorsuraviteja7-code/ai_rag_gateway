import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from pypdf import PdfReader
from pathlib import Path

# FAISS: use the new import path for 1.12.0
from langchain_community.vectorstores import FAISS

app = FastAPI(title="AI RAG Gateway", version="1.0.0")

INDEX_DIR = "faiss_index"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # small & cheap, good enough
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Embeddings wrapper using OpenAI (langchain-compatible) ----
class OpenAIEmb(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # batch appropriately for your size; here simple split
        out = []
        for t in texts:
            resp = client.embeddings.create(model=EMBED_MODEL, input=t)
            out.append(resp.data[0].embedding)
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding

def load_index():
    if not Path(INDEX_DIR).exists():
        return None
    return FAISS.load_local(INDEX_DIR, OpenAIEmb(), allow_dangerous_deserialization=True)

def save_index(db: FAISS):
    db.save_local(INDEX_DIR)

@app.get("/health")
def health():
    db = load_index()
    vectors = 0
    if db and hasattr(db, "index") and hasattr(db.index, "ntotal"):
        vectors = db.index.ntotal
    return {"status": "ok", "vectors": vectors}

@app.get("/")
def root():
    # Keep root minimal; docs are at /docs
    return {"message": "OK. See /docs for Swagger UI."}

# ---------- Upload PDF & (re)build index ----------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF")

    # Read PDF pages
    try:
        pdf_bytes = await file.read()
        reader = PdfReader(bytes(pdf_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parse failed: {e}")

    if not pages_text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks: List[str] = []
    for t in pages_text:
        chunks.extend(splitter.split_text(t))

    docs = [Document(page_content=ch) for ch in chunks]

    # Build FAISS index
    emb = OpenAIEmb()
    db = FAISS.from_documents(docs, emb)
    save_index(db)

    return {"status": "indexed", "chunks": len(docs)}

# ---------- Query ----------
class QueryIn(BaseModel):
    query: str
    k: int = 4
    score_threshold: float | None = None  # optional

@app.post("/query")
def query(inq: QueryIn):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    db = load_index()
    if not db:
        raise HTTPException(status_code=404, detail="No index on disk. Upload a PDF first.")

    # similarity search
    if inq.score_threshold is not None:
        pairs = db.similarity_search_with_score(inq.query, k=max(inq.k, 5))
        docs = [d for d, score in pairs if score is None or score > inq.score_threshold]
        if not docs:
            # fallback to top-k if threshold filters everything out
            docs = [d for d, _ in pairs[:inq.k]]
    else:
        docs = db.similarity_search(inq.query, k=inq.k)

    context = "\n\n".join([d.page_content for d in docs[:inq.k]])

    # Simple RAG answer with OpenAI responses API
    # (If you prefer chat models, you can switch to client.chat.completions)
    prompt = f"""You are a helpful banking assistant. Answer using only the facts in CONTEXT.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION: {inq.query}
ANSWER:"""

    resp = client.responses.create(
        model=os.getenv("GEN_MODEL", "gpt-4.1-mini"),
        input=prompt
    )
    answer = resp.output[0].content[0].text if resp and resp.output else "I couldn't generate an answer."

    # Return short source snippets
    sources = []
    for d in docs[:inq.k]:
        snippet = d.page_content[:220].replace("\n", " ")
        sources.append({"doc": "uploaded.pdf", "snippet": snippet + ("..." if len(d.page_content) > 220 else "")})

    return {"answer": answer, "sources": sources}
