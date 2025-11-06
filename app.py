import os
from pathlib import Path

import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss  # from faiss-cpu
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="AI RAG Gateway", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be set in Render -> Environment
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# In-memory state
# ---------------------------
# We maintain a single FAISS index & the corresponding chunk texts.
# (Render free tier may spin down; persistence is OPTIONAL here.)
INDEX = None              # faiss.IndexFlatIP (cosine similarity with normalized vectors)
EMBED_DIM = 1536          # text-embedding-3-small
CHUNKS: list[str] = []    # same order as vectors added to FAISS


# ---------------------------
# Helpers
# ---------------------------
def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize for cosine similarity with IndexFlatIP."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def get_embeddings_httpx(texts: list[str]) -> np.ndarray:
    """
    Call OpenAI embeddings via raw HTTP to avoid the SDK and the 'proxies' bug.
    Returns float32 numpy array of shape (n, EMBED_DIM)
    """
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts,
    }
    try:
        # do NOT pass proxies — we want a plain call
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        vecs = [d["embedding"] for d in data["data"]]
        arr = np.array(vecs, dtype=np.float32)
        return arr
    except Exception as e:
        raise RuntimeError(f"Embedding HTTP call failed: {e}")


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    out = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            out.append(t)
    return "\n\n".join(out)


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": len(CHUNKS),
        "embed_dim": EMBED_DIM,
        "has_index": INDEX is not None,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, split to chunks, embed via raw HTTP, and build FAISS index.
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF")

        # save locally (optional but handy for debugging)
        path = VECTOR_STORE_DIR / file.filename
        with open(path, "wb") as f:
            f.write(await file.read())

        text = pdf_to_text(path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text in PDF")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from PDF")

        # Get embeddings (n, d) and normalize for cosine/IP search
        embeds = get_embeddings_httpx(chunks)
        embeds = _normalize(embeds)

        # Build FAISS index
        global INDEX, CHUNKS
        INDEX = faiss.IndexFlatIP(embeds.shape[1])  # IP + normalized -> cosine
        INDEX.add(embeds)
        CHUNKS = chunks

        return {"message": f"{file.filename} processed", "chunks": len(chunks)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


class QueryRequest(BaseModel):
    query: str
    k: int | None = 3


@app.post("/query")
def query_rag(req: QueryRequest):
    """
    Embed the query, search top-k, and return the matched chunk texts.
    """
    try:
        if INDEX is None or not CHUNKS:
            raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

        q = req.query.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        q_vec = get_embeddings_httpx([q])
        q_vec = _normalize(q_vec)
        k = max(1, min(req.k or 3, len(CHUNKS)))

        D, I = INDEX.search(q_vec.astype(np.float32), k)  # (1, k)
        hits = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "text": CHUNKS[idx][:1000],  # clamp for payload size
                    "chunk_index": int(idx),
                }
            )
        return {"matches": hits}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")
