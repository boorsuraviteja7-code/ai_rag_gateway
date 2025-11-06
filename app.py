# app.py
import os
import sys
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Safe env for Render ----------
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "SENTENCE_TRANSFORMERS_HOME": "/tmp",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
})

# Keep imports light at boot
if os.environ.get("RENDER") == "true":
    sys.modules["torch"] = __import__("torch", fromlist=[""])

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

from PyPDF2 import PdfReader

# Lazy HF imports happen later
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(
    title="RaviTeja GenAI Gateway",
    description="PDF RAG microservice for Java integration: upload → ask",
    version="2.0.0"
)

# ---------- CORS so Java/Frontend can call this ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health & root (Render probes) ----------
@app.get("/")
def root():
    return {"status": "running ✅", "docs": "/docs", "health": "/health"}

@app.head("/")
def head_root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True, "message": "Healthy 💚"}

# ---------- Global in-memory store (ephemeral on Render) ----------
# store[doc_id] = {"chunks": List[str], "emb": np.ndarray (N, D), "meta": {...}}
store: Dict[str, Dict[str, Any]] = {}

# ---------- Lazy model loaders ----------
_TEXT_EMBED_MODEL = None
_GEN_PIPE = None

def get_embedder():
    """
    Lightweight sentence-transformers encoder (CPU-friendly).
    """
    global _TEXT_EMBED_MODEL
    if _TEXT_EMBED_MODEL is None:
        # Small, fast, good quality for retrieval
        _TEXT_EMBED_MODEL = SentenceTransformer(
            os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    return _TEXT_EMBED_MODEL

def get_generator():
    """
    Tiny text2text generator for forming concise answers.
    You can swap to a larger model later (e.g., flan-t5-base).
    """
    global _GEN_PIPE
    if _GEN_PIPE is None:
        model_name = os.getenv("GEN_MODEL", "google/flan-t5-small")
        _GEN_PIPE = pipeline("text2text-generation", model=model_name, device=-1)
    return _GEN_PIPE

# ---------- Simple helpers ----------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Character-based chunking that’s stable for PDFs.
    """
    text = text.replace("\r", " ").replace("\n", " ").strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to cut at a period/space if possible
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start + 200:
            cut = end
        chunks.append(text[start:cut].strip())
        start = max(0, cut - overlap)
    return [c for c in chunks if c]

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int = 4):
    """
    query_vec: (D,) ; mat: (N, D)
    returns indices of top-k most similar rows
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = l2_normalize(mat)
    scores = m @ q  # cosine similarity
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]

# ---------- API: Upload PDF ----------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF → returns doc_id that you’ll use when calling /ask.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file."}

    reader = PdfReader(file.file)
    full_text = ""
    for p in reader.pages:
        full_text += (p.extract_text() or "") + "\n"

    full_text = full_text.strip()
    if not full_text:
        return {"error": "No readable text found in PDF."}

    chunks = chunk_text(full_text)
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    doc_id = str(uuid.uuid4())
    store[doc_id] = {
        "chunks": chunks,
        "emb": embeddings,
        "meta": {"filename": file.filename, "num_chunks": len(chunks), "text_len": len(full_text)},
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_chunks": len(chunks),
        "text_length": len(full_text),
        "preview": chunks[0][:500] if chunks else ""
    }

# ---------- API: Ask a question over a PDF ----------
class AskIn(BaseModel):
    doc_id: str
    question: str
    top_k: int = 4
    max_new_tokens: int = 128

@app.post("/ask")
def ask_pdf(payload: AskIn):
    """
    Retrieve top-k chunks from the uploaded PDF and generate an answer.
    """
    if payload.doc_id not in store:
        return {"error": f"Unknown doc_id: {payload.doc_id}. Upload a PDF first."}

    entry = store[payload.doc_id]
    chunks: List[str] = entry["chunks"]
    emb: np.ndarray = entry["emb"]

    # Embed question
    embedder = get_embedder()
    q_vec = embedder.encode([payload.question], convert_to_numpy=True)[0].astype(np.float32)

    # Retrieve
    k = max(1, min(payload.top_k, len(chunks)))
    idx, scores = cosine_top_k(q_vec, emb, k)
    context = "\n\n".join([f"[{i}] {chunks[i]}" for i in idx])

    # Generate answer with context
    generator = get_generator()
    prompt = (
        "You are a helpful assistant. Answer strictly from the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.question}\n"
        "Answer:"
    )
    out = generator(prompt, max_new_tokens=payload.max_new_tokens)
    answer_text = out[0]["generated_text"].strip()

    return {
        "doc_id": payload.doc_id,
        "question": payload.question,
        "answer": answer_text,
        "retrieved_chunks": idx.tolist(),
        "scores": [float(s) for s in scores],
        "context_preview": context[:1000],
        "meta": entry["meta"],
    }

# ---------- (Optional) Basic text generation kept for compatibility ----------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64

@app.post("/generate")
def generate_text(request: GenerateRequest):
    generator = get_generator()
    out = generator(request.prompt, max_new_tokens=request.max_new_tokens)
    return {"input": request.prompt, "output": out[0]["generated_text"].strip()}

# ---------- Local dev entry ----------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RaviTeja GenAI Gateway on port 10000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
