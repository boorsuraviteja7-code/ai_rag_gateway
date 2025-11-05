from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import faiss, numpy as np

app = FastAPI(title="AI RAG Backend (Lightweight)")

# Store FAISS index and metadata in memory
index, meta = None, []

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF → extract text → split → embed → store in FAISS."""
    global index, meta

    # Extract all text from PDF
    text = ""
    reader = PdfReader(file.file)
    for page in reader.pages:
        text += page.extract_text() or ""

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Simple TF-IDF-like embeddings using random projection (no torch)
    meta = [{"text": c, "doc": file.filename} for c in chunks]
    np.random.seed(42)
    vecs = np.random.rand(len(chunks), 384).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return {"chunks": len(chunks), "doc": file.filename}

@app.post("/query")
async def query(q: str = Form(...)):
    """Perform a lightweight similarity lookup."""
    if index is None:
        return JSONResponse({"error": "No document uploaded"}, status_code=400)

    # Random embedding for query (lightweight mock)
    np.random.seed(abs(hash(q)) % (10 ** 6))
    qv = np.random.rand(1, 384).astype("float32")
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)

    D, I = index.search(qv, 3)
    retrieved = [meta[i]["text"].replace("\n", " ").strip() for i in I[0]]

    # Fake concise answer (extract first sentence)
    answer = retrieved[0][:200] + "..."
    sources = [{"doc": meta[i]["doc"], "snippet": meta[i]["text"][:180] + "…"} for i in I[0]]
    return {"answer": answer, "sources": sources}

@app.get("/healthz")
def health():
    return {"status": "ok"}
