import os
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

# langchain bits (vector store + chunking + doc type)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------------------
#         RUNTIME TUNING
# -------------------------------
# keep torch light on CPU
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Important: don't download huge models; we will use a tiny sentence-transformers model.
# DO NOT set TRANSFORMERS_OFFLINE=1 unless you have already cached the model on Render,
# otherwise first boot will fail when it can't download the files.
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp")  # cache in ephemeral disk

# -------------------------------
#         FASTAPI APP
# -------------------------------
app = FastAPI(title="AI RAG Gateway (Free & Stable)", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
#   LAZY-LOAD EMBEDDING MODEL
# -------------------------------
_embedding_model = None
_MODEL_NAME = "thenlper/gte-tiny"   # ~22M params, very small & accurate for retrieval

def _load_model():
    """Load the embedding model only when first needed."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # import here to avoid heavy import at boot
        _embedding_model = SentenceTransformer(_MODEL_NAME, device="cpu")
        # shorter sequence helps RAM; tune if needed
        try:
            _embedding_model.max_seq_length = 256
        except Exception:
            pass
    return _embedding_model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute embeddings with tiny model, return list of vectors."""
    try:
        model = _load_model()
        # convert to float32 (FAISS default)
        emb = model.encode(texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb.tolist()
    except Exception as e:
        # free any cached tensors if something failed
        gc.collect()
        raise RuntimeError(f"Embedding failed: {e}") from e

# -------------------------------
#         VECTOR STORE
# -------------------------------
VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# we keep index in-memory; (optional) you can persist to disk later
vector_store: FAISS | None = None

# -------------------------------
#         HEALTHCHECK
# -------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0,
        "model": _MODEL_NAME if _embedding_model is not None else "not_loaded"
    }

# -------------------------------
#         UPLOAD PDF
# -------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload one PDF, chunk it, embed it with tiny model, and build FAISS index.
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        pdf_path = Path(VECTOR_STORE_DIR) / file.filename
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Extract text (pypdf)
        reader = PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in the PDF.")

        # Split into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not split PDF into text chunks.")
        docs = [Document(page_content=c) for c in chunks]

        # Embeddings (lazy load happens here)
        vectors = get_embeddings([d.page_content for d in docs])

        # Build FAISS index (float32 for L2; FAISS expects float32)
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        faiss = dependable_faiss_import()
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.asarray(vectors, dtype=np.float32))

        global vector_store
        vector_store = FAISS(embedding_function=get_embeddings, index=index, documents=docs)

        # clean temp memory
        gc.collect()

        return {"message": f"{file.filename} processed successfully", "chunks": len(docs)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}") from e

# -------------------------------
#            QUERY
# -------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        global vector_store
        if not vector_store:
            raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

        qv = np.asarray(get_embeddings([request.query])[0], dtype=np.float32)

        # FAISS search
        D, I = vector_store.index.search(qv.reshape(1, -1), k=3)

        hits = []
        for idx in I[0]:
            if idx == -1:
                continue
            hits.append(vector_store.documents[idx].page_content)

        answer = " ".join(hits)[:1000] if hits else "No relevant text found."
        return {"answer": answer, "matches": len(hits)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}") from e
