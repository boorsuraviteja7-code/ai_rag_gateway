import os, gc, torch, numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------------- CONFIG ----------------------
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp"

app = FastAPI(title="AI RAG Gateway (Free-Tier Mode)", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store = None

# ---------------------- UTILS ----------------------
def get_embeddings(texts):
    """Load model on-demand, compute embeddings, unload to free RAM."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
    model.max_seq_length = 256
    emb = model.encode(texts, convert_to_numpy=True, batch_size=8, normalize_embeddings=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return emb.astype(np.float32).tolist()

# ---------------------- ROUTES ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Please upload a PDF file")

        pdf_path = Path(VECTOR_STORE_PATH) / file.filename
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        reader = PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise HTTPException(400, "No text found in PDF")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c) for c in chunks]

        vecs = get_embeddings([d.page_content for d in docs])
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        faiss = dependable_faiss_import()
        dim = len(vecs[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vecs, dtype=np.float32))

        global vector_store
        vector_store = FAISS(embedding_function=get_embeddings, index=index, documents=docs)

        gc.collect()
        return {"message": f"{file.filename} processed successfully", "chunks": len(docs)}
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(req: QueryRequest):
    try:
        global vector_store
        if not vector_store:
            raise HTTPException(400, "No documents indexed yet")

        qv = np.array(get_embeddings([req.query])[0], dtype=np.float32)
        D, I = vector_store.index.search(qv.reshape(1, -1), k=3)
        matches = [vector_store.documents[i].page_content for i in I[0] if i != -1]
        answer = " ".join(matches)[:1000] if matches else "No relevant text found."
        return {"answer": answer, "matches": len(matches)}
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
