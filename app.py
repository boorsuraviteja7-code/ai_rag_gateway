import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pathlib import Path
import numpy as np

# ----------------------------------------------------
# ✅ Optimize memory + offline model caching
# ----------------------------------------------------
torch.set_num_threads(1)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ----------------------------------------------------
# ✅ Initialize FastAPI
# ----------------------------------------------------
app = FastAPI(title="AI RAG Gateway (Free & Stable)", version="2.0")

# Allow browser/Postman access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# ✅ Load lightweight embedding model
# ----------------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

def get_embeddings(texts):
    """Generate local embeddings without any API or cost."""
    try:
        emb = embedding_model.encode(texts)
        return np.array(emb).tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

# ----------------------------------------------------
# ✅ Vector store setup
# ----------------------------------------------------
VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store = None

# ----------------------------------------------------
# ✅ Health check
# ----------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0
    }

# ----------------------------------------------------
# ✅ Upload PDF → generate embeddings → store in FAISS
# ----------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        pdf_path = Path(VECTOR_STORE_PATH) / file.filename
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Extract text
        pdf_reader = PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in the PDF.")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c) for c in chunks]

        # Embeddings
        emb_list = get_embeddings([d.page_content for d in docs])

        # Create FAISS index (float16 saves memory)
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        faiss = dependable_faiss_import()
        dim = len(emb_list[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(emb_list).astype("float16"))

        global vector_store
        vector_store = FAISS(embedding_function=get_embeddings, index=index, documents=docs)

        return {"message": f"{file.filename} processed successfully", "chunks": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ----------------------------------------------------
# ✅ Query endpoint
# ----------------------------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        global vector_store
        if not vector_store:
            raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a PDF first.")

        qv = get_embeddings([request.query])[0]
        D, I = vector_store.index.search(np.array([qv]).astype("float16"), k=3)
        results = [vector_store.documents[i].page_content for i in I[0]]

        return {"answer": " ".join(results)[:1000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
