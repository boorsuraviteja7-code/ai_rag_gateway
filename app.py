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
# ✅ Optimize for low memory
# ----------------------------------------------------
torch.set_num_threads(1)  # prevent PyTorch from spawning extra threads

# ----------------------------------------------------
# ✅ Initialize FastAPI
# ----------------------------------------------------
app = FastAPI(title="AI RAG Gateway (Free Version)", version="1.0")

# Enable CORS (so you can access via browser or Postman)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# ✅ Load Lightweight Local Embedding Model (cached)
# ----------------------------------------------------
# Cache model under /tmp (Render allows this space)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp"

# Use the lightweight model (only ~120 MB)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

def get_embeddings(texts):
    """Generate embeddings locally without OpenAI API."""
    try:
        embeddings = embedding_model.encode(texts)
        return np.array(embeddings).tolist()
    except Exception as e:
        raise RuntimeError(f"Local embedding generation failed: {e}")

# ----------------------------------------------------
# ✅ Create Vector Store Folder
# ----------------------------------------------------
VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store = None

# ----------------------------------------------------
# ✅ Health Check Endpoint
# ----------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0
    }

# ----------------------------------------------------
# ✅ PDF Upload Endpoint
# ----------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create vector embeddings."""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file")

        pdf_path = Path(VECTOR_STORE_PATH) / file.filename
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        pdf_reader = PdfReader(str(pdf_path))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in the PDF")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c) for c in chunks]

        # Generate local embeddings
        embeddings_list = get_embeddings([doc.page_content for doc in docs])

        # Create FAISS index
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        faiss = dependable_faiss_import()
        dim = len(embeddings_list[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings_list).astype("float32"))

        # Store globally
        global vector_store
        vector_store = FAISS(embedding_function=get_embeddings, index=index, documents=docs)

        return {"message": f"{file.filename} processed successfully", "chunks": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ----------------------------------------------------
# ✅ Query Endpoint
# ----------------------------------------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the stored embeddings for similarity search."""
    try:
        global vector_store
        if not vector_store:
            raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a PDF first.")

        query_vector = get_embeddings([request.query])[0]
        D, I = vector_store.index.search(np.array([query_vector]).astype("float32"), k=3)
        matched_docs = [vector_store.documents[i].page_content for i in I[0]]

        return {"answer": " ".join(matched_docs)[:1000]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
