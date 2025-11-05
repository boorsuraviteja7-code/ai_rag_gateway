from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os
import shutil
from tempfile import NamedTemporaryFile

# Initialize FastAPI app
app = FastAPI(title="AI RAG Gateway", version="1.0")

# Allow CORS (lets frontend / Postman access your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FAISS index variable
index = None


# 🩺 HEALTH CHECK ENDPOINT
@app.get("/health")
def health():
    """
    Simple endpoint to confirm if your API is running.
    It also shows how many vector embeddings are currently loaded.
    """
    global index
    return {
        "status": "ok",
        "vectors": int(index.index.ntotal) if index else 0
    }


# 📂 UPLOAD ENDPOINT
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file, split text, embed it using OpenAI,
    and save a FAISS vector index.
    """
    global index

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split text into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Build FAISS vector store
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local("faiss_index")

    os.remove(tmp_path)  # Clean up the temp file

    return {"doc": file.filename, "chunks": len(chunks), "vectors": index.index.ntotal}


# 💬 QUERY ENDPOINT
@app.post("/query")
async def query_document(payload: dict):
    """
    Query the FAISS index to find related text for a given question.
    """
    global index
    if not index:
        # Load FAISS index if not already loaded
        if os.path.exists("faiss_index"):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            return {"error": "No index loaded. Please upload a PDF first."}

    query_text = payload.get("query")
    if not query_text:
        return {"error": "Missing 'query' parameter"}

    # Perform similarity search
    docs = index.similarity_search(query_text, k=3)
    response = {
        "answer": docs[0].page_content if docs else "No relevant text found.",
        "sources": [
            {"doc": "deposit-account-agreement.pdf", "snippet": d.page_content[:200]} for d in docs
        ],
    }

    return response

