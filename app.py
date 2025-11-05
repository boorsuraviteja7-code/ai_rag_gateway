# app.py
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import tempfile
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

# --- Initialize FastAPI app ---
app = FastAPI(title="AI RAG Gateway (Lightweight Version)")

# --- Global vars (lazy initialization) ---
model = None
vectorstore = None

# --- Function to lazy-load the model only once ---
def load_model():
    global model
    if model is None:
        print("Loading embedding model (this may take ~30s)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@app.get("/")
async def root():
    return {"message": "AI RAG Gateway is running!"}


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload and embed a document"""
    global vectorstore
    model = load_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # Read and split the document
    with open(temp_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    os.remove(temp_path)
    return {"status": "File processed and stored"}


class Query(BaseModel):
    question: str


@app.post("/query")
async def query_doc(q: Query):
    """Query the embedded document"""
    global vectorstore
    if vectorstore is None:
        return {"error": "No document uploaded yet!"}

    docs = vectorstore.similarity_search(q.question, k=3)
    return {"answer_chunks": [d.page_content for d in docs]}
