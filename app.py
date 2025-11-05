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
from langchain_community.vectorstores import FAISS

# --------------------------------------------------------
# FastAPI App Initialization
# --------------------------------------------------------
app = FastAPI(title="AI RAG Gateway", version="1.0.0")

INDEX_DIR = "faiss_index"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # lightweight and free-tier friendly
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------------
# OpenAI Embeddings wrapper (LangChain-compatible)
# --------------------------------------------------------
class OpenAIEmbeddingsWrapper(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI embeddings."""
        vectors = []
        for text in texts:
            resp = client.embeddings.create(model=EMBED_MODEL, input=text)
            vectors.append(resp.data[0].embedding)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding


# --------------------------------------------------------
# Helper Functions: Save / Load FAISS Index
# --------------------------------------------------------
def load_index():
    if not Path(INDEX_DIR).exists():
        return None
    return FAISS.load_local(INDEX_DIR, OpenAIEmbeddingsWrapper(), allow_dangerous_deserialization=True)

def save_index(db: FAISS):
    db.save_local(INDEX_DIR)


# --------------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------------
@app.get("/health")
def health_check():
    """Check if the API and FAISS index are active."""
    db = load_index()
    vectors = 0
    if db and hasattr(db, "index") and hasattr(db.index, "ntotal"):
        vectors = db.index.ntotal
    return {"status": "ok", "vectors": vectors}


# --------------------------------------------------------
# Root Endpoint
# --------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to AI RAG Gateway! Visit /docs for Swagger UI."}


# --------------------------------------------------------
# Upload PDF and Build FAISS Index
# --------------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create a FAISS vector index."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in environment variables.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        pdf_bytes = await file.read()
        reader = PdfReader(file)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")

    if not pages_text:
        raise HTTPException(status_code=400, detail="No text found in the PDF file.")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for page_text in pages_text:
        chunks.extend(splitter.split_text(page_text))

    docs = [Document(page_content=chunk) for chunk in chunks]

    # Create FAISS vector index
    emb = OpenAIEmbeddingsWrapper()
    db = FAISS.from_documents(docs, emb)
    save_index(db)

    return {"status": "success", "chunks_indexed": len(docs)}


# --------------------------------------------------------
# Query Endpoint
# --------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 4
    score_threshold: float | None = None

@app.post("/query")
def query_pdf(request: QueryRequest):
    """Query the FAISS index and return relevant answers."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in environment variables.")

    db = load_index()
    if not db:
        raise HTTPException(status_code=404, detail="No index found. Please upload a PDF first.")

    if request.score_threshold is not None:
        pairs = db.similarity_search_with_score(request.query, k=max(request.k, 5))
        docs = [d for d, score in pairs if score is None or score > request.score_threshold]
        if not docs:
            docs = [d for d, _ in pairs[:request.k]]
    else:
        docs = db.similarity_search(request.query, k=request.k)

    # Combine the retrieved context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate an answer using OpenAI Responses API
    prompt = f"""
    You are a helpful banking assistant. Answer using only the facts in the context below.
    If the answer is not present, respond: "I don't know based on the provided document."

    CONTEXT:
    {context}

    QUESTION:
    {request.query}

    ANSWER:
    """

    try:
        response = client.responses.create(
            model=os.getenv("GEN_MODEL", "gpt-4.1-mini"),
            input=prompt
        )
        answer = response.output[0].content[0].text if response.output else "No answer generated."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    sources = []
    for doc in docs:
        snippet = doc.page_content[:200].replace("\n", " ")
        sources.append({"doc": "uploaded.pdf", "snippet": snippet + ("..." if len(doc.page_content) > 200 else "")})

    return {"answer": answer, "sources": sources}
