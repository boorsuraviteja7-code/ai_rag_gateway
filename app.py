import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from pathlib import Path
from openai import OpenAI

# ✅ Initialize OpenAI Client (new SDK style)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Initialize FastAPI
app = FastAPI(title="AI RAG Gateway", version="2.0")

# Global FAISS index
vectorstore = None


# ---------- MODELS ----------
class QueryRequest(BaseModel):
    query: str


# ---------- HELPERS ----------
def load_pdf_and_split(file_path: str) -> List[Document]:
    """Read a PDF file and split it into text chunks."""
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs


def get_embeddings(texts: List[str]):
    """Use OpenAI embeddings (text-embedding-3-small)"""
    embeddings = []
    for t in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


# ---------- ROUTES ----------
@app.get("/health")
def health():
    """Check API health"""
    return {"status": "ok", "vectors": 0 if vectorstore is None else len(vectorstore.index_to_docstore_id)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file into FAISS"""
    global vectorstore
    try:
        # Save uploaded PDF
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Read and split
        docs = load_pdf_and_split(str(file_path))
        texts = [d.page_content for d in docs]
        embeds = get_embeddings(texts)

        # Build FAISS index
        vectorstore = FAISS.from_embeddings(list(zip(texts, embeds)))
        return {"message": f"✅ {file.filename} processed successfully", "chunks": len(texts)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query uploaded knowledge base"""
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="❌ No documents uploaded yet")

    try:
        # Embed query
        query_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        ).data[0].embedding

        # Retrieve similar docs
        results = vectorstore.similarity_search_by_vector(query_embed, k=3)
        context = "\n".join([r.page_content for r in results])

        # Ask OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer based on the provided document context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "sources": [r.page_content[:200] for r in results]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")
