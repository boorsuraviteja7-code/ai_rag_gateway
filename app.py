import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import openai
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = FastAPI(title="AI RAG Gateway", version="1.0")

# CORS setup for browser or Postman access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store = None


def get_embeddings(texts):
    """Stateless embedding generator — no proxy issues."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise RuntimeError(f"Embedding API failed: {str(e)}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0
    }


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

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c) for c in chunks]

        # Generate embeddings
        embeddings_list = get_embeddings([doc.page_content for doc in docs])

        # Create FAISS vector index
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        faiss = dependable_faiss_import()
        dim = len(embeddings_list[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings_list).astype("float32"))

        global vector_store
        vector_store = FAISS(embedding_function=get_embeddings, index=index, documents=docs)

        return {"message": f"{file.filename} processed successfully", "chunks": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


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
