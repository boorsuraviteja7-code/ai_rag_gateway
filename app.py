import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI RAG Gateway", version="1.0")

# Allow frontend or Postman access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Embeddings ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# ---- In-memory store ----
VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store = None

@app.get("/health")
def health():
    return {"status": "ok", "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0}

# ---- PDF Upload ----
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

        global vector_store
        vector_store = FAISS.from_documents(docs, embedding=embeddings)

        return {"message": f"File {file.filename} processed successfully", "chunks": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ---- Query ----
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        global vector_store
        if not vector_store:
            raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a PDF first.")

        results = vector_store.similarity_search(request.query, k=3)
        combined_text = " ".join([r.page_content for r in results])

        # Simple heuristic response
        return {"answer": combined_text[:1000]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
