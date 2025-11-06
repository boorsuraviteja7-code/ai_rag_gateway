import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader
from pathlib import Path
import openai

app = FastAPI()

# ✅ Initialize OpenAI using new syntax
openai.api_key = os.getenv("OPENAI_API_KEY")

vector_store = None

# ✅ Embedding helper function
def get_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_path = Path(f"/tmp/{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        reader = PdfReader(temp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

        global vector_store
        vector_store = FAISS.from_documents(docs, embedding_function=get_embedding)

        return {"message": f"File {file.filename} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_rag(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=400, detail="No document uploaded yet")

    docs = vector_store.similarity_search(request.query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"Answer based on the document:\n{context}\n\nQuestion: {request.query}\nAnswer:"

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "vectors": len(vector_store.index_to_docstore_id) if vector_store else 0}
