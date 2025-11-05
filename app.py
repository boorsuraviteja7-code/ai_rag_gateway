from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="AI Knowledge Assistant")

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load small T5 model for generating short answers
t5_name = "google/flan-t5-base"
t5_tok = AutoTokenizer.from_pretrained(t5_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_name)

index, meta = None, []

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF → extract → split → embed → store in FAISS."""
    global index, meta
    text = ""
    reader = PdfReader(file.file)
    for p in reader.pages:
        text += p.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    meta = [{"text": c, "doc": file.filename} for c in chunks]
    vecs = embedder.encode([m["text"] for m in meta], normalize_embeddings=True)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.array(vecs, dtype="float32"))
    return {"chunks": len(chunks), "doc": file.filename}

@app.post("/query")
async def query(q: str = Form(...)):
    """Answer questions based on uploaded PDF content."""
    if index is None:
        return JSONResponse({"error": "No document uploaded"}, status_code=400)

    qv = embedder.encode([q], normalize_embeddings=True)
    D, I = index.search(np.array(qv, dtype="float32"), 3)
    retrieved = [meta[i]["text"].replace("\n", " ").strip() for i in I[0]]
    context = "\n\n".join(retrieved)
    prompt = (
        "Answer the question using ONLY the context. "
        "If the answer is not in context, reply exactly: I don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\nShort answer:"
    )
    inputs = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = t5_model.generate(**inputs, max_new_tokens=80, num_beams=4, temperature=0.2, early_stopping=True)
    answer = t5_tok.decode(outputs[0], skip_special_tokens=True).strip()
    sources = [{"doc": meta[i]["doc"], "snippet": meta[i]["text"][:180] + "…"} for i in I[0]]
    return JSONResponse({"answer": answer, "sources": sources})
