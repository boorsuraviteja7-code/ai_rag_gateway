import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------------------------------------------------
# 1️⃣  Initialize FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="RaviTeja GenAI Gateway",
    version="1.0.0",
    description="FastAPI app deployed on Render with HuggingFace lazy loading."
)

# -------------------------------------------------------------------
# 2️⃣  Lazy model loading
# -------------------------------------------------------------------
embedder = None
generator = None


def get_embedder():
    """Load embedding model lazily."""
    global embedder
    if embedder is None:
        print("🔹 Loading embedding model (thenlper/gte-tiny)...")
        embedder = SentenceTransformer("thenlper/gte-tiny")
        print("✅ Embedding model loaded successfully.")
    return embedder


def get_generator():
    """Load text generation model lazily."""
    global generator
    if generator is None:
        print("🔹 Loading generation model (google/flan-t5-base)...")
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        print("✅ Generation model loaded successfully.")
    return generator


# -------------------------------------------------------------------
# 3️⃣  Root & Health endpoints
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "RaviTeja GenAI Gateway is live ✅",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# -------------------------------------------------------------------
# 4️⃣  Text Generation Endpoint
# -------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_text(req: GenerateRequest):
    try:
        gen = get_generator()
        result = gen(req.prompt, max_length=256, do_sample=True)
        return {"prompt": req.prompt, "response": result[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


# -------------------------------------------------------------------
# 5️⃣  PDF Upload & Embedding Endpoint
# -------------------------------------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, and embed it using HuggingFace model."""
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        pdf = PdfReader(file.file)
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")

        model = get_embedder()
        embeddings = model.encode([text])
        doc_id = str(uuid.uuid4())

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "text_length": len(text),
            "embedding_shape": str(embeddings.shape),
            "preview": text[:500]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# -------------------------------------------------------------------
# 6️⃣  Entrypoint for local testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
