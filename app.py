import os
import sys
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------------------------------------------------
# 1️⃣  Environment Setup (Render-specific)
# -------------------------------------------------------------------
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "SENTENCE_TRANSFORMERS_HOME": "/tmp",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false"
})

# Load Hugging Face token from Render environment (DO NOT hard-code)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("⚠️  Warning: HUGGINGFACEHUB_API_TOKEN not found — "
          "model download may fail if private or rate-limited.")

# -------------------------------------------------------------------
# 2️⃣  Initialize FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="RaviTeja GenAI Gateway",
    version="1.0.0",
    description="FastAPI app deployed on Render integrating Hugging Face models with lazy loading."
)

# -------------------------------------------------------------------
# 3️⃣  Lazy model loading helpers
# -------------------------------------------------------------------
embedder = None
generator = None


def get_embedder():
    """Load embedding model once (lazy)."""
    global embedder
    if embedder is None:
        print("🔹 Loading embedding model...")
        embedder = SentenceTransformer(
            "thenlper/gte-tiny",
            use_auth_token=hf_token  # safe: uses env var, not hard-coded
        )
        print("✅ Embedding model loaded successfully.")
    return embedder


def get_generator():
    """Load text generation model once (lazy)."""
    global generator
    if generator is None:
        print("🔹 Loading generation model...")
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            token=hf_token
        )
        print("✅ Generation model loaded successfully.")
    return generator


# -------------------------------------------------------------------
# 4️⃣  Root & Health Routes
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
# 5️⃣  Text Generation Endpoint
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
# 6️⃣  PDF Upload & Embedding Endpoint
# -------------------------------------------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF → extract text → embed with Hugging Face model."""
    try:
        # ✅ Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        # ✅ Extract text
        pdf = PdfReader(file.file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")

        # ✅ Embed text using Hugging Face model
        model = get_embedder()
        embeddings = model.encode([text])

        doc_id = str(uuid.uuid4())

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "text_length": len(text),
            "embedding_shape": str(embeddings.shape),
            "preview": text[:1000]
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# -------------------------------------------------------------------
# 7️⃣  Application entrypoint for local testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
