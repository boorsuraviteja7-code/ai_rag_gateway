# app.py
import os
import sys
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# ------------------------------------------------------------
# ✅ FIX 1: Safe startup environment variables for Render
# ------------------------------------------------------------
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "SENTENCE_TRANSFORMERS_HOME": "/tmp",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
})

# ------------------------------------------------------------
# ✅ Optional: prevent heavy imports before Render health check
# ------------------------------------------------------------
if os.environ.get("RENDER") == "true":
    sys.modules["torch"] = __import__("torch", fromlist=[""])

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

# ------------------------------------------------------------
# ✅ Initialize FastAPI
# ------------------------------------------------------------
app = FastAPI(
    title="RaviTeja GenAI Gateway",
    description="FastAPI app deployed on Render with HuggingFace lazy loading + PDF support",
    version="1.1.0"
)

# ------------------------------------------------------------
# ✅ FIX 2: Root + Health routes
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running ✅",
        "docs": "/docs",
        "health": "/health",
        "message": "RaviTeja GenAI Gateway is live!"
    }

@app.get("/health")
def health():
    return {"ok": True, "message": "Healthy 💚"}

@app.head("/")
def head_root():
    return {"status": "ok"}

# ------------------------------------------------------------
# ✅ FIX 3: Lazy-load HuggingFace model
# ------------------------------------------------------------
_model = None
_tokenizer = None

def load_model_once():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.getenv("HF_MODEL_NAME", "sshleifer/tiny-gpt2")
    print(f"🔄 Loading HuggingFace model: {model_name}")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(model_name)

    print("✅ Model loaded successfully")
    return _model, _tokenizer


# ------------------------------------------------------------
# ✅ Text generation endpoint
# ------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.post("/generate")
def generate_text(request: GenerateRequest):
    model, tokenizer = load_model_once()
    if torch is None:
        raise RuntimeError("Torch not available — please ensure CPU wheel is installed.")

    inputs = tokenizer(request.prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False
        )

    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "input": request.prompt,
        "output": result_text.strip(),
        "tokens_generated": request.max_new_tokens
    }


# ------------------------------------------------------------
# ✅ NEW FEATURE: PDF Upload & Text Extraction
# ------------------------------------------------------------
from PyPDF2 import PdfReader

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), summarize: bool = True):
    """
    Upload a PDF, extract text, and optionally summarize using the model.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file."}

    # Read the PDF in-memory
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        return {"error": "No readable text found in PDF."}

    result = {
        "filename": file.filename,
        "text_length": len(text),
        "preview": text[:1000]
    }

    # Optional: summarize with model
    if summarize:
        prompt = f"Summarize this text:\n{text[:1500]}"
        model, tokenizer = load_model_once()

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=120)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result["summary"] = summary.strip()

    return result


# ------------------------------------------------------------
# ✅ Helper for no-torch environments
# ------------------------------------------------------------
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


# ------------------------------------------------------------
# ✅ Final startup message
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RaviTeja GenAI Gateway on port 10000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
