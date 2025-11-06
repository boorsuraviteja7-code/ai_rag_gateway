# app.py
import os
import sys
from typing import Optional

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
# ✅ Optional: Prevent heavy torch imports during Render health probe
# ------------------------------------------------------------
if os.environ.get("RENDER") == "true":
    sys.modules["torch"] = __import__("torch", fromlist=[""])

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

from fastapi import FastAPI
from pydantic import BaseModel

# ------------------------------------------------------------
# ✅ Initialize FastAPI
# ------------------------------------------------------------
app = FastAPI(
    title="RaviTeja GenAI Gateway",
    description="FastAPI app deployed on Render with HuggingFace lazy loading",
    version="1.0.0"
)

# ------------------------------------------------------------
# ✅ FIX 2: Root and Health Routes (Render health probe fix)
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


# ------------------------------------------------------------
# ✅ FIX 3: Lazy-load HuggingFace model (loaded on first request only)
# ------------------------------------------------------------
_model = None
_tokenizer = None

def load_model_once():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    # Import heavy dependencies only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.getenv("HF_MODEL_NAME", "sshleifer/tiny-gpt2")

    print(f"🔄 Loading HuggingFace model: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(model_name)

    print("✅ Model loaded successfully")
    return _model, _tokenizer


# ------------------------------------------------------------
# ✅ Example Text Generation Route
# ------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30

@app.post("/generate")
def generate_text(request: GenerateRequest):
    model, tokenizer = load_model_once()

    # Handle environments without torch gracefully
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
        "output": result_text,
        "tokens_generated": request.max_new_tokens
    }


# ------------------------------------------------------------
# ✅ Utility for environments missing torch context manager
# ------------------------------------------------------------
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


# ------------------------------------------------------------
# ✅ Final log confirmation for Render startup
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RaviTeja GenAI Gateway on port 10000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
