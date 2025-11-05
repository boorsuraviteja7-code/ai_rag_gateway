{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNCcy9DN9XNniBtqVYRAXs7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/boorsuraviteja7-code/ai_rag_gateway/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q fastapi uvicorn langchain sentence-transformers faiss-cpu pypdf transformers pyngrok\n"
      ],
      "metadata": {
        "id": "wloOtCrEWOfz"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import fastapi, uvicorn, langchain, sentence_transformers, faiss, pypdf, transformers, pyngrok\n",
        "print(\"✅ libs ready\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jH-4x2vbWQii",
        "outputId": "ac9f1d42-6189-4ef2-929a-6e44f086108e"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ libs ready\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from fastapi import FastAPI, UploadFile, File, Form\n",
        "from fastapi.responses import JSONResponse\n",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "from sentence_transformers import SentenceTransformer\n",
        "from pypdf import PdfReader\n",
        "import faiss, numpy as np\n",
        "\n",
        "app = FastAPI(title=\"AI Knowledge Assistant\")\n",
        "\n",
        "# Embeddings model: small, fast, good quality\n",
        "embedder = SentenceTransformer(\"sentence-transformers/all-MiniLM-L6-v2\")\n",
        "\n",
        "index, meta = None, []\n",
        "\n",
        "@app.post(\"/upload\")\n",
        "async def upload(file: UploadFile = File(...)):\n",
        "    \"\"\"Upload a PDF → extract text → split → embed → store in FAISS.\"\"\"\n",
        "    global index, meta\n",
        "    text = \"\"\n",
        "    reader = PdfReader(file.file)\n",
        "    for p in reader.pages:\n",
        "        text += p.extract_text() or \"\"\n",
        "\n",
        "    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)\n",
        "    chunks = splitter.split_text(text)\n",
        "\n",
        "    meta = [{\"text\": c, \"doc\": file.filename} for c in chunks]\n",
        "    vecs = embedder.encode([m[\"text\"] for m in meta], normalize_embeddings=True)\n",
        "\n",
        "    index = faiss.IndexFlatIP(vecs.shape[1])          # cosine via normalized vectors\n",
        "    index.add(np.array(vecs, dtype=\"float32\"))\n",
        "    return {\"chunks\": len(chunks), \"doc\": file.filename}\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3pkSU69IWS0G",
        "outputId": "65f743c8-02d7-4add-aa84-c0f9760a0279"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n",
        "\n",
        "t5_name = \"google/flan-t5-base\"   # compact and reliable model\n",
        "t5_tok  = AutoTokenizer.from_pretrained(t5_name)\n",
        "t5_model= AutoModelForSeq2SeqLM.from_pretrained(t5_name)\n",
        "\n",
        "@app.post(\"/query\")\n",
        "async def query(q: str = Form(...)):\n",
        "    if index is None:\n",
        "        return JSONResponse({\"error\": \"No document uploaded\"}, status_code=400)\n",
        "\n",
        "    # Embed the question\n",
        "    qv = embedder.encode([q], normalize_embeddings=True)\n",
        "    D, I = index.search(np.array(qv, dtype=\"float32\"), 3)\n",
        "\n",
        "    # Retrieve top text chunks\n",
        "    retrieved = [meta[i][\"text\"].replace(\"\\n\", \" \").strip() for i in I[0]]\n",
        "    context = \"\\n\\n\".join(retrieved)\n",
        "\n",
        "    # Prompt FLAN-T5 for concise answer\n",
        "    prompt = (\n",
        "        \"Answer the question using ONLY the context. \"\n",
        "        \"If the answer is not in context, reply exactly: I don't know.\\n\\n\"\n",
        "        f\"Context:\\n{context}\\n\\nQuestion: {q}\\nShort answer:\"\n",
        "    )\n",
        "\n",
        "    inputs  = t5_tok(prompt, return_tensors=\"pt\", truncation=True, max_length=2048)\n",
        "    outputs = t5_model.generate(\n",
        "        **inputs, max_new_tokens=80, num_beams=4, temperature=0.2, early_stopping=True\n",
        "    )\n",
        "    answer = t5_tok.decode(outputs[0], skip_special_tokens=True).strip()\n",
        "\n",
        "    # Add small source snippets\n",
        "    sources = [{\"doc\": meta[i][\"doc\"], \"snippet\": meta[i][\"text\"][:180] + \"…\"} for i in I[0]]\n",
        "    return JSONResponse({\"answer\": answer, \"sources\": sources})\n"
      ],
      "metadata": {
        "id": "skKCNWdvWxyX"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok\n",
        "ngrok.set_auth_token(\"34WWlVo6XgitsqKEjdFyGH6doDk_3ZRLPAgJrPvMLaHrgSguF\")\n",
        "print(\"✅ ngrok token set\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f_EuayTQW3aC",
        "outputId": "1cbfe63c-1224-41c2-e6bd-ef73b33ec188"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ ngrok token set\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok\n",
        "import subprocess, os, signal\n",
        "\n",
        "try: ngrok.kill()\n",
        "except: pass\n",
        "\n",
        "try:\n",
        "    pids = subprocess.check_output(\"lsof -t -i:8001\", shell=True).decode().split()\n",
        "    for pid in pids: os.kill(int(pid), signal.SIGKILL)\n",
        "except: pass\n",
        "\n",
        "print(\"✅ Cleaned old servers and tunnels\")\n"
      ],
      "metadata": {
        "id": "cHS5wk5sW88O",
        "outputId": "394701d4-d911-40bf-825a-7a1a40e11a65",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Cleaned old servers and tunnels\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import threading, uvicorn\n",
        "from pyngrok import ngrok\n",
        "\n",
        "PORT = 8001\n",
        "\n",
        "def run_app():\n",
        "    uvicorn.run(app, host=\"0.0.0.0\", port=PORT, log_level=\"info\")\n",
        "\n",
        "thread = threading.Thread(target=run_app, daemon=True)\n",
        "thread.start()\n",
        "\n",
        "# Random URL (always works)\n",
        "public_url = ngrok.connect(PORT)\n",
        "\n",
        "# Or use your own reserved subdomain (if you have paid plan)\n",
        "# public_url = ngrok.connect(addr=PORT, subdomain=\"ravitejaAIChatbot\")\n",
        "\n",
        "print(\"✅ Your AI API is live:\", public_url.public_url)\n",
        "print(\"🔎 Swagger UI:\", public_url.public_url + \"/docs\")\n"
      ],
      "metadata": {
        "id": "o_3skOnJXAa3",
        "outputId": "4fd73f12-b842-4a8c-ab91-c706183602fa",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "INFO:     Started server process [11569]\n",
            "INFO:     Waiting for application startup.\n",
            "INFO:     Application startup complete.\n",
            "INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Your AI API is live: https://unacclimatised-ingestible-loreen.ngrok-free.dev\n",
            "🔎 Swagger UI: https://unacclimatised-ingestible-loreen.ngrok-free.dev/docs\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "-LtLhDX4Zs2L"
      },
      "execution_count": 7,
      "outputs": []
    }
  ]
}