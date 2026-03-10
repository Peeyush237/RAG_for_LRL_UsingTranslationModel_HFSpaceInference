# LinguaBridge — Deployment Guide

## Architecture

```
User → Next.js (Vercel) → FastAPI (Railway/Render) → Groq API (LLM)
                                                    → HuggingFace Spaces (Translation)
                                                    → FAISS + BM25 (local index)
```

---

## Step 1: Upload LoRA Adapters to HuggingFace

Your LoRA adapters (`it2_en2or_lora_adapter/` and `it2_or2en_lora_adapter/`) need to be hosted on HuggingFace for the Spaces app to load them.

### Prerequisites
```bash
pip install huggingface_hub
```

### Login to HuggingFace
```bash
huggingface-cli login
# Paste your HuggingFace token (get one at https://huggingface.co/settings/tokens)
```

### Upload Adapters
```python
# Run this in Python (or in a Colab notebook)
from huggingface_hub import HfApi

api = HfApi()

# Upload English → Odia adapter
api.create_repo("your-username/it2-en2or-lora-adapter", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="/content/drive/MyDrive/it2_en2or_lora_adapter",  # your local path
    repo_id="your-username/it2-en2or-lora-adapter",
    repo_type="model",
)

# Upload Odia → English adapter
api.create_repo("your-username/it2-or2en-lora-adapter", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="/content/drive/MyDrive/it2_or2en_lora_adapter",  # your local path
    repo_id="your-username/it2-or2en-lora-adapter",
    repo_type="model",
)

print("✅ Both adapters uploaded!")
```

**Replace `your-username`** with your actual HuggingFace username.

### Alternative: Upload via CLI
```bash
# English → Odia adapter
huggingface-cli upload your-username/it2-en2or-lora-adapter ./it2_en2or_lora_adapter --repo-type model

# Odia → English adapter
huggingface-cli upload your-username/it2-or2en-lora-adapter ./it2_or2en_lora_adapter --repo-type model
```

---

## Step 2: Deploy Translation API to HuggingFace Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Owner**: Your username
   - **Space name**: `linguabridge-translate`
   - **SDK**: Gradio
   - **Hardware**: T4 GPU (free)
3. Clone the Space locally:
   ```bash
   git clone https://huggingface.co/spaces/your-username/linguabridge-translate
   ```
4. Copy your files:
   ```bash
   cp linguabridge/hf_spaces/app.py linguabridge-translate/app.py
   cp linguabridge/hf_spaces/requirements.txt linguabridge-translate/requirements.txt
   ```
5. **Edit `app.py`** — Update the adapter paths:
   ```python
   EN_TO_OR_ADAPTER = "your-username/it2-en2or-lora-adapter"
   OR_TO_EN_ADAPTER = "your-username/it2-or2en-lora-adapter"
   ```
6. Push to HuggingFace:
   ```bash
   cd linguabridge-translate
   git add .
   git commit -m "Initial deployment"
   git push
   ```
7. Wait for the Space to build (~5 minutes for first build)
8. Your API URL will be: `https://your-username-linguabridge-translate.hf.space` or `https://huggingface.co/spaces/Peeyush237/linguabridge-translate`

---

## Step 3: Deploy Backend to Railway or Render

### Option A: Render (Free Tier)

1. Push `backend/` to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Root Directory**: `linguabridge/backend`
   - **Environment**: Docker
5. Add environment variables:
   - `GROQ_API_KEY` = your Groq key
   - `HF_SPACES_URL` = `https://your-username-linguabridge-translate.hf.space`
6. Deploy!

### Option B: Railway

1. Go to [railway.app](https://railway.app) → New Project
2. Deploy from GitHub repo
3. Set root directory to `linguabridge/backend`
4. Add the same environment variables
5. Railway auto-detects the Dockerfile

### Verify Backend
```bash
curl https://your-backend-url.onrender.com/api/health
# Should return: {"status":"ok",...}
```

---

## Step 4: Deploy Frontend to Vercel

1. Push `frontend/` to a GitHub repo
2. Go to [vercel.com](https://vercel.com) → Import Project
3. Connect your GitHub repo
4. Set:
   - **Root Directory**: `linguabridge/frontend`
   - **Framework Preset**: Next.js
5. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your backend URL
6. Deploy!

---

## Environment Variables Summary

| Service | Variable | Value |
|---------|----------|-------|
| Backend | `GROQ_API_KEY` | Your Groq API key |
| Backend | `HF_SPACES_URL` | HuggingFace Space URL |
| Backend | `CORS_ORIGINS` | `["https://your-app.vercel.app"]` |
| Frontend | `NEXT_PUBLIC_API_URL` | Backend URL |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HF Space cold start (30-60s) | First request after inactivity is slow. Add a keep-alive ping or upgrade to HF Pro. |
| Render free tier sleeps | Free services sleep after 15 min inactivity. First request wakes it (~30s). |
| CORS errors | Make sure `CORS_ORIGINS` includes your Vercel URL exactly |
| Translation timeout | HF Space may be building. Check Space logs at huggingface.co |
