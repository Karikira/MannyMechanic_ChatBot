# Manny v4 — AI Auto Repair Assistant
### Groq + Streamlit — Free Public Deployment

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
# Add your Groq key:
echo 'GROQ_API_KEY = "gsk_your_key_here"' > .streamlit/secrets.toml
streamlit run streamlit_app.py
```

---

## Deploy Free on Streamlit Cloud

### Step 1 — Get a free Groq API key
1. Go to **https://console.groq.com**
2. Sign up (free)
3. Click **API Keys → Create API Key**
4. Copy the key (starts with `gsk_...`)

### Step 2 — Push to GitHub
```bash
git init
git add .
git commit -m "Manny v4 - Groq + Streamlit"
git remote add origin https://github.com/YOUR_USERNAME/manny-v4.git
git push -u origin main
```
> ⚠️ The `.gitignore` ensures `secrets.toml` is **never** pushed to GitHub.

### Step 3 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch `main` → main file: `streamlit_app.py`
5. Click **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Click **Deploy** — done! 🎉

Your app will be live at:
`https://your-username-manny-v4-streamlit-app-xxxxxx.streamlit.app`

---

## What changed from v3 (Flask + Ollama)

| Feature | v3 | v4 |
|---|---|---|
| UI | Flask + HTML/JS | **Streamlit** |
| Text LLM | Ollama llama3.2 (local) | **Groq llama-3.1-8b-instant** |
| Vision LLM | Ollama llama3.2-vision (local) | **Groq llama-3.2-11b-vision** |
| Vector DB | ChromaDB persistent (local) | **ChromaDB in-memory** |
| Deployment | Your laptop only | **Free public URL** |
| Cost | Free (local) | **Free (Groq free tier)** |

All agent logic (diagnostic flow, car lookup, RAG, agent.yaml config) is identical to v3.

---

## Groq Free Tier Limits
- **30 requests/minute** — plenty for a single-user chatbot
- **14,400 requests/day**
- No credit card required

## File Structure
```
manny_streamlit/
├── streamlit_app.py      ← Main app (replaces app.py + index.html)
├── config.py             ← Config loader (same as v3)
├── rag.py                ← RAG pipeline (in-memory ChromaDB)
├── agent.yaml            ← All settings (edit this, not the code)
├── requirements.txt      ← Python dependencies
├── .gitignore            ← Keeps secrets.toml out of GitHub
├── .streamlit/
│   └── secrets.toml      ← Your Groq API key (DO NOT commit)
└── knowledge_base/
    ├── shop_services.txt
    └── car_repair_guide.txt
```
