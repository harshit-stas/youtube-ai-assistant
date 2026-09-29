# 🎥 YouTube AI Assistant

Chat with any YouTube video. Paste a URL, and the app pulls the transcript, indexes it in a FAISS vector store with Gemini embeddings, and answers your questions with Gemini 3.8 Flash, citing clickable timestamps back to the video.

## How it works

```
YouTube URL → Transcript → Chunking → Gemini Embeddings → FAISS → Similarity Search → Gemini 3.8 Flash → Answer + timestamps
```

- **Transcript:** `youtube-transcript-api`. Prefers English and falls back to any available caption language.
- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap). Each chunk keeps the timestamp where it starts.
- **Embeddings:** `gemini-embedding-001`
- **Vector store:** FAISS (in memory, cached per video)
- **LLM:** `gemini-3.8-flash`, which answers only from the retrieved excerpts, cites timestamps, and uses recent chat history for follow-up questions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your key in .env
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (`.env` is git-ignored).
2. Create an app at https://share.streamlit.io pointing to `app.py`.
3. Under **Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-key"
   ```

> **Note:** YouTube often blocks transcript requests from cloud-provider IPs, so the deployed app may fail on some videos even though it works locally. See the [youtube-transcript-api docs](https://github.com/jdepoix/youtube-transcript-api#working-around-ip-bans-requestblocked-or-ipblocked-exception) for proxy workarounds.

## Project structure

```
app.py        Streamlit UI (sidebar video loader + chat)
rag.py        RAG pipeline: transcript, chunking, embeddings, retrieval, answering
```
