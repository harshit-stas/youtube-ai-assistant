import os

import streamlit as st
from dotenv import load_dotenv

from rag import (
    TranscriptError,
    answer_question,
    build_vector_store,
    extract_video_id,
    fetch_transcript,
    format_timestamp,
    split_transcript,
)

load_dotenv()

st.set_page_config(page_title="YouTube AI Assistant", page_icon="🎥", layout="centered")
st.title("🎥 YouTube AI Assistant")
st.caption("Paste a YouTube link, then ask questions about the video.")

MAX_HISTORY_TURNS = 6


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:  # no secrets.toml present
        key = None
    return key or os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def load_video(video_id: str, api_key: str):
    """Transcript -> chunks -> FAISS index, cached per video so reruns are instant."""
    transcript = fetch_transcript(video_id)
    docs = split_transcript(transcript)
    store = build_vector_store(docs, api_key)
    return store, transcript.language, len(docs)


api_key = get_api_key()
if not api_key:
    st.error(
        "No Gemini API key found. Add `GOOGLE_API_KEY` to a `.env` file "
        "(or `.streamlit/secrets.toml`) and restart. Get a key at https://aistudio.google.com/apikey"
    )
    st.stop()

with st.sidebar:
    st.header("Video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    load_clicked = st.button("Load video", type="primary", use_container_width=True)

    if load_clicked and url:
        try:
            video_id = extract_video_id(url)
            with st.spinner("Fetching transcript and building index..."):
                store, language, n_chunks = load_video(video_id, api_key)
            st.session_state.update(
                video_id=video_id, language=language, n_chunks=n_chunks, messages=[]
            )
        except TranscriptError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong while processing the video: {e}")

    if "video_id" in st.session_state:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        st.caption(
            f"Transcript language: `{st.session_state.language}` · "
            f"{st.session_state.n_chunks} chunks indexed"
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if "video_id" not in st.session_state:
    st.info("👈 Paste a YouTube URL in the sidebar and click **Load video** to start.")
    st.stop()

video_id = st.session_state.video_id
store, _, _ = load_video(video_id, api_key)
messages = st.session_state.setdefault("messages", [])


def render_sources(sources: list[dict]) -> None:
    with st.expander("Sources"):
        for src in sources:
            ts = int(src["start"])
            link = f"https://www.youtube.com/watch?v={video_id}&t={ts}s"
            st.markdown(f"**[{format_timestamp(ts)}]({link})** — {src['text'][:300]}…")


for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if question := st.chat_input("Ask something about the video..."):
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in messages[-MAX_HISTORY_TURNS * 2 :]
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, docs = answer_question(store, question, history, api_key)
            sources = [
                {"start": d.metadata["start_seconds"], "text": d.page_content} for d in docs
            ]
            st.markdown(answer)
            render_sources(sources)
        except Exception as e:
            answer, sources = f"Sorry, I couldn't answer that: {e}", []
            st.error(answer)

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer, "sources": sources})
