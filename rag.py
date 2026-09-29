"""RAG pipeline: YouTube URL -> transcript -> chunks -> Gemini embeddings -> FAISS -> Gemini answer."""

import bisect
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(Exception):
    """Raised with a user-friendly message when a transcript can't be loaded."""


@dataclass
class Transcript:
    video_id: str
    language: str
    text: str
    # (char offset in `text`, start time in seconds) for each caption snippet
    offsets: list[tuple[int, float]]


def extract_video_id(url: str) -> str:
    """Pull the 11-char video ID out of any common YouTube URL format (or a bare ID)."""
    url = url.strip()
    if VIDEO_ID_RE.match(url):
        return url

    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")

    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host == "youtube-nocookie.com":
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = parts[1]

    if candidate and VIDEO_ID_RE.match(candidate):
        return candidate
    raise TranscriptError("That doesn't look like a valid YouTube video URL.")


def fetch_transcript(video_id: str) -> Transcript:
    """Fetch captions, preferring English but falling back to any available language."""
    api = YouTubeTranscriptApi()
    try:
        transcripts = api.list(video_id)
        try:
            chosen = transcripts.find_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            chosen = next(iter(transcripts))
        fetched = chosen.fetch()
    except TranscriptsDisabled:
        raise TranscriptError("Captions are disabled for this video.")
    except VideoUnavailable:
        raise TranscriptError("This video is unavailable (private, removed, or region-locked).")
    except (NoTranscriptFound, StopIteration):
        raise TranscriptError("No transcript is available for this video.")
    except YouTubeTranscriptApiException as e:
        raise TranscriptError(f"Couldn't fetch the transcript: {type(e).__name__}")

    parts: list[str] = []
    offsets: list[tuple[int, float]] = []
    pos = 0
    for snippet in fetched.snippets:
        piece = " ".join(snippet.text.split())
        if not piece:
            continue
        offsets.append((pos, snippet.start))
        parts.append(piece)
        pos += len(piece) + 1  # +1 for the joining space

    if not parts:
        raise TranscriptError("The transcript for this video is empty.")
    return Transcript(video_id, chosen.language, " ".join(parts), offsets)


def _time_at(offsets: list[tuple[int, float]], char_index: int) -> float:
    positions = [p for p, _ in offsets]
    i = max(bisect.bisect_right(positions, char_index) - 1, 0)
    return offsets[i][1]


def split_transcript(transcript: Transcript) -> list[Document]:
    """Chunk the transcript and tag each chunk with the timestamp where it starts."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    docs = splitter.create_documents([transcript.text])
    for doc in docs:
        doc.metadata["start_seconds"] = _time_at(transcript.offsets, doc.metadata["start_index"])
        doc.metadata["video_id"] = transcript.video_id
    return docs


def build_vector_store(docs: list[Document], api_key: str) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about a YouTube video "
        "using ONLY the transcript excerpts provided below. Each excerpt is prefixed "
        "with its timestamp in [m:ss] form.\n\n"
        "Rules:\n"
        "- If the excerpts don't contain the answer, say you couldn't find it in the video. "
        "Do not make things up or use outside knowledge.\n"
        "- Cite the timestamp(s) you relied on, e.g. (at 3:42).\n"
        "- Answer in the same language the user asks in.\n\n"
        "Transcript excerpts:\n{context}",
    ),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])


def answer_question(
    store: FAISS, question: str, history: list[tuple[str, str]], api_key: str
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and ask Gemini. Returns (answer, source docs)."""
    docs = store.similarity_search(question, k=TOP_K)
    docs.sort(key=lambda d: d.metadata["start_seconds"])
    context = "\n\n".join(
        f"[{format_timestamp(d.metadata['start_seconds'])}] {d.page_content}" for d in docs
    )

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.2)
    chain = PROMPT | llm
    response = chain.invoke({"context": context, "history": history, "question": question})
    return response.text, docs
