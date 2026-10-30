import os
import sys
import time
from pathlib import Path
import streamlit as st

# ── Backend Modules ────────────────────────────────────────────────────────────
from src.video_processor import save_uploaded_video
from src.youtube_processor import download_youtube_video
from src.audio_processor import extract_audio
from src.transcription import transcribe_audio
from src.chunking import chunk_text
from src.vector_store import create_vector_store, COLLECTION_NAME
from src.embedding import embeddings
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain, retrieve_with_sources
from langchain_chroma import Chroma

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Intelligence AI | Multimodal Video Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper: Format seconds to HH:MM:SS ─────────────────────────────────────────
def format_seconds(seconds: float) -> str:
    if seconds is None:
        return "00:00"
    try:
        seconds = float(seconds)
    except (ValueError, TypeError):
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ── Custom CSS Design System ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #07090e;
        --bg-panel: rgba(17, 24, 39, 0.72);
        --bg-panel-hover: rgba(30, 41, 59, 0.82);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-focus: rgba(99, 102, 241, 0.55);
        --accent-primary: #6366f1;
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        --accent-glow: 0 0 25px rgba(99, 102, 241, 0.35);
        --emerald-accent: #10b981;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* ── Global Typography & Background ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-main);
        background-image:
            radial-gradient(ellipse 90% 50% at 50% -15%, rgba(99, 102, 241, 0.16) 0%, transparent 65%),
            radial-gradient(ellipse 65% 40% at 90% 90%, rgba(16, 185, 129, 0.08) 0%, transparent 60%),
            radial-gradient(ellipse 50% 30% at 10% 80%, rgba(217, 70, 239, 0.06) 0%, transparent 55%);
        background-attachment: fixed;
    }

    /* ── Streamlit Header / Chrome Cleanup ── */
    #MainMenu, header, footer { visibility: hidden; }
    .block-container {
        padding: 1.5rem 2.2rem 3rem !important;
        max-width: 1400px;
    }

    /* ── Top Hero Banner ── */
    .hero-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        margin-bottom: 1.5rem;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .brand-logo-badge {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: var(--accent-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }

    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 20%, #c7d2fe 60%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .brand-subtitle {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-top: 0.2rem;
    }

    .telemetry-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        align-items: center;
    }

    .pill-badge {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-subtle);
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.74rem;
        font-weight: 500;
        color: var(--text-secondary);
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }

    .pill-badge:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.16);
    }

    .pulse-dot-green {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 10px #10b981;
        display: inline-block;
        animation: pulseGreen 2s infinite ease-in-out;
    }

    .pulse-dot-purple {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #a855f7;
        box-shadow: 0 0 10px #a855f7;
        display: inline-block;
        animation: pulseGreen 2s infinite ease-in-out;
    }

    @keyframes pulseGreen {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }

    .card-header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.9rem;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1.2rem !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
    }

    [data-testid="stMetricLabel"] p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.76rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.1px !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.7rem !important;
        color: #ffffff !important;
    }

    /* ── Custom Styled Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.55);
        padding: 0.45rem;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.55rem 1.25rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-secondary);
        border: none;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.6rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.22s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Secondary / outline buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: none !important;
        color: var(--text-secondary) !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* ── Text Inputs & Textareas ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        background: rgba(15, 23, 42, 0.95) !important;
    }

    /* ── Modern Chat Input Bar ── */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        background: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.28), 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader > div {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(99, 102, 241, 0.4) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.25s ease !important;
    }

    .stFileUploader > div:hover {
        border-color: rgba(99, 102, 241, 0.85) !important;
        background: rgba(99, 102, 241, 0.05) !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.35) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.2rem !important;
    }

    /* ── Step Pipeline Card ── */
    .pipeline-stage {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    .pipeline-stage.stage-done {
        border-color: rgba(16, 185, 129, 0.3);
        background: rgba(16, 185, 129, 0.05);
    }

    .pipeline-stage.stage-active {
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(99, 102, 241, 0.08);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.15);
    }

    .stage-icon {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        flex-shrink: 0;
    }

    .icon-done { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .icon-active { background: rgba(99, 102, 241, 0.25); color: #818cf8; animation: pulseIcon 1.5s infinite; }
    .icon-pending { background: rgba(255, 255, 255, 0.05); color: #64748b; }

    @keyframes pulseIcon {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    .stage-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #f1f5f9;
    }

    .stage-desc {
        font-size: 0.75rem;
        color: var(--text-muted);
    }

    /* ── Citation Cards in Chat ── */
    .citation-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.6rem;
        margin-bottom: 0.5rem;
        font-size: 0.84rem;
    }

    .citation-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        background: rgba(99, 102, 241, 0.18);
        color: #a5b4fc;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.35rem;
    }

    /* ── Transcript Timeline Snippets ── */
    .transcript-row {
        display: flex;
        gap: 1rem;
        padding: 0.65rem 0.85rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        transition: background 0.15s ease;
    }
    .transcript-row:hover {
        background: rgba(255, 255, 255, 0.035);
    }
    .transcript-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        color: #818cf8;
        background: rgba(99, 102, 241, 0.12);
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        align-self: flex-start;
        white-space: nowrap;
    }
    .transcript-text {
        font-size: 0.86rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.6); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Management ───────────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []          # list of dict(role, text, sources, timestamp)
if "processed_videos" not in st.session_state:
    st.session_state["processed_videos"] = []      # unique list of video titles/filenames
if "total_chunks" not in st.session_state:
    st.session_state["total_chunks"] = 0
if "current_video" not in st.session_state:
    st.session_state["current_video"] = None
if "transcripts" not in st.session_state:
    st.session_state["transcripts"] = {}          # video_name -> list of segment dicts
if "video_scope" not in st.session_state:
    st.session_state["video_scope"] = "All Videos"
if "auto_loaded" not in st.session_state:
    st.session_state["auto_loaded"] = False

# ── Auto-Connect to Existing Knowledge Base in ChromaDB ─────────────────────────
def auto_sync_knowledge_base():
    """Detects and loads existing indexed vectors from ChromaDB on startup."""
    try:
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory="chroma_db",
        )
        col = vs._collection
        count = col.count()
        if count > 0:
            res = col.get()
            raw_metas = res.get("metadatas", [])
            existing_videos = sorted(
                list(set(m.get("video_name") for m in raw_metas if m and "video_name" in m))
            )
            st.session_state["vector_store"] = vs
            st.session_state["total_chunks"] = count
            st.session_state["processed_videos"] = existing_videos

            # Build default retriever and chain
            retriever = get_retriever(vs, current_video=None)
            st.session_state["retriever"] = retriever
            st.session_state["rag_chain"] = create_rag_chain(retriever)
            return True, count, len(existing_videos)
    except Exception as e:
        return False, 0, 0
    return False, 0, 0

if not st.session_state["auto_loaded"]:
    success, chunk_cnt, vid_cnt = auto_sync_knowledge_base()
    st.session_state["auto_loaded"] = True

# ── Helper to retrieve transcript segments from Chroma if needed ───────────────
def get_transcript_for_video(video_name: str):
    """Retrieves segment snippets for a video directly from session state or ChromaDB."""
    if video_name in st.session_state["transcripts"]:
        return st.session_state["transcripts"][video_name]

    if st.session_state["vector_store"]:
        try:
            col = st.session_state["vector_store"]._collection
            res = col.get(where={"video_name": video_name})
            if res and res.get("documents"):
                segments = []
                for doc, meta in zip(res["documents"], res["metadatas"]):
                    segments.append({
                        "text": doc,
                        "start": meta.get("start_time", 0.0),
                        "end": meta.get("end_time", 0.0),
                        "chunk_id": meta.get("chunk_id", 0),
                    })
                segments.sort(key=lambda s: (s.get("start", 0), s.get("chunk_id", 0)))
                st.session_state["transcripts"][video_name] = segments
                return segments
        except Exception:
            pass
    return []

# ── Visual Pipeline Renderer ───────────────────────────────────────────────────
def render_pipeline_card(steps: list[dict]) -> str:
    """Renders high-tech animated pipeline progress."""
    html = '<div class="glass-card" style="padding:1rem 1.25rem;">'
    html += '<div style="font-size:0.75rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#818cf8;margin-bottom:0.75rem;">'
    html += '⚡ LIVE PROCESSING ENGINE</div>'

    for step in steps:
        status = step["status"]
        if status == "done":
            cls = "stage-done"
            icon_cls = "icon-done"
            icon = "✓"
            badge = '<span style="color:#10b981;font-weight:600;font-size:0.75rem;margin-left:auto;">COMPLETED</span>'
        elif status == "active":
            cls = "stage-active"
            icon_cls = "icon-active"
            icon = "⏳"
            badge = '<span style="color:#818cf8;font-weight:600;font-size:0.75rem;margin-left:auto;">PROCESSING…</span>'
        elif status == "error":
            cls = "stage-active"
            icon_cls = "icon-active"
            icon = "✕"
            badge = '<span style="color:#ef4444;font-weight:600;font-size:0.75rem;margin-left:auto;">FAILED</span>'
        else:
            cls = ""
            icon_cls = "icon-pending"
            icon = "•"
            badge = '<span style="color:#64748b;font-weight:500;font-size:0.75rem;margin-left:auto;">QUEUED</span>'

        html += f"""
        <div class="pipeline-stage {cls}">
            <div class="stage-icon {icon_cls}">{icon}</div>
            <div>
                <div class="stage-title">{step['title']}</div>
                <div class="stage-desc">{step['desc']}</div>
            </div>
            {badge}
        </div>
        """
    html += '</div>'
    return html

# ── Processing Execution ───────────────────────────────────────────────────────
def execute_pipeline(file_path: str, progress_placeholder):
    steps = [
        {"title": "Audio Extraction", "desc": "Isolating mono audio stream via FFmpeg (16kHz)", "status": "active"},
        {"title": "Whisper Speech-to-Text", "desc": "Timestamped transcription via Groq Whisper-v3", "status": "pending"},
        {"title": "Recursive Chunking", "desc": "Semantic text splitting with overlap preservation", "status": "pending"},
        {"title": "Gemini Embeddings & Indexing", "desc": "Dense vector generation & ChromaDB persistent commit", "status": "pending"},
        {"title": "RAG Chain Construction", "desc": "Assembling MMR retriever & Gemini 3 Flash synthesis chain", "status": "pending"},
    ]

    def update_ui(idx, new_status):
        steps[idx]["status"] = new_status
        if new_status == "done" and idx + 1 < len(steps):
            steps[idx + 1]["status"] = "active"
        progress_placeholder.markdown(render_pipeline_card(steps), unsafe_allow_html=True)

    progress_placeholder.markdown(render_pipeline_card(steps), unsafe_allow_html=True)

    # 1. Audio
    audio_path = extract_audio(file_path)
    update_ui(0, "done")

    # 2. Transcription
    transcript = transcribe_audio(audio_path)
    update_ui(1, "done")

    # 3. Chunking
    chunks = chunk_text(transcript)
    update_ui(2, "done")

    # 4. Vector Store
    video_name = Path(file_path).name
    st.session_state["current_video"] = video_name
    vector_store = create_vector_store(chunks, video_name)
    st.session_state["vector_store"] = vector_store
    update_ui(3, "done")

    # 5. RAG Chain
    # Default to cross-video or current video scope
    retriever = get_retriever(vector_store, current_video=None)
    rag_chain = create_rag_chain(retriever)
    update_ui(4, "done")

    # Update session memory
    st.session_state["retriever"] = retriever
    st.session_state["rag_chain"] = rag_chain
    if video_name not in st.session_state["processed_videos"]:
        st.session_state["processed_videos"].append(video_name)
    st.session_state["total_chunks"] = vector_store._collection.count()
    st.session_state["transcripts"][video_name] = transcript

    return transcript, chunks

# ── TOP HERO & TELEMETRY BAR ───────────────────────────────────────────────────
processed_count = len(st.session_state["processed_videos"])
total_chunks = st.session_state["total_chunks"]

st.markdown(
    f"""
    <div class="hero-container">
        <div class="brand-wrap">
            <div class="brand-logo-badge">⚡</div>
            <div>
                <h1 class="brand-title">Video Intelligence AI</h1>
                <div class="brand-subtitle">Enterprise Multimodal Video Intelligence & Cross-Video RAG Engine</div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="pill-badge">
                <span class="pulse-dot-green"></span>
                <span>System <strong>Active</strong></span>
            </div>
            <div class="pill-badge">
                <span>🧠 LLM: <strong>Gemini 3 Flash</strong></span>
            </div>
            <div class="pill-badge">
                <span>🎙 Audio: <strong>Whisper Large-v3</strong></span>
            </div>
            <div class="pill-badge">
                <span class="pulse-dot-purple"></span>
                <span>ChromaDB: <strong>{total_chunks} Chunks</strong></span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR: Knowledge Base & Scope Control ────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">
            <span style="font-size:1.3rem;">🗂</span>
            <span style="font-family:'Outfit';font-weight:700;font-size:1.15rem;color:#f8fafc;">Knowledge Hub</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scope Selection
    scope_options = ["🌐 All Indexed Videos (Cross-Video RAG)"] + st.session_state["processed_videos"]
    selected_scope = st.selectbox(
        "Query Retrieval Scope",
        options=scope_options,
        index=0,
        help="Select 'All Videos' to search across your entire repository, or focus on a single video."
    )

    # Dynamic scope switching
    is_cross_video = selected_scope.startswith("🌐")
    active_target = None if is_cross_video else selected_scope

    if st.session_state["vector_store"] is not None:
        # Re-scope retriever if target changed
        new_retriever = get_retriever(st.session_state["vector_store"], current_video=active_target)
        st.session_state["retriever"] = new_retriever
        st.session_state["rag_chain"] = create_rag_chain(new_retriever)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.2rem 0;'>", unsafe_allow_html=True)

    # Processed Videos Registry
    st.markdown(
        f"<div style='font-size:0.76rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#818cf8;margin-bottom:0.6rem;'>"
        f"INDEXED REPOSITORY ({len(st.session_state['processed_videos'])})</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state["processed_videos"]:
        st.markdown(
            "<div style='color:var(--text-muted);font-size:0.82rem;padding:0.75rem;background:rgba(255,255,255,0.02);border-radius:10px;border:1px dashed rgba(255,255,255,0.08);text-align:center;'>"
            "No videos indexed yet.<br>Upload a video or YouTube link to start.</div>",
            unsafe_allow_html=True,
        )
    else:
        for v in st.session_state["processed_videos"]:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:0.55rem 0.75rem;margin-bottom:0.45rem;">
                    <div style="font-size:0.82rem;font-weight:600;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        🎬 {v}
                    </div>
                    <div style="font-size:0.7rem;color:#10b981;margin-top:0.15rem;display:flex;align-items:center;gap:0.35rem;">
                        <span style="width:5px;height:5px;border-radius:50%;background:#10b981;display:inline-block;"></span> Indexed & Ready
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.2rem 0;'>", unsafe_allow_html=True)

    # Quick System Utilities
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        if st.button("🔄 Sync DB", use_container_width=True, help="Reload Chroma vectors from disk"):
            auto_sync_knowledge_base()
            st.rerun()
    with col_u2:
        if st.button("🗑 Reset Chat", use_container_width=True, help="Clear chat messages"):
            st.session_state["chat_history"] = []
            st.rerun()

    # System specs footer
    st.markdown(
        """
        <div style="margin-top:2rem;padding:0.85rem;background:rgba(15,23,42,0.6);border-radius:12px;border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:0.72rem;color:var(--text-muted);line-height:1.6;">
                ⚡ <strong>Retrieval:</strong> MMR Search<br>
                🧬 <strong>Embedding:</strong> text-embedding-001<br>
                📦 <strong>Store:</strong> ChromaDB Persistent
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── KPI METRICS CARDS ──────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Videos Ingested", value=processed_count, delta=f"{processed_count} Ready" if processed_count > 0 else None)
with kpi2:
    st.metric(label="Vector Chunks", value=total_chunks, delta="ChromaDB" if total_chunks > 0 else None)
with kpi3:
    st.metric(label="Chat Interactions", value=len(st.session_state["chat_history"]))
with kpi4:
    scope_name = "Cross-Video" if is_cross_video else "Single-Target"
    st.metric(label="Active Scope", value=scope_name)

st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)

# ── MAIN TABBED WORKSPACE ──────────────────────────────────────────────────────
tab_chat, tab_player, tab_transcript, tab_ingest, tab_vectors = st.tabs([
    "💬 Intelligence Chat",
    "🎬 Video Player & Media",
    "📑 Interactive Transcript",
    "📥 Ingest & Process Hub",
    "🧩 Vector Store Explorer",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: INTELLIGENCE CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    chat_col_left, chat_col_right = st.columns([3.2, 1], gap="medium")

    with chat_col_right:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header-title">💡 Quick Prompts</div>
                <div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.8rem;">
                    Click any quick-start question to query your indexed videos immediately:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preset_prompts = [
            ("📌 Key Takeaways", "Summarize the primary thesis, core ideas, and key takeaways from the video."),
            ("⏱ Topic Timeline", "Provide a detailed chronological timeline of all topics discussed with timestamps."),
            ("❓ Technical Concepts", "What are the fundamental concepts and definitions explained in the video?"),
            ("🎯 Actionable Insights", "What are the practical applications, pros, and cons highlighted?"),
        ]

        for label, p_text in preset_prompts:
            if st.button(label, use_container_width=True, disabled=(st.session_state["rag_chain"] is None)):
                st.session_state["queued_prompt"] = p_text
                st.rerun()

        st.markdown(
            f"""
            <div class="glass-card" style="margin-top:1rem;">
                <div style="font-size:0.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#818cf8;margin-bottom:0.4rem;">
                    CURRENT QUERY SCOPE
                </div>
                <div style="font-size:0.85rem;color:#f8fafc;font-weight:600;">
                    {'🌐 All Indexed Knowledge' if is_cross_video else f'🎬 {active_target}'}
                </div>
                <div style="font-size:0.74rem;color:var(--text-muted);margin-top:0.3rem;">
                    Responses synthesize context strictly from verified video chunks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chat_col_left:
        # Check if RAG is unlocked
        if st.session_state["rag_chain"] is None:
            st.markdown(
                """
                <div style="text-align:center;padding:3.5rem 2rem;background:rgba(15,23,42,0.5);border:1px dashed rgba(99,102,241,0.3);border-radius:18px;">
                    <div style="font-size:2.4rem;margin-bottom:0.8rem;">🎬</div>
                    <div style="font-family:'Outfit';font-size:1.25rem;font-weight:700;color:#f1f5f9;margin-bottom:0.4rem;">
                        No Indexed Videos Found
                    </div>
                    <div style="color:var(--text-secondary);font-size:0.88rem;max-width:440px;margin:0 auto 1.4rem;">
                        To unlock conversational intelligence, upload a video file or paste a YouTube link in the <strong>Ingest Hub</strong> tab.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Chat messages display container
            chat_container = st.container()

            with chat_container:
                if not st.session_state["chat_history"]:
                    st.markdown(
                        f"""
                        <div style="text-align:center;padding:2.8rem 1.5rem;background:rgba(15,23,42,0.45);border:1px dashed rgba(255,255,255,0.09);border-radius:16px;margin-bottom:1rem;">
                            <div style="font-size:2rem;margin-bottom:0.5rem;">✨</div>
                            <div style="font-weight:600;font-size:1rem;color:#f8fafc;margin-bottom:0.25rem;">
                                Video Knowledge Ready for Questions
                            </div>
                            <div style="color:var(--text-muted);font-size:0.84rem;">
                                Ask specific questions, query timestamps, or select a quick prompt from the right.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    for item in st.session_state["chat_history"]:
                        if item["role"] == "user":
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(f"**{item['text']}**")
                        else:
                            with st.chat_message("assistant", avatar="⚡"):
                                st.markdown(item["text"])
                                if item.get("sources"):
                                    with st.expander(f"📍 View {len(item['sources'])} Cited Video Citations & Timestamps", expanded=False):
                                        for s_idx, src in enumerate(item["sources"], 1):
                                            start_fmt = format_seconds(src.get("start"))
                                            end_fmt = format_seconds(src.get("end"))
                                            st.markdown(
                                                f"""
                                                <div class="citation-box">
                                                    <span class="citation-tag">SOURCE #{s_idx} • {src.get('video')}</span>
                                                    <div style="color:#818cf8;font-family:'JetBrains Mono';font-size:0.75rem;margin-bottom:0.3rem;">
                                                        ⏱ Timestamps: {start_fmt} ➔ {end_fmt}
                                                    </div>
                                                    <div style="color:#cbd5e1;font-size:0.82rem;font-style:italic;">
                                                        "{src.get('content', '')}"
                                                    </div>
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )

            # Input logic (supports chat_input or queued preset)
            user_input = st.chat_input("Ask anything about your video knowledge base...")

            active_prompt = None
            if "queued_prompt" in st.session_state and st.session_state["queued_prompt"]:
                active_prompt = st.session_state.pop("queued_prompt")
            elif user_input:
                active_prompt = user_input

            if active_prompt and st.session_state["rag_chain"]:
                # Save user message
                st.session_state["chat_history"].append({
                    "role": "user",
                    "text": active_prompt,
                    "timestamp": time.strftime("%H:%M:%S")
                })

                with st.spinner("⚡ Synthesizing response from video vector chunks..."):
                    try:
                        # 1. Retrieve sources for citations
                        docs, raw_sources = retrieve_with_sources(st.session_state["retriever"], active_prompt)
                        citations = []
                        for d, s in zip(docs, raw_sources):
                            citations.append({
                                "video": s.get("video"),
                                "start": s.get("start"),
                                "end": s.get("end"),
                                "content": d.page_content[:240].strip()
                            })

                        # 2. Invoke RAG chain
                        answer = st.session_state["rag_chain"].invoke({"question": active_prompt})
                        answer_text = str(answer)
                    except Exception as e:
                        answer_text = f"⚠️ Error answering question: {e}"
                        citations = []

                # Save AI message
                st.session_state["chat_history"].append({
                    "role": "ai",
                    "text": answer_text,
                    "sources": citations,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: VIDEO PLAYER & MEDIA HUB
# ═════════════════════════════════════════════════════════════════════════════
with tab_player:
    st.markdown('<div class="card-header-title">🎬 Multimodal Media Viewer</div>', unsafe_allow_html=True)

    available_files = []
    upload_dir = Path("uploads")
    if upload_dir.exists():
        available_files = [f for f in upload_dir.glob("*") if f.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]]

    if not available_files:
        st.info("No video files found in local storage. Upload or download a video in the Ingest tab.")
    else:
        v_selected = st.selectbox(
            "Select Video to Play & Inspect",
            options=[f.name for f in available_files],
            index=0
        )
        selected_path = upload_dir / v_selected

        col_v1, col_v2 = st.columns([2, 1], gap="medium")

        with col_v1:
            st.markdown(f"**Now Playing:** `{v_selected}`")
            try:
                st.video(str(selected_path))
            except Exception as e:
                st.error(f"Cannot preview video: {e}")

        with col_v2:
            st.markdown(
                """
                <div class="glass-card">
                    <div style="font-size:0.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#818cf8;margin-bottom:0.5rem;">
                        MEDIA SPECIFICATIONS
                    </div>
                """,
                unsafe_allow_html=True,
            )

            file_size_mb = selected_path.stat().st_size / (1024 * 1024)
            st.markdown(f"- **Filename:** `{selected_path.name}`")
            st.markdown(f"- **Format:** `{selected_path.suffix.upper()}`")
            st.markdown(f"- **File Size:** `{file_size_mb:.2f} MB`")

            # Check if extracted audio exists
            temp_dir = Path("temp")
            audio_candidate = temp_dir / f"{selected_path.stem}.mp3"
            if audio_candidate.exists():
                st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.8rem 0;'>", unsafe_allow_html=True)
                st.markdown("🔊 **Extracted Mono Audio (16kHz):**")
                st.audio(str(audio_candidate))

            st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: INTERACTIVE TRANSCRIPT
# ═════════════════════════════════════════════════════════════════════════════
with tab_transcript:
    st.markdown('<div class="card-header-title">📑 Timestamped Transcript Navigator</div>', unsafe_allow_html=True)

    if not st.session_state["processed_videos"]:
        st.info("No transcripts available. Ingest a video to view its timestamped transcript.")
    else:
        selected_t_vid = st.selectbox(
            "Select Video Transcript",
            options=st.session_state["processed_videos"],
            key="t_video_sel"
        )

        filter_kw = st.text_input("🔍 Filter transcript by keyword", placeholder="Type a concept, word, or topic...")

        segments = get_transcript_for_video(selected_t_vid)

        if not segments:
            st.warning("No transcript segments found for this video.")
        else:
            if filter_kw:
                filtered_segments = [s for s in segments if filter_kw.lower() in s.get("text", "").lower()]
                st.caption(f"Showing {len(filtered_segments)} of {len(segments)} segments matching '{filter_kw}'")
            else:
                filtered_segments = segments
                st.caption(f"Total Segments: {len(segments)}")

            full_transcript_str = "\n".join(
                f"[{format_seconds(s.get('start'))} - {format_seconds(s.get('end'))}] {s.get('text')}"
                for s in segments
            )

            st.download_button(
                "📥 Download Full Transcript (.txt)",
                data=full_transcript_str,
                file_name=f"{selected_t_vid}_transcript.txt",
                mime="text/plain",
            )

            st.markdown("<div style='max-height:500px;overflow-y:auto;padding-right:0.5rem;'>", unsafe_allow_html=True)
            for seg in filtered_segments[:100]:
                start_str = format_seconds(seg.get("start"))
                end_str = format_seconds(seg.get("end"))
                text = seg.get("text")

                # Highlight keyword if present
                if filter_kw:
                    text_display = text.replace(filter_kw, f"<mark style='background:#f59e0b;color:#000;border-radius:3px;padding:0 2px;'>{filter_kw}</mark>")
                else:
                    text_display = text

                st.markdown(
                    f"""
                    <div class="transcript-row">
                        <div class="transcript-time">{start_str} ➔ {end_str}</div>
                        <div class="transcript-text">{text_display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: INGEST & PROCESS HUB
# ═════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    ingest_col1, ingest_col2 = st.columns([1, 1], gap="large")

    with ingest_col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header-title">📁 Local Video Upload</div>
                <div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.9rem;">
                    Upload high-resolution video files (.MP4, .MOV, .AVI, .MKV). Files are processed locally.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Select or Drag Video Files",
            type=["mp4", "avi", "mkv", "mov", "webm"],
            accept_multiple_files=True,
            help="High-speed ingestion with Whisper audio extraction",
            label_visibility="collapsed",
        )

    with ingest_col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header-title">🌐 YouTube Import Engine</div>
                <div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.9rem;">
                    Paste one or more YouTube video links (one URL per line) to download and extract intelligence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        youtube_input = st.text_area(
            "YouTube URLs",
            placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
            height=110,
            label_visibility="collapsed",
        )

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    start_btn = st.button("🚀 Ingest & Index Videos into Vector Knowledge Base", use_container_width=True)

    if start_btn:
        urls = [u.strip() for u in (youtube_input or "").split("\n") if u.strip()]
        total_items = len(uploaded_files or []) + len(urls)

        if total_items == 0:
            st.warning("Please upload at least one video file or provide a valid YouTube URL.")
        else:
            progress_box = st.empty()

            # Process Uploads
            for vid in (uploaded_files or []):
                st.markdown(f"**Ingesting Local Video:** `{vid.name}`")
                try:
                    fpath = save_uploaded_video(vid)
                    execute_pipeline(fpath, progress_box)
                    st.success(f"✓ Successfully indexed: **{vid.name}**")
                except Exception as ex:
                    st.error(f"Failed processing {vid.name}: {ex}")

            # Process YouTube URLs
            for url in urls:
                st.markdown(f"**Downloading YouTube Stream:** `{url}`")
                try:
                    fpath = download_youtube_video(url)
                    execute_pipeline(fpath, progress_box)
                    st.success(f"✓ Successfully downloaded and indexed: **{url}**")
                except Exception as ex:
                    st.error(f"YouTube processing error: {ex}")

            st.balloons()
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: VECTOR STORE EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab_vectors:
    st.markdown('<div class="card-header-title">🧩 Vector Knowledge Base Inspector</div>', unsafe_allow_html=True)

    if st.session_state["vector_store"] is None:
        st.info("ChromaDB vector store is not currently loaded.")
    else:
        try:
            col = st.session_state["vector_store"]._collection
            total_db_chunks = col.count()

            v_info1, v_info2, v_info3 = st.columns(3)
            with v_info1:
                st.metric("Collection Name", COLLECTION_NAME)
            with v_info2:
                st.metric("Total Indexed Chunks", total_db_chunks)
            with v_info3:
                st.metric("Embedding Model", "gemini-embedding-001")

            st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.2rem 0;'>", unsafe_allow_html=True)

            st.markdown("#### 🔍 Semantic MMR Search Probe")
            probe_query = st.text_input("Test retrieval query", value="What are decision trees?", key="probe_q")
            if probe_query:
                retriever = st.session_state["retriever"]
                if retriever:
                    test_docs = retriever.invoke(probe_query)
                    st.markdown(f"Retrieved **{len(test_docs)} relevant chunks** via MMR:")

                    for d_i, d in enumerate(test_docs[:5], 1):
                        start_f = format_seconds(d.metadata.get("start_time"))
                        end_f = format_seconds(d.metadata.get("end_time"))
                        vid_src = d.metadata.get("video_name", "Unknown")

                        st.markdown(
                            f"""
                            <div class="glass-card" style="margin-bottom:0.75rem;padding:0.9rem 1.1rem;">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                                    <span style="font-family:'Outfit';font-weight:700;color:#818cf8;font-size:0.85rem;">RESULT #{d_i} • {vid_src}</span>
                                    <span style="font-family:'JetBrains Mono';font-size:0.75rem;color:#10b981;background:rgba(16,185,129,0.12);padding:0.15rem 0.5rem;border-radius:4px;">⏱ {start_f} ➔ {end_f}</span>
                                </div>
                                <div style="font-size:0.83rem;color:#cbd5e1;line-height:1.5;">
                                    {d.page_content}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Error inspecting ChromaDB: {e}")
