"""
app.py — AstroRAG Streamlit frontend.

Run: streamlit run ui/app.py
"""
import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AstroRAG — Space Mission Intelligence",
    page_icon="🚀",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────────── #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ─────────────────────────────────────────────────────────── #
with st.sidebar:
    st.title("🚀 AstroRAG")
    st.caption("Space Mission Intelligence Assistant")
    st.divider()

    st.subheader("📥 Ingest Data")
    tab_nasa, tab_arxiv, tab_file = st.tabs(["NASA API", "arXiv", "Upload PDF"])

    with tab_nasa:
        apod_count = st.slider("APOD entries", 10, 100, 30)
        if st.button("Ingest NASA Data", type="primary"):
            with st.spinner("Ingesting NASA data…"):
                r = requests.post(f"{API_BASE}/ingest/nasa?apod_count={apod_count}")
                if r.ok:
                    d = r.json()
                    st.success(f"✅ {d['chunks_created']} chunks indexed in {d['elapsed_seconds']}s")
                else:
                    st.error(r.json().get("detail", "Error"))

    with tab_arxiv:
        arxiv_query = st.text_input("arXiv query", "NASA ESA space mission")
        arxiv_max = st.slider("Max papers", 5, 50, 15)
        if st.button("Ingest arXiv Papers", type="primary"):
            with st.spinner("Fetching papers…"):
                r = requests.post(
                    f"{API_BASE}/ingest/arxiv",
                    params={"query": arxiv_query, "max_results": arxiv_max},
                )
                if r.ok:
                    d = r.json()
                    st.success(f"✅ {d['chunks_created']} chunks from {d['raw_documents']} papers")
                else:
                    st.error(r.json().get("detail", "Error"))

    with tab_file:
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded and st.button("Ingest PDF"):
            with st.spinner("Processing PDF…"):
                r = requests.post(
                    f"{API_BASE}/ingest/file",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
                if r.ok:
                    d = r.json()
                    st.success(f"✅ {d['chunks_created']} chunks from {uploaded.name}")
                else:
                    st.error(r.json().get("detail", "Error"))

    st.divider()
    st.subheader("🔬 Tools")

    with st.expander("Classify Document"):
        classify_text = st.text_area("Paste document excerpt", height=100)
        if st.button("Classify"):
            if classify_text:
                r = requests.post(f"{API_BASE}/classify", json={"text": classify_text})
                if r.ok:
                    d = r.json()
                    st.metric("Mission type", d["top_label"])
                    st.caption(f"Confidence: {d['top_score']:.1%}")

    with st.expander("Summarize Document"):
        summarize_text = st.text_area("Paste document text", height=100, key="sum")
        if st.button("Summarize"):
            if summarize_text:
                with st.spinner("Summarizing…"):
                    r = requests.post(f"{API_BASE}/summarize", json={"text": summarize_text})
                    if r.ok:
                        d = r.json()
                        st.json(d)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        requests.delete(f"{API_BASE}/query/{st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ── Main chat area ──────────────────────────────────────────────────── #
st.title("🛰️ Ask AstroRAG")
st.caption("Ask anything about space missions, NASA, ESA, or space science from the indexed documents.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}** `{src['source_type']}`")
                    st.caption(src["snippet"])
                    if src.get("url"):
                        st.markdown(f"[Open source]({src['url']})")
                    st.divider()

# Input
if prompt := st.chat_input("e.g. What fuel did Artemis I use?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching mission documents…"):
            r = requests.post(
                f"{API_BASE}/query",
                json={"question": prompt, "session_id": st.session_state.session_id},
            )
        if r.ok:
            d = r.json()
            st.markdown(d["answer"])
            if d.get("sources"):
                with st.expander(f"📄 {len(d['sources'])} source(s)"):
                    for src in d["sources"]:
                        st.markdown(f"**{src['title']}** `{src['source_type']}`")
                        st.caption(src["snippet"])
                        if src.get("url"):
                            st.markdown(f"[Open source]({src['url']})")
                        st.divider()
            st.session_state.messages.append(
                {"role": "assistant", "content": d["answer"], "sources": d.get("sources", [])}
            )
        else:
            err = r.json().get("detail", "Request failed")
            st.error(f"Error: {err}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
