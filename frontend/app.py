import streamlit as st
import requests
from dotenv import load_dotenv
from pathlib import Path
import os

from chat_ui import (
    build_assistant_message,
    build_chat_payload,
    format_source_label,
    summarize_citation_verification,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
STRATEGY_REQUEST_TIMEOUT_SECONDS = min(10, REQUEST_TIMEOUT_SECONDS)

FALLBACK_CHUNKING_STRATEGIES = {
    "default_strategy": "research_paper",
    "strategies": [
        {
            "key": "research_paper",
            "label": "Research Paper",
            "description": "Balanced chunks for sectioned academic writing, citations, and method-heavy PDFs.",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        {
            "key": "general_article",
            "label": "Article / Report",
            "description": "Larger chunks for continuous prose like blogs, essays, and business reports.",
            "chunk_size": 1200,
            "chunk_overlap": 150,
        },
        {
            "key": "notes_transcript",
            "label": "Notes / Transcript",
            "description": "Smaller chunks for fast topic shifts, bullet points, meeting notes, or transcripts.",
            "chunk_size": 650,
            "chunk_overlap": 120,
        },
    ],
}

RETRIEVAL_MODE_LABELS = {
    "chunk": "Chunk retrieval",
    "hybrid": "Hybrid (BM25 + vector)",
    "page": "Page retrieval",
    "cag": "CAG (full doc)",
}


def extract_error_message(response):
    if response is None:
        return "The request failed before the backend returned a response."

    try:
        payload = response.json()
    except ValueError:
        return response.text or f"Request failed with status code {response.status_code}."

    return (
        payload.get("detail")
        or payload.get("message")
        or response.text
        or f"Request failed with status code {response.status_code}."
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chunking_strategies():
    try:
        response = requests.get(
            f"{BACKEND_URL}/chunking-strategies",
            timeout=STRATEGY_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return FALLBACK_CHUNKING_STRATEGIES


def render_sources(sources):
    if not sources:
        return

    with st.expander("Sources", expanded=True):
        for source in sources:
            st.markdown(f"**{format_source_label(source)}**")
            excerpt = source.get("excerpt")
            if excerpt:
                st.caption(excerpt)


def render_citation_verification(citation_verification):
    summary = summarize_citation_verification(citation_verification)
    if not summary:
        return

    if (
        citation_verification.get("grounded")
        and citation_verification.get("all_citations_valid")
    ):
        st.caption(summary)
        return

    st.warning(summary)
    for issue in citation_verification.get("unsupported_claims") or []:
        claim = issue.get("claim", "Unsupported claim")
        reason = issue.get("reason", "")
        cited_source_ids = issue.get("cited_source_ids") or []
        citation_label = f" Cited: {', '.join(cited_source_ids)}." if cited_source_ids else ""
        st.write(f"- {claim}. {reason}{citation_label}".strip())

st.title("DocuTalk - Talk to Research Papers")

uploaded_file = st.file_uploader("Upload your research paper (PDF or TXT)", type=["pdf", "txt"])
chunking_strategies = fetch_chunking_strategies()
strategy_options = chunking_strategies["strategies"]
strategy_map = {strategy["key"]: strategy for strategy in strategy_options}
strategy_keys = [strategy["key"] for strategy in strategy_options]
default_strategy = chunking_strategies.get("default_strategy", strategy_keys[0])

if default_strategy not in strategy_map:
    default_strategy = strategy_keys[0]

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None

selected_strategy_key = st.session_state.get("selected_chunking_strategy", default_strategy)
if selected_strategy_key not in strategy_map:
    selected_strategy_key = default_strategy

selected_strategy_key = st.selectbox(
    "Chunking strategy",
    options=strategy_keys,
    index=strategy_keys.index(selected_strategy_key),
    format_func=lambda key: strategy_map[key]["label"],
    key="selected_chunking_strategy",
    help="Choose the chunking preset that best matches the type of document you're uploading.",
)
selected_strategy = strategy_map[selected_strategy_key]
st.caption(
    f"{selected_strategy['description']} "
    f"Chunk size: {selected_strategy['chunk_size']}, overlap: {selected_strategy['chunk_overlap']}."
)

selected_retrieval_mode = st.radio(
    "Retrieval mode",
    options=list(RETRIEVAL_MODE_LABELS.keys()),
    index=0,
    horizontal=True,
    format_func=lambda key: RETRIEVAL_MODE_LABELS[key],
    key="selected_retrieval_mode",
    help=(
        "Chunk retrieval is narrower and more precise. "
        "Hybrid blends BM25 keyword matching with vector search. "
        "Page retrieval searches full pages for broader context. "
        "CAG loads the full document for smaller uploads."
    ),
)

if uploaded_file:
    if st.button("Process Document"):
        with st.spinner("Indexing document.."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    data={"chunking_strategy": selected_strategy_key},
                    files=files,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                st.session_state.active_document_id = payload.get("document_id")
                used_strategy = strategy_map.get(payload.get("chunking_strategy"), selected_strategy)
                st.success(
                    f"Document indexed successfully using the {used_strategy['label']} strategy."
                )
            except requests.Timeout:
                st.error(
                    "The upload request timed out while the backend was indexing the document. "
                    "Check the backend logs for the last completed ingest step."
                )
            except requests.RequestException as exc:
                st.error(extract_error_message(exc.response))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citation_verification(message.get("citation_verification"))
            render_sources(message.get("sources") or [])

if prompt := st.chat_input("Ask a question about the paper.."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        payload = build_chat_payload(
            question=prompt,
            active_document_id=st.session_state.active_document_id,
            retrieval_mode=selected_retrieval_mode,
        )
        response_payload = None

        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response_payload = response.json()
            answer = response_payload.get("answer", "Something went wrong")
            if response_payload.get("document_id"):
                st.session_state.active_document_id = response_payload["document_id"]
        except requests.Timeout:
            answer = "The chat request timed out while waiting for the backend."
        except requests.RequestException as exc:
            answer = extract_error_message(exc.response)

        assistant_message = build_assistant_message(
            response_payload,
            fallback_answer=answer,
        )
        st.markdown(assistant_message["content"])
        render_citation_verification(assistant_message.get("citation_verification"))
        render_sources(assistant_message.get("sources") or [])
    st.session_state.messages.append(assistant_message)
