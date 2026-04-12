import streamlit as st
import requests
from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))


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

st.title("DocuTalk - Talk to Research Papers")

uploaded_file = st.file_uploader("Upload your research paper (PDF or TXT)", type=["pdf", "txt"])

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None

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
                    files=files,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                st.session_state.active_document_id = payload.get("document_id")
                st.success("Document indexed successfully!")
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

if prompt := st.chat_input("Ask a question about the paper.."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        payload = {"question": prompt}
        if st.session_state.active_document_id:
            payload["document_id"] = st.session_state.active_document_id

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

        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
