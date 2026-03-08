import streamlit as st
import requests
from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

BACKEND_URL = os.getenv("BACKEND_URL")

st.title("DocuTalk - Talk to Research Papers")

uploaded_file = st.file_uploader("Upload your research paper (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    if st.button("Process Document"):
        with st.spinner("Indexing document.."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
            if response.status_code == 200:
                st.success("Document indexed successfully!")

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
        response = requests.post("http://localhost:8000/chat", json={"question": prompt})
        answer = response.json().get("answer", "Something went wrong")
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})