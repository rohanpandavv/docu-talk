from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

app = FastAPI()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory=f"{BASE_DIR}/chroma_db")
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class ChatRequest(BaseModel):
    question: str

