from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/".env")

app = FastAPI()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory=f"{BASE_DIR}/chroma_db")
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class ChatRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    print("content_type ----> ", file.content_type)
    content_type = file.content_type.lower()
    if content_type == "application/pdf":
        pdf =  PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif content_type == "text/plain":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Text file is not valid UTF-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    vectorstore.add_texts(texts=chunks)
    return {"message": "Document indexed successfully!"}