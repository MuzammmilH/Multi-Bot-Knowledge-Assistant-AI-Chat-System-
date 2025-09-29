# server/main.py

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import faiss
import numpy as np
from server.embeddings import Embedder
from server.retriever import Retriever

app = FastAPI()

# Global retriever
retriever = Retriever()

@app.post("/upload")
async def upload_document(file: UploadFile, bot_name: str = Form("default")):
    """
    Uploads a text or PDF file, embeds it, and updates FAISS index.
    """
    try:
        contents = await file.read()
        file_text = ""

        if file.filename.endswith(".txt"):
            file_text = contents.decode("utf-8")
        elif file.filename.endswith(".pdf"):
            from PyPDF2 import PdfReader
            import io
            pdf = PdfReader(io.BytesIO(contents))
            file_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

        if not file_text.strip():
            return JSONResponse(status_code=400, content={"error": "No text found in file"})

        # Embed new text
        embedder = Embedder()
        new_vec = embedder.encode(file_text)
        retriever.add_to_index(new_vec, file_text, bot_name)

        return {"status": "success", "message": f"File {file.filename} added to bot {bot_name} KB"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
