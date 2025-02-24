import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import uvicorn
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone
from google import genai 
load_dotenv()
PICOCODE_APIKEY = os.getenv("PICOCODE_APIKEY")
EMBED_MODEL = os.getenv("EMBED_MODEL") 
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
pc = Pinecone(api_key=PICOCODE_APIKEY)
index = pc.Index("rag-update")
app = FastAPI()
client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_auth_token=EMBED_MODEL
)
def get_embedding(text):
    """Generate embeddings for the given text."""
    return embed_model.encode(text, convert_to_tensor=True).to("cpu").tolist()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def search_similar_embeddings(query, top_k=7):
    """Query the Pinecone index to find similar embeddings."""
    response = index.query(
        vector=get_embedding(query),
        top_k=top_k,
        include_values=True,
        include_metadata=True,
    )
    return response
def generate_answer_from_google(context, query):
    """
    Uses Google's Gemini API (gemini-2.0-flash) to generate a response
    based on the context retrieved from Pinecone.
    The internal RAG process is not exposed to the requester.
    """
    if not context.strip():
        return "I'm sorry, I don't have enough context to answer that."
    prompt = (
        f"Based on the following context, provide a concise, clear answer to the question.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print("Error generating answer:", e)
        return "Error generating answer."
class QueryRequest(BaseModel):
    question: str
@app.post("/ask")
def ask_question(request: QueryRequest) -> Dict[str, str]:
    """
    Endpoint that:
      1. Queries Pinecone for context related to the user's question.
      2. Generates an answer using Google's Gemini API.
    The user receives only the final answer, without any indication of the underlying RAG process.
    """
    query = request.question
    result = search_similar_embeddings(query)
    
    if not result or not hasattr(result, "matches") or not result.matches:
        return {"answer": "No relevant context found in the knowledge base."}
    context = " ".join(
        [item.metadata.get("text", "") for item in result.matches if "text" in item.metadata]
    )
    answer = generate_answer_from_google(context, query)
    print(answer)
    return {"answer": answer}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
