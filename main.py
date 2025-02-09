from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import uvicorn
from gradio_client import Client
from dotenv import load_dotenv
from pinecone import Pinecone
import os 
import torch
from sentence_transformers import SentenceTransformer

load_dotenv()
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")
INDEX_URL = os.getenv("INDEX_URL")

PICOCODE_APIKEY=os.getenv("PICOCODE_APIKEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")

pc = Pinecone(api_key=PICOCODE_APIKEY)
index = pc.Index("rag-update")

app = FastAPI()


embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    use_auth_token=EMBED_MODEL
)


def get_embedding(text):
    """Generate embeddings for the given text using GPU."""
    return embed_model.encode(text, convert_to_tensor=True).to("cpu").tolist() 

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy function to simulate RAG pipeline
def search_similar_embeddings(query, top_k=7):
    response = index.query(
        vector= get_embedding(query),
        top_k=10,
        include_values=True,
        include_metadata=True,
    )
def generate_answer_from_deployed_model(context, query):
  client = Client("lewisnjue/mistralai-Mistral-7B-Instruct-v0.3") 
  prompt = f"Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

  result = client.predict(
      prompt,  
      api_name="/chat"  
  )

  return result 
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest) -> Dict[str, str]:
    """Handles user question and generates an answer."""
    query = request.question
    result = search_similar_embeddings(query)
    context = " ".join([item.metadata["text"] for item in result.matches])
    answer = generate_answer_from_deployed_model(context, query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
