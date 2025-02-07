from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import uvicorn
from gradio_client import Client
from dotenv import load_dotenv
from upstash_vector import Index, Vector
import os 
load_dotenv()
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")
INDEX_URL = os.getenv("INDEX_URL")

index = Index(url=INDEX_URL, token=UPSTASH_TOKEN)

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy function to simulate RAG pipeline
def search_similar_embeddings(query, top_k=3):
    """Search for embeddings similar to the query."""
    results = index.query(
        data=query,
        top_k=top_k,
        include_vectors=True,
        include_metadata=True,
    )
    return results

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
    context = " ".join([item.metadata["text"] for item in result])
    answer = generate_answer_from_deployed_model(context, query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
