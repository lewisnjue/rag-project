import os
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from gradio_client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")
INDEX_URL = os.getenv("INDEX_URL")
PICOCODE_APIKEY = os.getenv("PICOCODE_APIKEY")
EMBED_MODEL = os.getenv("EMBED_MODEL")

# Initialize Pinecone
pc = Pinecone(api_key=PICOCODE_APIKEY)
index = pc.Index("rag-update")

# Load embedding model
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_auth_token=EMBED_MODEL
)

def get_embedding(text):
    """Generate embeddings for the given text."""
    return embed_model.encode(text, convert_to_tensor=True).to("cpu").tolist()

def search_similar_embeddings(query, top_k=7):
    """Search for similar embeddings in Pinecone."""
    response = index.query(
        vector=get_embedding(query),
        top_k=top_k,
        include_values=True,
        include_metadata=True,
    )
    return response

def generate_answer_from_deployed_model(context, query):
    """Use the deployed model to generate an answer."""
    client = Client("lewisnjue/mistralai-Mistral-7B-Instruct-v0.3") 
    prompt = f"Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    result = client.predict(
        prompt,  
        api_name="/chat"
    )
    return result

def ask_question(question):
    """Handles user input and generates an answer."""
    result = search_similar_embeddings(question)
    if not result or not hasattr(result, "matches"):
        return "No relevant context found in the knowledge base."

    context = " ".join([item.metadata["text"] for item in result.matches if "text" in item.metadata])
    answer = generate_answer_from_deployed_model(context, question)
    return answer

# Gradio UI
interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question..."),
    outputs="text",
    title="RAG-Powered Q&A",
    description="Ask questions and get AI-generated answers using a retrieval-augmented generation (RAG) pipeline."
)

# Launch Gradio
if __name__ == "__main__":
    interface.launch()
