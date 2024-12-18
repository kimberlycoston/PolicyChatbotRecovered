import os
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# Use the environment variables in your code
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Access the index
index_name = "testbrt"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings dimension
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

def query_pinecone(question):
    # Generate embeddings for the question
    response = openai.Embedding.create(input=question, model="text-embedding-ada-002")
    query_embedding = response['data'][0]['embedding']
    
    # Query Pinecone
    results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    return results

def generate_answer(question, context_chunks):
    context = "\n".join([f"Chunk {i + 1}: {chunk['metadata']['text']}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""
    You are a helpful assistant. Use ONLY the provided policy documents to answer the question below. Start each answer with "Based on (Insert Policy Name) on Page (Insert Page Number),..." If you cannot answer it with the provided documents, state that you are unable to provide an answer based on the provided policies.
    Context:
    {context}
    Question: {question}
    Answer:
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

@app.route("/")
def home():
    return "Welcome to the Policy Chatbot API! Use the /query endpoint to interact."


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")
    
    # Query Pinecone and generate an answer
    results = query_pinecone(question)
    answer = generate_answer(question, results['matches'])
    
    # Return the answer and sources
    return jsonify({
        "answer": answer,
        "sources": [{"document": match['metadata']['document'], "chunk": match['metadata']['chunk_index']} for match in results['matches']]
    })

if __name__ == "__main__":
    app.run(debug=True)
