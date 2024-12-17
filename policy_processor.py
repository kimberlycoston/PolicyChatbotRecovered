import os
import openai
import pinecone
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Use the environment variables in your code
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Create or connect to the index
index_name = "testbrt"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # For OpenAI embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud="gcp", region="us-east1")
    )

# Connect to the index
index = pc.Index(index_name)

# Functions for PDF processing
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_into_chunks(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def upload_to_pinecone(chunks, document_name, index):
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        index.upsert([{
            "id": f"{document_name}_chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk, "document": document_name, "chunk_index": i}
        }])

def query_pinecone(question, top_k=5):
    query_embedding = generate_embedding(question)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

def generate_answer(question, context_chunks):
    context = "\n".join([f"Chunk {i + 1}: {chunk['metadata']['text']}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""
    You are a helpful assistant. Use ONLY the provided policy documents to answer the question below. If you cannot answer it with the provided documents, state that you are unable to provide an answer based on the provided policies.
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

# Example usage: Upload a policy PDF
# pdf_text = extract_text_from_pdf("policy_document.pdf")
# chunks = split_into_chunks(pdf_text)
# upload_to_pinecone(chunks, "Policy Document")

# Example usage: Ask a question
# question = "Can a patient refuse care medication if they are on a medical hold?"
# results = query_pinecone(question)
# answer = generate_answer(question, results['matches'])
# print("Answer:", answer)

# Automatically find all PDFs in the folder
pdf_folder = "policy_pdfs"  # Path to your folder with PDFs
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Process each PDF
for pdf_file in pdf_files:
    document_name = pdf_file.split(".pdf")[0]  # Use the file name without the extension
    print(f"Processing document: {document_name}")
    
    # Extract text
    pdf_text = extract_text_from_pdf(pdf_file)
    
    # Split into chunks
    chunks = split_into_chunks(pdf_text)
    
    # Upload to Pinecone
    upload_to_pinecone(chunks, document_name, index)
    
    print(f"Uploaded {document_name} to Pinecone.")

# Include document references in the response
# print("Sources:")
# for match in results['matches']:
#     print(f"- {match['metadata']['document']}, Chunk {match['metadata']['chunk_index']}")
