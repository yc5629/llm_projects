import os
import requests
import logging
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sambanova_embeddings import SambaNovaEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from pathlib import Path

# 👇 Load .env from parent directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# 📁 Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 🔐 ENV config
SAMBA_API_KEY = os.getenv("SAMBA_API_KEY")
SAMBA_EMBED_MODEL = "E5-Mistral-7B-Instruct"
SAMBA_CHAT_MODEL = "Meta-Llama-3.2-1B-Instruct"
SAMBA_API_BASE = "https://api.sambanova.ai/v1"


HEADERS = {
    "Authorization": f"Bearer {SAMBA_API_KEY}",
    "Content-Type": "application/json"
}

# 📄 Load and split text file
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": filename}))
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# 🗄️ Build ChromaDB
def build_vectorstore(docs: List[Document]):
    persist_dir = "chroma_storage"
    embeddings = SambaNovaEmbeddings(api_key=SAMBA_API_KEY, model=SAMBA_EMBED_MODEL, base_url=SAMBA_API_BASE)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_docs",
        persist_directory=persist_dir
    )
    return vectorstore

# 🔍 Query
def query_vectorstore(query, vectorstore, top_k=3):
    embedding = SambaNovaEmbeddings(api_key=SAMBA_API_KEY, model=SAMBA_EMBED_MODEL).embed_query(query)
    results = vectorstore.similarity_search_by_vector(embedding, k=top_k)
    return results

# 💬 Generate answer from retrieved context
def generate_answer(question, context_docs):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Use the following context to answer the question:\n\n{context_text}\n\nQ: {question}\nA:"""
    payload = {
        "model": SAMBA_CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(f"{SAMBA_API_BASE}/chat/completions", headers=HEADERS, json=payload)

    try:
        response.raise_for_status()
        response_json = response.json()
        answer = response_json["choices"][0]["message"]["content"]
        logging.info("🧠 Prompt: %s", question)
        logging.info("🤖 Response: %s", answer)
        return answer
    except requests.exceptions.RequestException as e:
        logging.error("❌ Request failed: %s", str(e))
        logging.error("🔎 Full response content: %s", response.text)
        return "Error: Failed to generate answer."
    except KeyError:
        logging.error("❌ Unexpected response format: %s", response.text)
        return "Error: Unexpected response format."


# 🚀 Main logic
if __name__ == "__main__":
    print("📚 Loading and indexing documents...")
    raw_docs = load_documents("data")
    split_docs = split_documents(raw_docs)
    vectordb = build_vectorstore(split_docs)

    while True:
        user_q = input("\n❓ Ask a question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        retrieved_docs = query_vectorstore(user_q, vectordb)
        answer = generate_answer(user_q, retrieved_docs)
        print("\n🤖 Answer:\n", answer)
