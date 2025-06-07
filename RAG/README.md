# 🧠 RAG Application with SambaNova + LangChain

This project implements a full Retrieval-Augmented Generation (RAG) system using:

- 💡 **SambaNova Systems API** for embeddings and LLMs
- 🔗 **LangChain** to manage document pipelines and vector search
- 🗃️ **ChromaDB** for vector store
- 📄 **Text file loader** for `.txt` sources
- 🧪 Interactive Q&A loop

## 🚀 Features

- Embeds and indexes `.txt` files in `./data`
- Queries top-k relevant chunks using semantic search
- Sends context to SambaNova chat model for response
- Logs all interactions to `logs/app.log`

## 🛠 Setup

```bash
git clone https://github.com/<your-username>/llm_projects.git
cd llm_projects/RAG

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
