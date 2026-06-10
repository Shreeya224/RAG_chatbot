# RAG_chatbot
A production-style Retrieval Augmented Generation (RAG) pipeline built with LangChain, OpenAI, FAISS, and Flask. Upload documents to a vector knowledge base and query them with natural language — the LLM answers using only your data as context.
User Question
     │
     ▼
Flask REST API (/chat)
     │
     ▼
FAISS Vector Store ──► Retrieve top-k relevant chunks
     │
     ▼
LangChain RetrievalQA Chain
     │
     ▼
OpenAI GPT-3.5-turbo ──► Generate grounded answer
     │
     ▼
JSON Response (answer + sources)

