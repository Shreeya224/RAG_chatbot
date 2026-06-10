from flask import Flask, request, jsonify
from rag_engine import RAGEngine
import os

app = Flask(__name__)
rag = RAGEngine()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "RAG Chatbot API is running", "endpoints": ["/upload", "/chat", "/health"]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_document():
    """Upload a text document to the knowledge base."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Provide JSON with 'text' field"}), 400

    doc_id = rag.add_document(data["text"], metadata=data.get("metadata", {}))
    return jsonify({"message": "Document added successfully", "doc_id": doc_id})

@app.route("/chat", methods=["POST"])
def chat():
    """Ask a question — RAG retrieves context and LLM answers."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Provide JSON with 'question' field"}), 400

    result = rag.query(data["question"])
    return jsonify({
        "question": data["question"],
        "answer": result["answer"],
        "sources": result["sources"]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
