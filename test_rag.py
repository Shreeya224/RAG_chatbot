"""
Quick test script to verify the RAG pipeline works end-to-end.
Run: python test_rag.py
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("Health check:", r.json())

def test_upload():
    # Sample automotive knowledge base document
    doc = {
        "text": """
        HARMAN Automotive specializes in connected car technologies and in-vehicle infotainment systems.
        Their products include audio systems, navigation, telematics, and ADAS solutions.
        HARMAN is a subsidiary of Samsung Electronics and serves major automakers worldwide.
        Key technologies include digital cockpits, OTA updates, and cloud-connected vehicle platforms.
        HARMAN's Ignite platform enables seamless over-the-air software updates for vehicles.
        The company works with brands like BMW, Toyota, Mercedes-Benz, and General Motors.
        """,
        "metadata": {"source": "harman_overview", "topic": "automotive"}
    }
    r = requests.post(f"{BASE_URL}/upload", json=doc)
    print("Upload response:", r.json())

def test_chat():
    questions = [
        "What does HARMAN Automotive specialize in?",
        "Which car brands does HARMAN work with?",
        "What is the Ignite platform?"
    ]
    for q in questions:
        r = requests.post(f"{BASE_URL}/chat", json={"question": q})
        result = r.json()
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources used: {len(result['sources'])}")

if __name__ == "__main__":
    print("=== Testing RAG Chatbot API ===\n")
    test_health()
    test_upload()
    test_chat()
