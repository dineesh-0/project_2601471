# Module 3 — Zepto RAG Service (single file version)

import os, glob
from typing import TypedDict
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Step 1: Build Corpus Embeddings
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("zepto_policies")

# Embed all docs (run once at startup)
for file in glob.glob("docs/*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        embedding = model.encode(text).tolist()
        collection.add(documents=[text], embeddings=[embedding], ids=[os.path.basename(file)])

print("✅ Corpus embedded and stored in ChromaDB")

# -----------------------------
# Step 2: Define State & Graph Nodes
# -----------------------------
MOCK_LLM = os.getenv("MOCK_LLM", "1")

class State(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float

def classify_intent(state: State) -> State:
    q = state["query"].lower()
    keywords = ["delivery","return","refund","membership","tracking","cancel","gift card","support hours"]
    state["intent"] = "policy_question" if any(k in q for k in keywords) else "general_question"
    return state

def retrieve_and_answer(state: State) -> State:
    results = collection.query(query_texts=[state["query"]], n_results=3)
    if results["documents"] and results["documents"][0]:
        top_chunk = results["documents"][0][0][:200]
        if MOCK_LLM == "1":
            state["answer"] = f"Based on the retrieved context: {top_chunk}"
        else:
            state["answer"] = "(LLM grounded answer)"
        state["sources"] = results["ids"][0]
    else:
        state["answer"] = "No relevant context found in corpus."
        state["sources"] = []
    state["confidence"] = 1.0
    return state

def direct_answer(state: State) -> State:
    if MOCK_LLM == "1":
        state["answer"] = "I can only answer questions about Zepto policies right now."
    else:
        state["answer"] = "(LLM direct answer)"
    state["sources"] = []
    state["confidence"] = 1.0
    return state

# -----------------------------
# Step 3: FastAPI Wrapper
# -----------------------------
app = FastAPI()

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    state: State = {"query": req.query, "intent": "", "answer": "", "sources": [], "confidence": 0.0}
    state = classify_intent(state)
    if state["intent"] == "policy_question":
        state = retrieve_and_answer(state)
    else:
        state = direct_answer(state)
    return AskResponse(answer=state["answer"], sources=state["sources"], confidence=state["confidence"])

# -----------------------------
# Step 4: Example Runs (Mock Mode)
# -----------------------------
if __name__ == "__main__":
    # Example 1: Policy question
    state = {"query": "What is Zepto's delivery policy?", "intent": "", "answer": "", "sources": [], "confidence": 0.0}
    state = classify_intent(state)
    state = retrieve_and_answer(state)
    print("Policy question result:", state)

    # Example 2: General question
    state = {"query": "Who is the CEO of Zepto?", "intent": "", "answer": "", "sources": [], "confidence": 0.0}
    state = classify_intent(state)
    state = direct_answer(state)
    print("General question result:", state)

    # To run FastAPI: uvicorn module3:app --reload --port 7868
