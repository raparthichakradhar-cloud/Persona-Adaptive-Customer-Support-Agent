from google import genai
from google.genai import types
from pydantic import BaseModel
from src.config import GEMINI_API_KEY, GENERATION_MODEL, RETRIEVAL_CONFIDENCE_THRESHOLD

class HandoffSummary(BaseModel):
    persona: str
    issue: str
    documents_used: list[str]
    attempted_steps: list[str]
    recommendation: str

def evaluate_escalation(query: str, persona_info: str, context_chunks: list, conversation_history: list) -> tuple[bool, str]:
    """Evaluates multi-pronged escalation criteria rules."""
    
    # Trigger 1: Empty or completely irrelevant context
    if not context_chunks:
        return True, "No operational knowledge matching query found in local DB."
        
    # Trigger 2: Highest chunk match confidence is below configuration safety threshold
    best_score = max([chunk["score"] for chunk in context_chunks])
    if best_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return True, f"Knowledge base confidence score ({best_score:.2f}) drops below required safety ceiling."

    # Trigger 3: Explicit high-risk sensitive topics detected
    sensitive_keywords = ["legal action", "sue", "refund request", "compliance breach", "fraud"]
    if any(keyword in query.lower() for keyword in sensitive_keywords):
        return True, "High-risk structural keywords caught by safety filters."

    return False, ""

def generate_handoff_json(persona: str, query: str, chunks: list, history: list) -> str:
    """Uses structured definitions to pass clean, state-tracked handoffs to human engineering reps."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = (
        f"Analyze this failed conversation context. Build an incident handoff ticket tracking schema.\n"
        f"User Persona: {persona}\n"
        f"Current Query: {query}\n"
        f"Context Chunks Used: {[c['source'] for c in chunks]}\n"
        f"History Raw Log: {history}"
    )
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=HandoffSummary,
            temperature=0.2
        )
    )
    return response.text