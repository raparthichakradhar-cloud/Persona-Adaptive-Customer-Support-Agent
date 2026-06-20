from google import genai
from google.genai import types
from pydantic import BaseModel
from src.config import GEMINI_API_KEY, GENERATION_MODEL

class PersonaClassification(BaseModel):
    persona: str # "Technical Expert", "Frustrated User", "Business Executive"
    sentiment: str
    confidence: float
    reasoning: str

def classify_customer_persona(user_message: str) -> PersonaClassification:
    """Analyzes message structural traits & sentiment to extract user archetype."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    system_instruction = (
        "You are an advanced classification engine. Analyze the sentiment, traits, and "
        "vocabulary of incoming support tickets. Classify them into exactly one of these categories:\n"
        "1. 'Technical Expert': Uses developer jargon, asks about APIs, code, or logs.\n"
        "2. 'Frustrated User': Uses emotional/urgent text, ALL CAPS, or expresses intense dissatisfaction.\n"
        "3. 'Business Executive': Focuses on corporate impact, timelines, brevity, and operational effects."
    )
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=PersonaClassification,
            temperature=0.1
        )
    )
    
    return PersonaClassification.model_validate_json(response.text)