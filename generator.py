from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GENERATION_MODEL

def generate_adaptive_response(user_query: str, persona: str, context_chunks: list) -> str:
    """Selects target structural system guidelines matching client archetype traits."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Archetype Prompts Definition Matrix
    if persona == "Technical Expert":
        system_prompt = (
            "You are a Senior Infrastructure Support Engineer. Provide comprehensive root cause explanations, "
            "precise code fragments, configurations, network pathways, and detailed infrastructure diagnostics. "
            "Maintain technical integrity."
        )
    elif persona == "Frustrated User":
        system_prompt = (
            "You are an Empathetic Customer Care Lead. Begin immediately with an explicit validation statement "
            "acknowledging their frustration. Avoid complex architectural terms or jargon. Present solutions in "
            "highly visible, simple, sequential action steps."
        )
    else:  # Business Executive
        system_prompt = (
            "You are a Concise Client Relations Executive. Deliver straight summaries emphasizing operational impact, "
            "downtime metrics, and clear resolution schedules. Avoid all fine details, code paths, and verbose technical descriptions."
        )
        
    # Append explicit constraints preventing grounding drift
    system_prompt += (
        "\nCRITICAL INSTRUCTION: Formulate your answer exclusively based on the provided context fragments. "
        "Do not extrapolate or assume unverified operational facts outside this text data."
    )
    
    # Construct context bundle text
    context_str = "\n\n".join([f"Source [{c['source']}]:\n{c['text']}" for c in context_chunks])
    
    prompt = f"Context Documentation:\n{context_str}\n\nUser Question:\n{user_query}"
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3
        )
    )
    return response.text