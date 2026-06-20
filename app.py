import streamlit as Streamlit
import os
from src.classifier import classify_customer_persona
from src.rag_pipeline import KnowledgeBaseRAG
from src.escalator import evaluate_escalation, generate_handoff_json
from src.generator import generate_adaptive_response

Streamlit.set_page_config(page_title="Persona Adaptive Support Agent", layout="wide")
Streamlit.title("🤖 Persona-Adaptive Customer Support Engine")
Streamlit.caption("Intermediate AI Engineering Assignment | Grounded RAG Pipeline & Multi-Archetype Handoff Logic")

# Setup Persistent Session States
if "rag_engine" not in Streamlit.session_state:
    with Streamlit.spinner("Initializing Chroma Vector Database and Ingesting Documents..."):
        rag = KnowledgeBaseRAG()
        # Seed dummy folder setup automatically if missing for deployment check
        if not os.path.exists("./data"):
            os.makedirs("./data")
            with open("./data/api_troubleshooting.md", "w") as f:
                f.write("# API System Errors\nCode 401: Unauthorized. Regeneration of API Keys required inside security dashboard.")
            with open("./data/billing_policy.txt", "w") as f:
                f.write("Billing Refund Policy: Refunds are processed strictly within 14 subscription days via account panel.")
        rag.ingest_data_directory()
        Streamlit.session_state.rag_engine = rag

if "chat_history" not in Streamlit.session_state:
    Streamlit.session_state.chat_history = []

# Sidebar Context
with Streamlit.sidebar:
    Streamlit.header("⚙️ Engine Architecture Control")
    Streamlit.info("System uses: `gemini-2.5-flash` for high-speed dynamic tone adaptation and `text-embedding-004` via the modern `google-genai` SDK.")
    if Streamlit.button("Clear Thread Memory"):
        Streamlit.session_state.chat_history = []
        Streamlit.rerun()

# Dynamic Input Box Form
with Streamlit.form(key="chat_form", clear_on_submit=True):
    user_input = Streamlit.text_input("Enter your support query here:", placeholder="Type a message tracking your problem...")
    submit_button = Streamlit.form_submit_with_key = Streamlit.form_submit_button("Transmit Ticket")

if submit_button and user_input:
    # 1. Process Classification Strategy
    classification = classify_customer_persona(user_input)
    
    # 2. Query Context from Chroma Vector Pipeline
    retrieved_chunks = Streamlit.session_state.rag_engine.retrieve_context(user_input, top_k=2)
    
    # 3. Check System Operational Safety Boundaries / Escalation
    escalate_triggered, cause_reason = evaluate_escalation(
        query=user_input, 
        persona_info=classification.persona, 
        context_chunks=retrieved_chunks, 
        conversation_history=Streamlit.session_state.chat_history
    )
    
    # Render Output Segment Blocks
    Streamlit.markdown(f"### **User Request:** *\"{user_input}\"*")
    
    # Layout Output Info Columns
    col1, col2, col3 = Streamlit.columns(3)
    with col1:
        Streamlit.metric(label="Detected Persona Archetype", value=classification.persona)
    with col2:
        Streamlit.metric(label="Sentiment Score Tone", value=classification.sentiment)
    with col3:
        status_value = "🚨 ESCALATED TO HUMAN" if escalate_triggered else "🟢 RESOLVED BY AI"
        Streamlit.metric(label="System Core Status", value=status_value)
        
    Streamlit.divider()
    
    if escalate_triggered:
        Streamlit.error(f"**Escalation Trigger Activated:** {cause_reason}")
        # Generate official handoff tracking JSON package matching assignments requirements
        handoff_json = generate_handoff_json(
            persona=classification.persona,
            query=user_input,
            chunks=retrieved_chunks,
            history=Streamlit.session_state.chat_history
        )
        Streamlit.subheader("📋 Human Engineering Hand-off System Payload (JSON)")
        Streamlit.code(handoff_json, language="json")
    else:
        # 4. Generate Tone Adapted Responses
        with Streamlit.spinner("Adapting persona instructions and formulating grounded response..."):
            agent_response = generate_adaptive_response(user_input, classification.persona, retrieved_chunks)
            
        Streamlit.subheader("💬 Generated Support Response")
        Streamlit.markdown(agent_response)
        
        # Display Sources Transparently for Audit Checks
        with Streamlit.expander("📚 Grounding Sources Retrieved from ChromaDB"):
            for chunk in retrieved_chunks:
                Streamlit.write(f"**Source Document File:** {chunk['source']} | *Matching Confidence Score:* {chunk['score']:.2f}")
                Streamlit.caption(chunk['text'])
                Streamlit.divider()
                
        # Append Memory Cache State
        Streamlit.session_state.chat_history.append({"user": user_input, "agent": agent_response})