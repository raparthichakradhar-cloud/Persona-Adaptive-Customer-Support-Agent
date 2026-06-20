[ User Message Input ]
                 │
                 ▼
     [ 1. Persona Classifier ] ──► (Outputs: Technical / Frustrated / Executive)
                 │
                 ▼
     [ 2. Embedding Engine ] (Gemini text-embedding-004)
                 │
                 ▼
     [ 3. Vector DB Search ] (ChromaDB Cosine Similarity Query)
                 │
                 ├──► [ 4. Escalation Safety Evaluator ]
                 │         ├── Low Confidence (< 0.45) ──┐
                 │         ├── High-Risk Topic          ──┼─► [ Generate Handoff JSON ]
                 │         └── Max Turns Exceeded       ──┘           │
                 │                                                    ▼
                 ▼                                           [ Route to Human UI ]
     [ 5. Adaptive Prompt Compiler ] 
        (Inject Persona System Prompt + Context Chunks)
                 │
                 ▼
     [ 6. Gemini Generation Engine ] ──► [ Render Adaptive Response to User ]