import os
import chromadb
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL

class KnowledgeBaseRAG:
    def __init__(self, db_dir="./chroma_db"):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="support_kb",
            metadata={"hnsw:space": "cosine"}
        )
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

    def ingest_data_directory(self, data_dir="./data"):
        """Reads, chunks, and indexes documents into the vector database."""
        if not os.path.exists(data_dir):
            return
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)
        
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            content = ""
            
            if file_name.endswith(".pdf"):
                content = self.extract_text_from_pdf(file_path)
            elif file_name.endswith((".txt", ".md")):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
            if not content.strip():
                continue
                
            chunks = text_splitter.split_text(content)
            
            for idx, chunk in enumerate(chunks):
                # Generate Embedding using modern SDK
                emb_res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk
                )
                embedding = emb_res.embeddings[0].values
                
                self.collection.upsert(
                    ids=[f"{file_name}_{idx}"],
                    embeddings=[embedding],
                    metadatas=[{"source": file_name, "chunk_idx": idx}],
                    documents=[chunk]
                )

    def retrieve_context(self, query: str, top_k: int = 3) -> list:
        """Retrieves top-k documentation matches and handles distance conversion."""
        emb_res = self.client.models.embed_content(model=EMBEDDING_MODEL, contents=query)
        query_vector = emb_res.embeddings[0].values
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        parsed_items = []
        if not results or not results["documents"] or not results["documents"][0]:
            return parsed_items
            
        for i in range(len(results["documents"][0])):
            # Distance mapping to similarity score
            distance = results["distances"][0][i]
            similarity_score = 1.0 - distance
            
            parsed_items.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": max(0.0, similarity_score)
            })
            
        return parsed_items