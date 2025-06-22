import os
import re
import json
import PyPDF2
import numpy as np
import streamlit as st
from datetime import datetime
from typing import List, Dict, Tuple
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 500
        self.chunk_overlap = 100
        
    def process_pdf(self, file_path: str) -> List[Dict]:
        text_chunks = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks = self._chunk_text(text, f"page_{page_num+1}")
                    text_chunks.extend(chunks)
        return text_chunks
    
    def process_transcript(self, file_path: str) -> List[Dict]:
        text_chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            current_chunk = ""
            timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}')
            
            for line in lines:
                if timestamp_pattern.match(line):
                    if current_chunk:
                        chunks = self._chunk_text(current_chunk, "transcript")
                        text_chunks.extend(chunks)
                        current_chunk = ""
                current_chunk += line
            
            if current_chunk:
                chunks = self._chunk_text(current_chunk, "transcript")
                text_chunks.extend(chunks)
        
        return text_chunks
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'timestamp': datetime.now().isoformat()
                })
                current_chunk = current_chunk[-self.chunk_overlap:]
                current_length = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'source': source,
                'timestamp': datetime.now().isoformat()
            })
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks

class VectorStore:
    def __init__(self, storage_dir='vector_store'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_embeddings(self, chunks: List[Dict], name: str) -> str:
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(chunks, f)
        
        return filepath
    
    def load_embeddings(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_stored_embeddings(self) -> List[Tuple[str, str]]:
        files = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                files.append((filename, filepath))
        return files
    
    def search(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(query)
        
        similarities = []
        for chunk in chunks:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((similarity, chunk))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in similarities[:top_k]]

class GeminiChatbot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        context_str = "\n\n".join([
            f"Source: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in context
        ])
        
        prompt = f"""
        Use the following context to answer the question. 
        Cite your sources clearly and be as specific as possible.
        
        Context:
        {context_str}
        
        Question: {query}
        
        Answer:
        """
        
        response = self.model.generate_content(prompt)
        return response.text

def main():
    st.set_page_config(page_title="RAG Prototype with Gemini", layout="wide")
    st.title("RAG Prototype with Gemini")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'gemini_chatbot' not in st.session_state:
        st.session_state.gemini_chatbot = None
    
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.session_state.gemini_chatbot = GeminiChatbot(api_key)
            st.success("API Key set!")
        
        st.header("Document Processing")
        file_type = st.radio("Select file type", ("PDF", "Transcript"))
        uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt"])
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if file_type == "PDF":
                    chunks = st.session_state.processor.process_pdf(temp_file)
                else:
                    chunks = st.session_state.processor.process_transcript(temp_file)
                
                chunks_with_embeddings = st.session_state.processor.generate_embeddings(chunks)
                save_path = st.session_state.vector_store.save_embeddings(
                    chunks_with_embeddings, 
                    os.path.splitext(uploaded_file.name)[0]
                )
                
                st.session_state.current_embeddings = chunks_with_embeddings
                st.success(f"Processed {len(chunks)} chunks and saved embeddings!")
                os.remove(temp_file)
        
        st.header("Load Existing Embeddings")
        embedding_files = st.session_state.vector_store.list_stored_embeddings()
        if embedding_files:
            selected_file = st.selectbox(
                "Select embeddings to load",
                [f[0] for f in embedding_files]
            )
            
            if st.button("Load Selected Embeddings"):
                filepath = next(f[1] for f in embedding_files if f[0] == selected_file)
                st.session_state.current_embeddings = st.session_state.vector_store.load_embeddings(filepath)
                st.success(f"Loaded {len(st.session_state.current_embeddings)} chunks!")
    
    if 'current_embeddings' not in st.session_state:
        st.warning("Please process or load a document first.")
        return
    
    st.header("Chat with the Document")
    query = st.text_input("Enter your question about the document:")
    
    if query and st.button("Search"):
        with st.spinner("Searching and generating response..."):
            relevant_chunks = st.session_state.vector_store.search(
                query, 
                st.session_state.current_embeddings
            )
            
            if st.session_state.gemini_chatbot:
                response = st.session_state.gemini_chatbot.generate_response(query, relevant_chunks)
                
                st.subheader("Answer")
                st.write(response)
                
                st.subheader("Sources Used")
                for i, chunk in enumerate(relevant_chunks, 1):
                    with st.expander(f"Source {i}: {chunk['source']}"):
                        st.write(chunk['text'])
            else:
                st.error("Please set your Gemini API key first.")

if __name__ == "__main__":
    main()
