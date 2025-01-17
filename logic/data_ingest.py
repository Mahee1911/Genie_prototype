import os
from typing import List, Dict, Any
from io import BytesIO
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

class DataIngestionAgent:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path

    def process_documents(self, pdf_contents: List[BytesIO]) -> Dict[str, Any]:
        """Handle PDF content processing and vector DB creation/loading."""
        pages = []
        
        for pdf_content in pdf_contents:
            if not isinstance(pdf_content, BytesIO):
                raise ValueError("Each item in pdf_contents must be a BytesIO object.")

            with tempfile.NamedTemporaryFile(dir=TEMP_DIR, delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_content.getvalue()) 
                temp_pdf.flush()
                temp_pdf_path = temp_pdf.name

            try:
                loader = PyPDFLoader(temp_pdf_path)
                pages.extend(loader.load_and_split()) 
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
        
        if not pages:
            raise ValueError("No valid pages found in the uploaded PDF files.")

        text_splitter = CharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)

        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            embeddings = OpenAIEmbeddings()  
            vector_db = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new FAISS index...")
            embeddings = OpenAIEmbeddings()
            vector_db = FAISS.from_documents(docs, embeddings)
            
            print("Saving FAISS index...")
            vector_db.save_local(self.index_path)

        return docs, vector_db
