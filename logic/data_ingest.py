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
        """Handle PDF content processing and vector DB creation."""
        pages = []
        for pdf_content in pdf_contents:
            # Save the BytesIO content to a custom temporary directory
            with tempfile.NamedTemporaryFile(dir=TEMP_DIR, delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_content.getvalue())
                temp_pdf.flush()
                temp_pdf_path = temp_pdf.name
            
            try:
                loader = PyPDFLoader(temp_pdf_path)
                pages.extend(loader.load_and_split())
            finally:
                # Clean up the temporary file
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

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(docs, embeddings)
        
        return docs, vector_db