from flask import Blueprint, request
from io import BytesIO
from logic.topic_extract import TopicExtractorAgent
from logic.data_ingest import DataIngestionAgent

router = Blueprint('upload', __name__)

@router.route("/api/upload/", methods=['POST'])
def upload_files():
    try:
        if not request.files:
            return {"error": "No files uploaded."}

        files = request.files.getlist('files')
        
        pdf_contents = [
            BytesIO(file.read()) 
            for file in files 
            if file.content_type == "application/pdf"
        ]
    
        if not pdf_contents:
            return {"error": "No valid PDF files uploaded."}

        ingestion_agent = DataIngestionAgent()
        docs, vector_db = ingestion_agent.process_documents(pdf_contents)

        topic_agent = TopicExtractorAgent(vector_db=vector_db)
        topic_data = topic_agent.extract_topics(docs)
        print(f"Embeddings created with {len(docs)} chunks.")
        return {
            "data": topic_data
        }

    except Exception as e:
        return {"error": str(e)}
