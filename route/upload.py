from flask import Blueprint, request
from io import BytesIO
from logic.topic_extract import TopicExtractorAgent
from logic.data_ingest import DataIngestionAgent

router = Blueprint('upload', __name__)

@router.route("/ping/", methods=['GET'])
def ping():
    return {"status": "ok", "message": "pong"}


@router.route("/upload/", methods=['POST'])
def upload_files():
    try:
        # Check if files were uploaded
        if not request.files:
            return {"error": "No files uploaded."}

        files = request.files.getlist('files')
        
        # Read PDF content
        pdf_contents = [
            BytesIO(file.read()) 
            for file in files 
            if file.content_type == "application/pdf"
        ]
        
        if not pdf_contents:
            return {"error": "No valid PDF files uploaded."}

        # Process documents
        ingestion_agent = DataIngestionAgent()
        docs, _ = ingestion_agent.process_documents(pdf_contents)

        # Extract topics
        topic_agent = TopicExtractorAgent()
        topic_data = topic_agent.extract_topics(docs)

        # Return the processed results
        return {
            "data": topic_data
        }

    except Exception as e:
        return {"error": str(e)}
