from fastapi import FastAPI, UploadFile, File, APIRouter
from typing import List, Dict, Any
from io import BytesIO
from logic.topic_extract import TopicExtractorAgent
from logic.data_ingest import DataIngestionAgent

router = APIRouter()


@router.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Read PDF content
        pdf_contents = [BytesIO(await file.read()) for file in files if file.content_type == "application/pdf"]
        if not pdf_contents:
            return {"error": "No valid PDF files uploaded."}

        # Process documents
        ingestion_agent = DataIngestionAgent()
        docs, _ = ingestion_agent.process_documents(pdf_contents)

        # Extract topics
        topic_agent = TopicExtractorAgent()
        topic_data = topic_agent.extract_topics(docs)

        # Return the processed results
        return {"files": [file.filename for file in files], "analysis": topic_data}

    except Exception as e:
        return {"error": str(e)}
    

