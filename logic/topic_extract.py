from typing import List, Dict, Any
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from const.key import MODEL

class TopicExtractorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.2,
            model=MODEL,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    def extract_topics(self, docs: List) -> Dict:
        """Extract topics and subtopics with their percentages."""
        template = """
        You are a precise document analyzer. Analyze the following text and identify the main topics and subtopics.
        Focus on the key themes and concepts that appear most frequently and are most significant.

        Guidelines:
        1. Identify 3-5 main topics that cover the major themes
        2. For each main topic, identify 2-4 specific subtopics
        3. Calculate accurate percentage coverage based on content relevance and frequency
        4. Ensure percentages reflect actual content distribution
        5. Be specific and precise in topic naming

        Return the analysis as a JSON object with this exact structure:
        {{
            "topics": [
                {{
                    "name": "Specific Main Topic Name",
                    "percentage": number,
                    "subtopics": [
                        {{"name": "Specific Subtopic Name", "percentage": number}},
                        {{"name": "Specific Subtopic Name", "percentage": number}}
                    ]
                }}
            ]
        }}
        
        Text for analysis: {text}
        
        Requirements:
        1. All percentages must be numbers and sum to 100
        2. Only include topics with >5% coverage
        3. Main topic percentages must sum to 100%
        4. Subtopic percentages must sum to their parent topic's percentage
        5. Be specific and avoid generic topic names
        6. Use precise terminology from the text
        """
        
        prompt = PromptTemplate(input_variables=["text"], template=template)
        all_text = " ".join([doc.page_content for doc in docs])

        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": all_text})
            parsed_response = json.loads(response.content)
            return parsed_response
        except Exception as e:
            raise ValueError(f"Failed to process topics: {str(e)}")
