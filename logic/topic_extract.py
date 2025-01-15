from langchain.prompts import PromptTemplate
from const.key import MODEL
from typing import Dict, List
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

class TopicExtractorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.2,
            model=MODEL,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Define the main analysis template
        self.template = """
        You are an investment analyst working in M&A. You are receiving a Confidential Information Memorandum (CIM), and you need to quickly analyze the content to assess whether the company is an interesting acquisition or investment target. Your task is to classify the content of the document into topics, subtopics, and sub-subtopics (mutually exclusive). For each subtopic and sub-subtopic, provide 2 to 3 lines of content directly from the PDF to define that subtopic, with the exact citation pointing to the specific lines in the document.

        Structure the response as follows:
        - Each main topic should include a list of subtopics with their name, percentage value, content (2 to 3 lines of text from the document), and sub-subtopics if applicable.
        - Cite the exact location from the document (e.g., page number and line range) from which the content was extracted.
        - Ensure that the percentage values of each level (main topic, subtopic, sub-subtopic) sum up appropriately.

        Return the response as a JSON object structured like this:
        {{
            "topics": [
                {{
                    "name": "Main Topic Name",
                    "value": number,
                    "citation": "Content/Excerpt from Document (2-3 lines)",
                    "pages": "Page 3, Lines 15-18",
                    "subtopics": [
                        {{
                            "name": "Subtopic Name",
                            "value": number,
                            "citation": "Content/Excerpt from Document (2-3 lines)",
                            "pages": "Page 5, Lines 20-23",
                            "subsubtopics": [
                                {{
                                    "name": "Sub-Subtopic Name",
                                    "value": number,
                                    "citation": "Content/Excerpt from Document (2-3 lines)",
                                    "pages": "Page 8, Lines 10-12"
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}

        Text for analysis: {text}
        """
        self.prompt = PromptTemplate(input_variables=["text"], template=self.template)

        # Define the combine template
        self.combine_template = """
        You are an expert topic analyzer. Combine these separate analyses into one coherent structure.
        Previous analyses: {text}

        Guidelines:
        1. Identify common themes across all analyses.
        2. Merge similar topics and adjust values accordingly.
        3. Ensure accurate value distribution.
        4. Prioritize topics that appear consistently across chunks.

        Return a JSON object with the structure:
        {{
            "topics": [
                {{
                    "name": "Specific Main Topic Name",
                    "value": number,
                    "citation": "Text Content",
                    "pages": "Page Numbers and Line Ranges",
                    "subtopics": [
                        {{
                            "name": "Specific Subtopic Name",
                            "value": number,
                            "citation": "Text Content",
                            "pages": "Page Numbers and Line Ranges",
                            "subsubtopics": [
                                {{
                                    "name": "Specific Sub-Subtopic Name",
                                    "value": number,
                                    "citation": "Text Content",
                                    "pages": "Page Numbers and Line Ranges"
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}

        Requirements:
        1. All values must be numbers.
        2. Main topics must sum to 100% and have their content and page info.
        3. Subtopic values must sum to their parent topic's values and have their content and page info.
        4. Sub-subtopic values must sum to their parent subtopic's values and have their content and page info.
        """
        self.combine_prompt = PromptTemplate(input_variables=["text"], template=self.combine_template)

    async def process_chunk(self, chunk_docs: List, chunk_index: int) -> Dict:
        """Process a single chunk of documents"""
        chunk_text = " ".join([doc.page_content for doc in chunk_docs])
        print(f"Processing chunk {chunk_index} with {len(chunk_docs)} documents")
        
        try:
            result = self.prompt | self.llm
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: result.invoke({"text": chunk_text})
            )
            
            parsed_response = json.loads(response.content)
            parsed_response = self._validate_and_normalize_percentages(parsed_response)

            if self._validate_response(parsed_response):
                return parsed_response
            else:
                print(f"Invalid response for chunk {chunk_index}, skipping.")
                return None
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            return None

    async def extract_topics_async(self, docs: List) -> List:
        """Asynchronous version of extract_topics"""
        base_chunk_size = len(docs) // 8
        remainder = len(docs) % 8
        tasks = []
        start = 0

        print(f"Processing {len(docs)} documents in chunks, with base chunk size {base_chunk_size} and {remainder} remainder documents.")

        # Create tasks for all chunks
        for i in range(8):
            current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            chunk_docs = docs[start:start + current_chunk_size]
            
            task = asyncio.create_task(self.process_chunk(chunk_docs, i))
            tasks.append(task)
            
            start += current_chunk_size

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        all_results = [r for r in results if r is not None]

        # Combine results
        final_result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self._combine_topic_results(all_results, "Combined Results")
        )

        # Flatten the hierarchy
        if isinstance(final_result.get("topics"), list):
            print("Found 'topics' as a list, proceeding to flatten.")
            flat_list = self.flatten_hierarchy(final_result["topics"])
        else:
            print(f"'topics' is not a list! Please check the structure of 'final_result': {final_result}")
            flat_list = []

        return flat_list

    def extract_topics(self, docs: List) -> List:
        """Synchronous wrapper for extract_topics_async"""
        return asyncio.run(self.extract_topics_async(docs))

    def _validate_and_normalize_percentages(self, parsed_response: Dict) -> Dict:
        """Validate and normalize topic percentages"""
        total_percentage = sum(topic['value'] for topic in parsed_response['topics'])
        if total_percentage != 100:
            print(f"Warning: Total percentage is {total_percentage}. Normalizing percentages.")
            for topic in parsed_response['topics']:
                topic['value'] = topic['value'] / total_percentage * 100
        return parsed_response

    def _combine_topic_results(self, results: List[Dict], doc_name: str) -> Dict:
        """Combine and normalize topic results from multiple chunks"""
        combined_text = json.dumps(results)
        response = self.combine_prompt | self.llm
        return json.loads(response.invoke({"text": combined_text}).content)

    def _validate_response(self, parsed_response: Dict) -> bool:
        """Validate if the response structure is correct"""
        try:
            if "topics" not in parsed_response or not isinstance(parsed_response["topics"], list):
                return False

            for topic in parsed_response["topics"]:
                if "value" not in topic or not isinstance(topic["value"], (int, float)):
                    return False
                if topic["value"] < 0 or topic["value"] > 100:
                    return False

            return True
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return False

    def flatten_hierarchy(self, data, parent_id=''):
        """Flatten the data into a list with only id, parent, name, and other relevant fields"""
        flat_list = []
        current_id = 1

        if not isinstance(data, list):
            print("Received data is not a list:", type(data))
            raise ValueError("Input data should be a list")

        for topic in data:
            topic_id = f"{parent_id}.{current_id}" if parent_id else f"{current_id}"

            flat_list.append({
                "id": topic_id,
                "parent": parent_id,
                "name": topic["name"],
                "value": topic.get("value", ""),
                "citation": topic.get("citation", ""),
                "pages": topic.get("pages", ""),
            })

            if "subtopics" in topic and topic["subtopics"]:
                flat_list.extend(self.flatten_hierarchy(topic["subtopics"], topic_id))

            if "subsubtopics" in topic and topic["subsubtopics"]:
                flat_list.extend(self.flatten_hierarchy(topic["subsubtopics"], topic_id))

            current_id += 1

        return flat_list

    def __del__(self):
        """Cleanup method"""
        self.executor.shutdown(wait=True)
