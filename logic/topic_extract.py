from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import asyncio
import json
from typing import Dict, List
from const.key import MODEL
from concurrent.futures import ThreadPoolExecutor

class TopicExtractorAgent:
    def __init__(self, vector_db: FAISS = None):
        if vector_db:
            self.vector_db = vector_db
        else:
            raise ValueError("A vector database is required for topic extraction.")

        self.llm = ChatOpenAI(
            temperature=0.2,
            model=MODEL,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        self.retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.executor = ThreadPoolExecutor(max_workers=8)

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
                    "citation": "Text Content",
                    "pages": "Page Numbers and Line Ranges",
                    "subtopics": [
                        {{
                            "name": "Subtopic Name",
                            "value": number,
                            "citation": "Text Content",
                            "pages": "Page Numbers and Line Ranges",
                            "subsubtopics": [
                                {{
                                    "name": "Sub-Subtopic Name",
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

        Text for analysis: {text}
        """       
        self.prompt = PromptTemplate(input_variables=["text"], template=self.template)
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

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
        
        self.combine_llm_chain = LLMChain(llm=self.llm, prompt=self.combine_prompt)

    def _distribute_values_proportionally(self, data: Dict) -> Dict:
        """Distribute values based on word or line length"""
        for topic in data["topics"]:
            topic_length = len(topic.get("citation", "").split())
            topic["value"] = round(topic["value"], 2)
            if "subtopics" in topic:
                total_sub_lengths = sum(len(sub["citation"].split()) for sub in topic["subtopics"])
                for sub in topic["subtopics"]:
                    sub_length = len(sub["citation"].split())
                    sub["value"] = round((sub_length / total_sub_lengths) * topic["value"], 2)
                    if "subsubtopics" in sub:
                        total_subsub_lengths = sum(len(subsub["citation"].split()) for subsub in sub["subsubtopics"])
                        for subsub in sub["subsubtopics"]:
                            subsub_length = len(subsub["citation"].split())
                            subsub["value"] = round((subsub_length / total_subsub_lengths) * sub["value"], 2)
        return data

    def _validate_response(self, response: Dict) -> bool:
        """Check if the response is valid"""
        try:
            for topic in response["topics"]:
                if not isinstance(topic["value"], (int, float)):
                    return False
                if "subtopics" in topic:
                    sub_total = sum(sub["value"] for sub in topic["subtopics"])
                    if round(sub_total, 2) != round(topic["value"], 2):
                        return False
                    for sub in topic["subtopics"]:
                        if "subsubtopics" in sub:
                            subsub_total = sum(subsub["value"] for subsub in sub["subsubtopics"])
                            if round(subsub_total, 2) != round(sub["value"], 2):
                                return False
            return True
        except KeyError:
            return False

    async def process_chunk(self, chunk_docs: List, chunk_index: int) -> Dict:
        """Process a single chunk of documents using the LLMChain."""
        chunk_text = " ".join([doc.page_content for doc in chunk_docs])
        print(f"Processing chunk {chunk_index} with {len(chunk_docs)} documents")

        try:
            context = self.retriever.get_relevant_documents(chunk_text) 

            input_text = chunk_text + "\n" + "\n".join([getattr(doc, 'page_content', '') for doc in context])

            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.llm_chain.run({"text": input_text})
            )

            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                print(f"Invalid JSON response for chunk {chunk_index}: {response}")
                return None
            parsed_response = self._distribute_values_proportionally(parsed_response)

            if self._validate_response(parsed_response):
                return parsed_response
            else:
                print(f"Invalid response for chunk {chunk_index}, skipping.")
                return None
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            return None

    async def extract_topics_async(self, docs: List) -> List:
        """Asynchronous version of extract_topics with LLMChain."""
        base_chunk_size = len(docs) // 8
        remainder = len(docs) % 8
        tasks = []
        start = 0

        print(f"Processing {len(docs)} documents in chunks, with base chunk size {base_chunk_size} and {remainder} remainder documents.")

        for i in range(8):
            current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            chunk_docs = docs[start:start + current_chunk_size]

            task = asyncio.create_task(self.process_chunk(chunk_docs, i))
            tasks.append(task)

            start += current_chunk_size

        results = await asyncio.gather(*tasks)

        all_results = [result for result in results if result is not None]
        final_result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self._combine_topic_results(all_results)
        )

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
        
    def _combine_topic_results(self, results: List[Dict]) -> Dict:
        """Combine and normalize topic results from multiple chunks."""
        combined_text = json.dumps(results)
        response = self.combine_llm_chain.run({"text": combined_text})
        return json.loads(response)

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
