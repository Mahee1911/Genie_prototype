from langchain.prompts import PromptTemplate
from const.key import MODEL
import os
from typing import Dict, List, Tuple, Any
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import LLMChain
import hashlib


class TopicExtractorAgent:
    def __init__(self, cache_dir: str = "topic_cache"):
        self.llm = ChatOpenAI(
            temperature=0.2,
            model=MODEL,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _validate_and_normalize_percentages(self, parsed_response: Dict) -> Dict:
        """Validate and normalize topic percentages to ensure correct sum and consistency"""
        total_percentage = sum(topic['percentage'] for topic in parsed_response['topics'])
        if total_percentage != 100:
            print(f"Warning: Total percentage is {total_percentage}. Normalizing percentages.")
            for topic in parsed_response['topics']:
                topic['percentage'] = topic['percentage'] / total_percentage * 100
        return parsed_response

    def _combine_topic_results(self, results: List[Dict], doc_name: str) -> Dict:
        """Combine and normalize topic results from multiple chunks and include citation content"""
        combine_template = """
        You are an expert topic analyzer. Combine these separate analyses into one coherent structure.
        Previous analyses: {text}

        Guidelines:
        1. Identify common themes across all analyses.
        2. Merge similar topics and adjust percentages accordingly.
        3. Ensure accurate percentage distribution.
        4. Prioritize topics that appear consistently across chunks.

        Return a JSON object with the structure:
        {{
            "topics": [
                {{
                    "name": "Specific Main Topic Name",
                    "percentage": number,
                    "value": "Text Content",
                    "pages": "Page Numbers and Line Ranges",
                    "subtopics": [
                        {{
                            "name": "Specific Subtopic Name",
                            "percentage": number,
                            "value": "Text Content",
                            "pages": "Page Numbers and Line Ranges",
                            "subsubtopics": [
                                {{
                                    "name": "Specific Sub-Subtopic Name",
                                    "percentage": number,
                                    "value": "Text Content",
                                    "pages": "Page Numbers and Line Ranges"
                                }} 
                            ]
                        }} 
                    ]
                }}
            ]
        }}

        Requirements:
        1. All percentages must be numbers.
        2. Main topics must sum to 100% and have their content and page info.
        3. Subtopic percentages must sum to their parent topic's percentage and have their content and page info.
        4. Sub-subtopic percentages must sum to their parent subtopic's percentage and have their content and page info.
        """
        
        prompt = PromptTemplate(input_variables=["text"], template=combine_template)
        combined_text = json.dumps(results)
        response = prompt | self.llm
        return json.loads(response.invoke({"text": combined_text}).content)

    def _validate_response(self, parsed_response: Dict) -> bool:
        """Validate if the response structure is correct"""
        try:
            if "topics" not in parsed_response or not isinstance(parsed_response["topics"], list):
                return False
            
            for topic in parsed_response["topics"]:
                if "percentage" not in topic or not isinstance(topic["percentage"], (int, float)):
                    return False
                if topic["percentage"] < 0 or topic["percentage"] > 100:
                    return False

            return True
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return False

    def extract_topics(self, docs: List) -> Dict:
        """Extract topics, subtopics, sub-subtopics with their percentages and citations"""
        template = """
        You are an investment analyst working in M&A. You are receiving a Confidential Information Memorandum (CIM), and you need to quickly analyze the content to assess whether the company is an interesting acquisition or investment target. Your task is to classify the content of the document into topics, subtopics, and sub-subtopics (mutually exclusive). For each subtopic and sub-subtopic, provide 2 to 3 lines of content directly from the PDF to define that subtopic, with the exact citation pointing to the specific lines in the document.

        Structure the response as follows:
        - Each main topic should include a list of subtopics with their name, percentage, content (2 to 3 lines of text from the document), and sub-subtopics if applicable.
        - Cite the exact location from the document (e.g., page number and line range) from which the content was extracted.
        - Ensure that the percentages of each level (main topic, subtopic, sub-subtopic) sum up appropriately.

        Return the response as a JSON object structured like this:

        {{
            "topics": [
                {{
                    "name": "Main Topic Name",
                    "percentage": number,
                    "value": "Content/Excerpt from Document (2-3 lines)",
                    "pages": "Page 3, Lines 15-18",
                    "subtopics": [
                        {{
                            "name": "Subtopic Name",
                            "percentage": number,
                            "value": "Content/Excerpt from Document (2-3 lines)",
                            "pages": "Page 5, Lines 20-23",
                            "subsubtopics": [
                                {{
                                    "name": "Sub-Subtopic Name",
                                    "percentage": number,
                                    "value": "Content/Excerpt from Document (2-3 lines)",
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
        
        prompt = PromptTemplate(input_variables=["text"], template=template)

        chunk_size = 2  # Reduced chunk size for more detailed analysis
        all_results = []
        print(f"Processing {len(docs)} documents in chunks of {chunk_size}")
        
        for i in range(0, len(docs), chunk_size):
            chunk_docs = docs[i:i+chunk_size]
            chunk_text = " ".join([doc.page_content for doc in chunk_docs])
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"chunk_{chunk_hash}.json")
            
            if os.path.exists(cache_file):
                print(f"Loading cached results for chunk {i}")
                try:
                    with open(cache_file, 'r') as f:
                        cached_response = json.load(f)
                    all_results.append(cached_response)
                    continue
                except Exception as e:
                    print(f"Error loading cache for chunk {i}: {str(e)}")
            
            print(f"Processing chunk {i}")
            try:
                result = prompt | self.llm
                parsed_response = json.loads(result.invoke({"text": chunk_text}).content)
                parsed_response = self._validate_and_normalize_percentages(parsed_response)

                if self._validate_response(parsed_response):
                    with open(cache_file, 'w') as f:
                        json.dump(parsed_response, f, indent=4)
                    all_results.append(parsed_response)
                else:
                    print(f"Invalid response for chunk {i}, skipping.")
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")

        # print(f"Combining results from {len(all_results)} chunks")
        # final_result = self._combine_topic_results(all_results, "Combined Results")
       
        # # Convert the final result to hierarchy and return
        # print("Converting to hierarchy")
        # return self.flatten_hierarchy(final_result["topics"])

        final_result = self._combine_topic_results(all_results, "Combined Results")

        # Step 1: Check if 'final_result["topics"]' is a list
        # Check if 'topics' is a list
        if isinstance(final_result.get("topics"), list):
            print("Found 'topics' as a list, proceeding to flatten.")
            # Step 2: Flatten the hierarchy
            flat_list = self.flatten_hierarchy(final_result["topics"])
        else:
            print(f"'topics' is not a list! Please check the structure of 'final_result': {final_result}")
            flat_list = []  # Set an empty list or handle the error accordingly


        # Now you can return or work with 'flat_list'
        return flat_list

    def flatten_hierarchy(self, data, parent_id=''):
        """Flatten the data into a list with only id, parent, name, and other relevant fields."""
        flat_list = []
        current_id = 1

        # Debugging the type of data
        if not isinstance(data, list):
            print("Received data is not a list:", type(data))
            raise ValueError("Input data should be a list")

        for topic in data:
            # Create a unique ID for each level
            topic_id = f"{parent_id}.{current_id}" if parent_id else f"{current_id}"

            # Add the current topic to the flat list
            flat_list.append({
                "id": topic_id,
                "parent": parent_id,
                "name": topic["name"],
                "value": topic.get("value", ""),
                "percentage": topic.get("percentage", 0),
                "pages": topic.get("pages", ""),
            })

            # Recursively process subtopics (direct children)
            if "subtopics" in topic and topic["subtopics"]:
                flat_list.extend(self.flatten_hierarchy(topic["subtopics"], topic_id))

            # Recursively process subsubtopics (children of subtopics)
            if "subsubtopics" in topic and topic["subsubtopics"]:
                flat_list.extend(self.flatten_hierarchy(topic["subsubtopics"], topic_id))

            current_id += 1

        return flat_list

            