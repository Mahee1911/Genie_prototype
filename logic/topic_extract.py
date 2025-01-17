from typing import Dict, List

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from core.utils import validate_response, flatten_hierarchy
from const.key import MODEL
from const.prompts import TOPIC_EXTRACT_PROMPT, TOPIC_COMBINE_PROMPT


class TopicExtractorAgent:
    def __init__(self, vector_db: FAISS = None):
        if vector_db:
            self.vector_db = vector_db
        else:
            raise ValueError("A vector database is required for topic extraction.")

        self.llm = ChatOpenAI(
            temperature=0.2,
            model=MODEL,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        self.executor = ThreadPoolExecutor(max_workers=8)

        self.template = TOPIC_EXTRACT_PROMPT
        self.prompt = PromptTemplate(input_variables=["text"], template=self.template)

        self.combine_template = TOPIC_COMBINE_PROMPT
        self.combine_prompt = PromptTemplate(
            input_variables=["text"], template=self.combine_template
        )

    def _distribute_values_proportionally(self, data: Dict) -> Dict:
        """Distribute values based on word or line length"""
        for topic in data["topics"]:
            topic_length = len(topic.get("citation", "").split())
            topic["value"] = round(topic["value"], 2)
            if "subtopics" in topic:
                total_sub_lengths = sum(
                    len(sub["citation"].split()) for sub in topic["subtopics"]
                )
                for sub in topic["subtopics"]:
                    sub_length = len(sub["citation"].split())
                    sub["value"] = round(
                        (sub_length / total_sub_lengths) * topic["value"], 2
                    )
                    if "subsubtopics" in sub:
                        total_subsub_lengths = sum(
                            len(subsub["citation"].split())
                            for subsub in sub["subsubtopics"]
                        )
                        for subsub in sub["subsubtopics"]:
                            subsub_length = len(subsub["citation"].split())
                            subsub["value"] = round(
                                (subsub_length / total_subsub_lengths) * sub["value"], 2
                            )
        return data

    async def process_chunk(self, chunk_docs: List, chunk_index: int) -> Dict:
        """Process a single chunk of documents and retrieve context for LLM"""
        chunk_text = " ".join([doc.page_content for doc in chunk_docs])
        print(f"Processing chunk {chunk_index} with {len(chunk_docs)} documents")

        try:
            context = self.retriever.get_relevant_documents(chunk_text)

            input_text = (
                chunk_text + "\n" + "\n".join([doc.page_content for doc in context])
            )

            result = self.prompt | self.llm
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: result.invoke({"text": input_text})
            )

            parsed_response = json.loads(response.content)
            parsed_response = self._distribute_values_proportionally(parsed_response)

            if validate_response(parsed_response):
                return parsed_response
            else:
                print(f"Invalid response for chunk {chunk_index}, skipping.")
                return None
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            return None

    async def extract_topics_async(self, docs: List) -> List:
        """Asynchronous version of extract_topics with RAG concept"""
        base_chunk_size = len(docs) // 8
        remainder = len(docs) % 8
        tasks = []
        start = 0

        print(
            f"Processing {len(docs)} documents in chunks, with base chunk size {base_chunk_size} and {remainder} remainder documents."
        )

        for i in range(8):
            current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            chunk_docs = docs[start : start + current_chunk_size]

            task = asyncio.create_task(self.process_chunk(chunk_docs, i))
            tasks.append(task)

            start += current_chunk_size

        results = await asyncio.gather(*tasks)

        all_results = [result for result in results if result is not None]
        final_result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self._combine_topic_results(all_results, "Combined Results"),
        )

        if isinstance(final_result.get("topics"), list):
            print("Found 'topics' as a list, proceeding to flatten.")
            flat_list = flatten_hierarchy(final_result["topics"])
        else:
            print(
                f"'topics' is not a list! Please check the structure of 'final_result': {final_result}"
            )
            flat_list = []

        return flat_list

    def extract_topics(self, docs: List) -> List:
        """Synchronous wrapper for extract_topics_async"""
        return asyncio.run(self.extract_topics_async(docs))

    def _combine_topic_results(self, results: List[Dict], doc_name: str) -> Dict:
        """Combine and normalize topic results from multiple chunks"""
        combined_text = json.dumps(results)
        response = self.combine_prompt | self.llm
        return json.loads(response.invoke({"text": combined_text}).content)

    def __del__(self):
        """Cleanup method"""
        self.executor.shutdown(wait=True)
