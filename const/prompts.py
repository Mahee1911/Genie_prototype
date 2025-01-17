TOPIC_EXTRACT_PROMPT = """
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


TOPIC_COMBINE_PROMPT = """
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