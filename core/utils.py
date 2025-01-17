from typing import Dict, List


def validate_response(response: Dict) -> bool:
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
                        subsub_total = sum(
                            subsub["value"] for subsub in sub["subsubtopics"]
                        )
                        if round(subsub_total, 2) != round(sub["value"], 2):
                            return False
        return True
    except KeyError:
        return False


def flatten_hierarchy(data, parent_id=""):
    """Flatten the data into a list with only id, parent, name, and other relevant fields"""
    flat_list = []
    current_id = 1

    if not isinstance(data, list):
        print("Received data is not a list:", type(data))
        raise ValueError("Input data should be a list")

    for topic in data:
        topic_id = f"{parent_id}.{current_id}" if parent_id else f"{current_id}"

        flat_list.append(
            {
                "id": topic_id,
                "parent": parent_id,
                "name": topic["name"],
                "value": topic.get("value", ""),
                "citation": topic.get("citation", ""),
                "pages": topic.get("pages", ""),
            }
        )

        if "subtopics" in topic and topic["subtopics"]:
            flat_list.extend(flatten_hierarchy(topic["subtopics"], topic_id))

        if "subsubtopics" in topic and topic["subsubtopics"]:
            flat_list.extend(
                flatten_hierarchy(topic["subsubtopics"], topic_id)
            )

        current_id += 1

    return flat_list