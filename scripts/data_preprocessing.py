import re
import os
import pickle
import xml.etree.ElementTree as ET
from typing import Dict, List

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from unidecode import unidecode

# Set up file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MEDLINE_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'medline.xml')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'files')

def normalize_filename(filename: str) -> str:
    """
    Normalize and sanitize a filename.

    Args:
        filename (str): The original filename.

    Returns:
        str: A normalized and sanitized filename.
    """
    normalized_text = unidecode(filename).lower()
    return re.sub(r'[^a-z0-9]+', '_', normalized_text).strip('_')

def save_markdown_file(html_content: str, title: str, save_path: str) -> None:
    """
    Convert HTML content to Markdown and save it as a file.

    Args:
        html_content (str): The HTML content to convert.
        title (str): The title of the content (used for filename).
        save_path (str): The directory to save the file in.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    markdown_content = md(str(soup))
    filename = os.path.join(save_path, f'{normalize_filename(title)}.md')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

def extract_subtitles(health_topic: ET.Element) -> List[str]:
    """
    Extract subtitles (also-called) from a health topic.

    Args:
        health_topic (ET.Element): The health topic XML element.

    Returns:
        List[str]: A list of subtitles.
    """
    return [also_called.text for also_called in health_topic.findall("also-called")]

def extract_related_topics(health_topic: ET.Element) -> List[str]:
    """
    Extract related topics from a health topic.

    Args:
        health_topic (ET.Element): The health topic XML element.

    Returns:
        List[str]: A list of related topics.
    """
    return [related.text for related in health_topic.findall("related-topic")]

def process_health_topic(health_topic: ET.Element, save_path: str) -> Dict[str, Dict[str, any]]:
    """
    Process a single health topic and return its metadata.

    Args:
        health_topic (ET.Element): The health topic XML element.
        save_path (str): The directory to save the Markdown file in.

    Returns:
        Dict[str, Dict[str, any]]: A dictionary containing the topic's metadata.
    """
    title = health_topic.attrib['title']
    subtitles = extract_subtitles(health_topic)
    related_topics = extract_related_topics(health_topic)
    content = f"<h1>{title}</h1>\n{health_topic.find('full-summary').text}"
    url = health_topic.attrib['url']

    save_markdown_file(content, title, save_path)

    return {
        normalize_filename(title): {
            'also_called': subtitles,
            'related_topic': related_topics,
            'url': url
        }
    }

def process_medline_xml(load_path: str, save_path: str) -> Dict[str, Dict[str, any]]:
    """
    Process the MEDLINE XML file and extract health topics.

    Args:
        load_path (str): The path to the MEDLINE XML file.
        save_path (str): The directory to save individual Markdown files.

    Returns:
        Dict[str, Dict[str, any]]: A dictionary containing metadata for all processed topics.
    """
    tree = ET.parse(load_path)
    root = tree.getroot()
    all_topics_metadata = {}

    for health_topic in root.findall('.//health-topic[@language="Spanish"]'):
        topic_metadata = process_health_topic(health_topic, save_path)
        all_topics_metadata.update(topic_metadata)

    return all_topics_metadata

def save_metadata(metadata: Dict[str, Dict[str, any]], save_path: str) -> None:
    """
    Save metadata dictionary as a pickle file.

    Args:
        metadata (Dict[str, Dict[str, any]]): The metadata to save.
        save_path (str): The directory to save the pickle file in.
    """
    with open(os.path.join(save_path, "medline_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def main():
    """
    Main function to process the MEDLINE XML file, extract health topics, and save the metadata.
    """
    medline_metadata = process_medline_xml(MEDLINE_PATH, DATA_PATH)
    save_metadata(medline_metadata, METADATA_PATH)

if __name__ == "__main__":
    main()