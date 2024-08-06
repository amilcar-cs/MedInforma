import os
import pickle
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm
from datasets import Dataset
from unidecode import unidecode

# Set up file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
QUESTION_ANSWERS_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'questions_answers.csv')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

# RAG API configuration
RAG_API_URL = "http://localhost:8000/ask"
RAG_API_HEADERS = {"Content-Type": "application/json"}

def query_rag_api(question: str) -> Optional[Dict]:
    """
    Query the RAG API with a given question.

    Args:
        question (str): The question to be sent to the API.

    Returns:
        Optional[Dict]: The API response or None if an error occurs.
    """
    payload = {"text": question}
    try:
        response = requests.post(RAG_API_URL, headers=RAG_API_HEADERS, json=payload)
        response.raise_for_status()
        return response.json()['answer']
    except requests.RequestException as e:
        print(f"Error querying the API: {e}")
        return None

def read_questions_and_answers(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Read questions and ground truths from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Tuple[List[str], List[str]]: Lists of questions and ground truths.
    """
    questions, ground_truths = [], []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['question'])
            ground_truths.append(row['ground_truths'])
    return questions, ground_truths

def process_query(query: str, ground_truth: str) -> Optional[Dict]:
    """
    Process a single query-ground truth pair using the RAG API.

    Args:
        query (str): The question to be processed.
        ground_truth (str): The ground truth answer.

    Returns:
        Optional[Dict]: Processed data or None if an error occurs.
    """
    try:
        response = query_rag_api(query)
        contexts = response['context']
        answer = response['completion']
        if answer is None:
            return None
        
        return {
            "question": unidecode(query),
            "ground_truth": unidecode(ground_truth),
            "answer": unidecode(answer),
            "contexts": [unidecode(context) for context in contexts]
        }
    except Exception as e:
        print(f"Error processing query: {query}, ground_truth: {ground_truth}. Error: {e}")
        return None

def create_dataset_from_queries(queries: List[str], ground_truths: List[str], max_workers: int = 12) -> Dataset:
    """
    Create a dataset from a list of queries and ground truths.

    Args:
        queries (List[str]): List of questions.
        ground_truths (List[str]): List of ground truth answers.
        max_workers (int): Maximum number of concurrent workers.

    Returns:
        Dataset: The created dataset.
    """
    processed_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_query, query, ground_truth) 
                   for query, ground_truth in zip(queries, ground_truths)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    processed_data.append(result)
            except Exception as e:
                print(f"Error in future: {e}")

    data_dict = {
        "question": [item["question"] for item in processed_data],
        "ground_truth": [item["ground_truth"] for item in processed_data],
        "answer": [item["answer"] for item in processed_data],
        "contexts": [item["contexts"] for item in processed_data]
    }

    return Dataset.from_dict(data_dict)

def save_dataset_to_file(dataset: Dataset, save_path: str) -> None:
    """
    Save a dataset to a file using pickle.

    Args:
        dataset (Dataset): The dataset to be saved.
        save_path (str): Path where the dataset will be saved.
    """
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

def main():
    """
    Main function to orchestrate the dataset creation and saving process.
    """
    questions, ground_truths = read_questions_and_answers(QUESTION_ANSWERS_PATH)
    dataset = create_dataset_from_queries(questions, ground_truths)
    save_dataset_to_file(dataset, os.path.join(PROCESSED_DATA_PATH, "dataset.pkl"))

if __name__ == "__main__":
    main()