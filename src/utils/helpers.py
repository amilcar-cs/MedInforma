import google.auth
import json
import pickle
from typing import Any, Dict

import pandas as pd

def get_google_credentials(project_id: str) -> google.auth.credentials.Credentials:
    """
    Retrieve Google Cloud credentials for the specified project.

    Args:
        project_id (str): The Google Cloud project ID.

    Returns:
        google.auth.credentials.Credentials: The Google Cloud credentials.
    """
    credentials, _ = google.auth.default(quota_project_id=project_id)
    return credentials

def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
    """
    with open(file_path, 'r') as config_file:
        return json.load(config_file)

def load_pickle_data(file_path: str) -> Any:
    """
    Load data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: The loaded data object.
    """
    with open(file_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def save_dataframe_to_excel(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to an Excel file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        file_path (str): Path to save the Excel file.
    """
    dataframe.to_excel(file_path, index=False)

# Alias functions for specific use cases
load_metadata = load_pickle_data
load_dataset = load_pickle_data