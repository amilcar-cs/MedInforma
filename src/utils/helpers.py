import google.auth
import json
import pickle

def load_credentials(project_id: str):
    creds, _ = google.auth.default(quota_project_id=project_id)
    return creds

def load_configs(path: str):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_metadata(path: str):
    with open(path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata