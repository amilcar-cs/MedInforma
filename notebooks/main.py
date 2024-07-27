import os
import sys
import google.auth
import json
import pickle

sys.path.append(os.path.abspath(os.path.join('..')))
from packages.model import MedicalAssistantModel
from packages.database import DocumentDatabase
from packages.rag import Assistant
from packages.metrics import Evaluator

enviroment_dir = os.path.abspath(os.path.join('..', 'enviroment'))
JSON_KEY = os.path.join(enviroment_dir, 'kabooma.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = JSON_KEY

db = os.path.abspath(os.path.join('..', 'resources'))
CHROMA_PATH = os.path.abspath(os.path.join('..', 'db'))

with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == "__main__":
    creds, _ = google.auth.default(quota_project_id=config["CLOUD_PROJECT"])
    model = MedicalAssistantModel(credentials=creds, embedding_model_name=config["EMBEDDING_MODEL"], chat_model_name=config["LLM_MODEL"])
    embeddings = model.embeddings
    database = DocumentDatabase(chroma_path=CHROMA_PATH, embedding_function=embeddings)
    rag = Assistant(database=database, model=model)
    eval_rag = Evaluator(credentials=creds, model_name=config["EVAL_MODEL"], embedding_model_name=config["EMBEDDING_MODEL"])

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

df = eval_rag.evaluate_dataset(dataset.select(range(0, 2)))
print(df.head())