import os
import sys
from dotenv import load_dotenv

# Setting System Paths and Environment Variables
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Imports of internal modules
from src.packages.model import MedicalAssistantModel
from src.packages.database import DocumentDatabase
from src.packages.rag import Assistant
from src.packages.metrics import Evaluator
import src.utils.helpers as helpers

# Path and environment variable settings
dotenv_path = os.path.join('..', '..', 'configs', '.env')
load_dotenv(dotenv_path)

JSON_KEY = os.path.abspath(os.path.join('..', '..', 'configs', 'service_account_key.json'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = JSON_KEY

CHROMA_PATH = os.path.abspath(os.path.join('..', '..', 'src', 'db', 'chroma'))
CONFIG_PATH = os.path.abspath(os.path.join('..', '..', 'configs', 'config.json'))

DATASET_PATH = os.path.abspath(os.path.join('..', '..', 'data', 'processed', 'dataset.pkl'))

# JSON file configuration load
config = helpers.load_json_config(CONFIG_PATH)

# Initialisation of credentials and templates
project_id = os.getenv('CLOUD_PROJECT')
creds = helpers.get_google_credentials(project_id)

model = MedicalAssistantModel(
    credentials=creds, 
    embedding_model_name=config["EMBEDDING_MODEL"], 
    chat_model_name=config["LLM_MODEL"]
)
embeddings = model.embeddings
database = DocumentDatabase(chroma_path=CHROMA_PATH, embedding_function=embeddings)
rag = Assistant(database=database, model=model)
eval_rag = Evaluator(
    credentials=creds, 
    model_name=config["EVAL_MODEL"], 
    embedding_model_name=config["EMBEDDING_MODEL"]
)

#Evaluation
dataset = helpers.load_dataset(DATASET_PATH)
df = eval_rag.evaluate_dataset(dataset)

# Save the evaluation results to a file
helpers.save_dataframe_to_excel(df, 'results/output.xlsx')