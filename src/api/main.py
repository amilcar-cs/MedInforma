import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

# Setting System Paths and Environment Variables
sys.path.append(os.path.abspath(os.path.join('..')))

# Imports of internal modules
from packages.model import MedicalAssistantModel
from packages.database import DocumentDatabase
from packages.rag import Assistant
import utils.helpers as helpers

# Path and environment variable settings
dotenv_path = os.path.join('..', '..', 'configs', '.env')
load_dotenv(dotenv_path)

JSON_KEY = os.path.abspath(os.path.join('..', '..', 'configs', 'service_account_key.json'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = JSON_KEY

db = os.path.abspath(os.path.join('..', 'resources'))
CHROMA_PATH = os.path.abspath(os.path.join('..', 'db', 'chroma'))
METADATA_PATH = os.path.abspath(os.path.join('..', '..', 'data', 'processed', 'medline_metadata.pkl'))
PROCESSED_DATA_PATH = os.path.abspath(os.path.join('..', '..', 'data', 'processed', 'files'))
CONFIG_PATH = os.path.abspath(os.path.join('..', '..', 'configs', 'config.json'))

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

if os.path.exists(CHROMA_PATH):
    database = DocumentDatabase(chroma_path=CHROMA_PATH, embedding_function=embeddings)
else:
    print(METADATA_PATH)
    metadata = helpers.load_metadata(METADATA_PATH)
    database = DocumentDatabase(
            chroma_path=CHROMA_PATH,
            embedding_function=embeddings,
            create_new=True,
            files_directory=PROCESSED_DATA_PATH,
            metadata_dict=metadata,
            batch_size=config["BATCH_SIZE"],
        )

rag = Assistant(database=database, model=model)

# FastAPI application initialisation
app = FastAPI(
    title="MedInforma", 
    description="MedInforma provides accurate health and medical information based on a Spanish-language dataset from Medline. Ideal for healthcare professionals and medical students seeking reliable data."
)

class Question(BaseModel):
    text: str

# API Routes
@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = rag.ask(question.text)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to MedInforma API"}

# Entry point of the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)