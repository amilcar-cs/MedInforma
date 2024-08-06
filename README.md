# MedInforma

**MedInforma** is an intelligent medical assistance tool designed to help users with health-related questions. **MedInforma** is a Retrieval-Augmented Generation (RAG) application that combines the power of Large Language Models (LLMs) with a comprehensive medical knowledge base. This project aims to provide accurate and contextual responses to medical queries.

---

## Features

- **RAG Technology**: Utilizes Retrieval-Augmented Generation to enhance the accuracy and relevance of responses.
- **Medical Knowledge Base**: Powered by [MedlinePlus](https://medlineplus.gov/spanish/) data, ensuring a rich source of medical information.
- **LLM Integration**: Leverages advanced language models for natural language understanding and generation.
- **User-Friendly Assistance**: Designed to help individuals access reliable medical information easily.

## Purpose

The primary goal of MedInforma is to serve as a reliable first point of contact for people (spanish speakers) seeking medical information. By combining extensive medical data with state-of-the-art language processing, MedInforma strives to provide helpful and accurate responses to a wide range of health-related questions.

## How to run
It is necessary to know that this application is primarily designed to run as an API using GET and POST requests. In this case, I will show the necessary steps to run this API locally using the [FastAPI](https://fastapi.tiangolo.com/) framework.

1. **Install requirements:** This project runs in python using external libraries and API's like VertexAI. It is therefore essential to install all the requirements to successfully run this project. To do this, simply locate the "MEDLINE_RAG" directory and run the command ```pip install .```

2. **.env Configuration and Service Account Key:** This project requires a LLM for its operation. In this case, I have chosen to use Google's [Vertex AI API](https://console.cloud.google.com/marketplace/product/google/aiplatform.googleapis.com) to take advantage of its Gemini 1.5 model. One of the requirements to use this API is to create a project in Google Cloud and generate a service client key. Both requirements are sensitive and should be handled with discretion, as each use of the API generates a cost for the owner. Therefore, it is important to store this information in environment variables and not to share it with third parties.
To add your cloud project name and service account key, go to the "configs" directory and create the following files:
    * An .env file following variable:

    ```javascript
    CLOUD_PROJECT =
    ```

    * A .json file that includes the service account key named as `service_account_key.json`. The content of this file should looks like this:

    ```javascript
    {
      "type": "service_account",
      "project_id": "...",
      "private_key_id": "...",
      "private_key": "...",
      "client_email": "...",
      "client_id": "...",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "...",
      "universe_domain": "googleapis.com"
    }
    ```

3. **API use:** To run the API, first go to the "api" directory inside the "src" folder. Then, run the `uvicorn main:app` command. When the API is ready for use, open the "api_evaluation" notebook in the "notebooks" folder. In this notebook, you will find more information on how to use this API.


## Tech Stack

### Back-End
- **Language:**
  - Python

- **Libraries and Frameworks:**
  - Langchain:  A framework for developing applications powered by language models. It provides tools for combining LLMs with other sources of computation or knowledge.
  - VertexAI: Google Cloud's machine learning platform that allows developers to use embedding and pretrained models like gemini.
  - chromaDB: An open-source embedding database that makes it easy to build AI applications with embeddings and semantic search.
  - RAGAS: A framework for evaluating RAG pipelines, offering metrics and tools to assess the quality of RAG systems.
  - datasets: A Hugging Face library that facilitates the use of the Dataset data type, especially used in the field of AI.
  - FastAPI: A modern, fast (high-performance) web framework for building APIs with Python based on standard Python type hints.

### Note

MedInforma is a valuable tool for quick medical research, offering easy access to health-related information. While the AI assistant can provide medical information and suggestions, it's crucial to remember:

1. Always verify the sources of the answers for more comprehensive information.
2. Any medical diagnoses or advice provided should be considered as general recommendations only.
3. The information provided does not have the same relevance or professional weight as that given by a qualified healthcare professional.
4. For any specific health concerns or diagnoses, it is strongly recommended to consult with a medical professional.

MedInforma is designed to be a helpful starting point for medical information, but it should not replace professional medical advice, diagnosis, or treatment.