from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

class MedicalAssistantModel:
    def __init__(self, credentials, embedding_model_name: str, chat_model_name: str, max_tokens: int = 8192, temperature: float = 1.0) -> None:
        """
        Initialize the MedicalAssistantModel with the specified parameters.

        This constructor sets up the necessary components for the MedicalAssistantModel, including
        the chat model and the embedding model. It uses the provided credentials to authenticate
        with the models and configures them with the specified parameters.

        Args:
            credentials: The credentials required to authenticate with the models.
            embedding_model_name (str): The name of the embedding model to be used.
            chat_model_name (str): The name of the chat model to be used.
            max_tokens (int, optional): The maximum number of tokens for the model's output. Defaults to 8192.
            temperature (float, optional): The temperature setting for the model, which controls the randomness of the output. Defaults to 1.0.
        """
        self.credentials = credentials
        self.chat_chain = self._initialize_chat_model(chat_model_name, max_tokens, temperature)
        self.embeddings = self._initialize_embedding_model(embedding_model_name)

    def _initialize_chat_model(self, model_name: str, max_tokens: int, temperature: float):
        """
        Initialize the chat model with the specified parameters.

        This method sets up the chat model by defining the system and human message templates,
        creating a prompt template, configuring the chat model with safety settings, and combining
        the prompt template and chat model into a chain.

        Args:
            model_name (str): The name of the chat model to be used.
            max_tokens (int): The maximum number of tokens for the model's output.
            temperature (float): The temperature setting for the model, which controls the randomness of the output.

        Returns:
            ChatVertexAI: A configured chat model chain ready to generate responses based on the provided context and query.
        """
        system_message = (
            """You are a helpful medical assistant who always tries to answer a question in Spanish in a detailed and professional manner based on the following context:

            {context}

            Your answer can be a variation of the context, such as a conclusion, medical diagnoses (if possible using context), comparison, paraphrase, or modify the wording to fit the question (such as being more informal or more scientific) etc. But you can't add additional information, please don't try, kindly reply that you can't respond with the information you currently have.
            If the question has no relation to your provided context, kindly mention that you cannot answer that question, but do not mention anything about the context."""
        )

        human_message = "{text}"

        prompt_template = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])

        chat_model = ChatVertexAI(
            max_input_tokens=1048576,
            max_output_tokens=max_tokens,
            model_name=model_name,
            max_tokens=max_tokens,
            credentials=self.credentials,
            top_p=0.95,
            top_k=40,
            temperature=temperature,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )

        chat_chain = prompt_template | chat_model

        return chat_chain

    def _initialize_embedding_model(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            embedding_model_name (str): The name of the embedding model.
        
        Returns:
            VertexAIEmbeddings: An instance of the embedding model.
        """
        return VertexAIEmbeddings(model_name=model_name, credentials=self.credentials)

    def predict(self, context: str, query: str) -> str:
        """
        Generate a prediction based on the provided context and query.

        This method uses the initialized chat model to generate a response in Spanish
        to a given query, based on the provided context. The response is generated
        according to the constraints and guidelines defined in the system message template.

        Args:
            context (str): The context information that the model will use to generate the response.
            query (str): The query or question that needs to be answered by the model.

        Returns:
            str: The generated response from the model based on the provided context and query.
        """
        prediction = self.chat_chain.invoke(
            {
                "context": context,
                "text": query,
            }
        )

        return prediction.content