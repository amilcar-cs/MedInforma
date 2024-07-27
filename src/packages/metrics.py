from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from datasets import Dataset
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
    answer_similarity,
    answer_correctness,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate
import pandas as pd

class Evaluator:
    def __init__(self, credentials, model_name: str, embedding_model_name: str) -> None:
        """
        Initialize the Evaluator with credentials, model name, and embedding model name.
        
        Args:
            credentials: Authentication credentials for accessing the models.
            model_name (str): The name of the evaluation model.
            embedding_model_name (str): The name of the embedding model.
        """
        self.credentials = credentials
        self.language_model = ChatVertexAI(model_name=model_name, credentials=self.credentials)
        self.embedding_model = self._initialize_embedding_model(embedding_model_name)
        self.metrics = self._initialize_metrics()

    def _initialize_embedding_model(self, embedding_model_name: str) -> VertexAIEmbeddings:
        """
        Initialize the embedding model.
        
        Args:
            embedding_model_name (str): The name of the embedding model.
        
        Returns:
            VertexAIEmbeddings: An instance of the embedding model.
        """
        return VertexAIEmbeddings(model_name=embedding_model_name, credentials=self.credentials)

    def _initialize_metrics(self) -> list:
        """
        Initialize the list of metrics to be used for evaluation.
        
        Returns:
            list: A list of metric functions.
        """
        return [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            harmfulness,
            answer_similarity,
            answer_correctness,
        ]

    def evaluate_dataset(self, dataset: Dataset, max_attempts: int = 5) -> pd.DataFrame | None:
        """
        Evaluate the dataset using the specified metrics and return the results as a DataFrame.
        
        Args:
            dataset (Dataset): The dataset to be evaluated. Must contain the following features:
            - "question": The user's query.
            - "ground_truth": The correct answer to the user's query.
            - "answer": The model's answer to the user's query.
            - "contexts": The context string prepared from search results.
        
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results.
            None: If no evaluation results are available.
        """
        if len(dataset) == 0:
            return None
        
        evaluation_results = []

        for index in range(len(dataset)):
            single_item_dataset = dataset.select(range(index, index + 1))
            evaluation_result = self._evaluate_single_item(single_item_dataset, max_attempts)
            if evaluation_result is not None:
                evaluation_results.append(evaluation_result)
        
        if not evaluation_results:
            return None
        
        combined_results = pd.concat(evaluation_results, ignore_index=True)
        return combined_results
            

    def _evaluate_single_item(self, single_item_dataset: Dataset, max_attempts: int) -> pd.DataFrame:
        """
        Evaluate a single item from the dataset.
        
        Args:
            single_item_dataset (Dataset): A dataset containing a single item.
        
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation result for the single item.
        """
        if len(single_item_dataset) == 0:
            return None
        
        attempt = 0
        while attempt < max_attempts:
            try:
                result = evaluate(
                    single_item_dataset,
                    metrics=self.metrics,
                    is_async=False,
                    raise_exceptions=False,
                    llm=self.language_model,
                    embeddings=self.embedding_model
                )
                return result.to_pandas()
            except:
                attempt += 1
        return None