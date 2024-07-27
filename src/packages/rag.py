from .database import DocumentDatabase
from .model import MedicalAssistantModel

class Assistant:
    def __init__(self, database: DocumentDatabase, model: MedicalAssistantModel):
        """
        Initialize the Assistant with a database and a model.
        """
        self.database = database
        self.model = model
    
    def ask(self, query: str, num_context_files: int = 3, min_similarity: float = 0.60, max_attempts: int = 5) -> dict:
        """
        Process a query by searching the database and generating a response using the model.

        Args:
            query (str): The user's query.
            num_context_files (int): Number of context files to retrieve from the database.
            min_similarity (float): Minimum similarity score required to consider a result relevant.
            max_attempts (int): Maximum number of attempts to get a response from the model.

        Returns:
            dict: A dictionary containing the response status, query, context, metadata, similarity, and completion.
        """
        if not query:
            response["status"] = "Model unable to respond: The query was empty."
            return response

        response = self._initialize_response(query)

        search_results = self.database.search(query=query, top_k=num_context_files)
        response["context"] = self._extract_context(search_results)
        response["similarity"] = search_results[0][1] if search_results else 0
        response["metadata"] = self._extract_metadata(search_results)
        
        if not search_results or search_results[0][1] < min_similarity:
            response["status"] = "No matches found: No relevant information was found in the database to answer the query."
            return response

        context = self._prepare_context(search_results)

        response["completion"] = self._generate_response(context, query, max_attempts)
        if response["completion"] is None:
            response["status"] = "Model unable to respond: The model was unable to generate a response."
        
        return response
    
    def _initialize_response(self, query: str) -> dict:
        """
        Initialize the response dictionary with default values.

        Args:
            query (str): The user's query.

        Returns:
            dict: A dictionary with default response values.
        """
        return {
            "status": "Success: The query was successfully answered.",
            "query": query,
            "context": [],
            "metadata": [],
            "similarity": 1,
            "completion": None,
        }

    def _extract_context(self, search_results) -> list:
        """
        Extract and clean the context from search results.

        Args:
            search_results (list): List of search results from the database.

        Returns:
            list: A list of cleaned context strings.
        """
        return [doc.page_content.replace('\n', ' ').replace('  ', ' ') for doc, _score in search_results]

    def _extract_metadata(self, search_results) -> list:
        """
        Extract metadata from search results.

        Args:
            search_results (list): List of search results from the database.

        Returns:
            list: A list of metadata dictionaries.
        """
        files = []
        metadata = []

        for doc, _score in search_results:
            file_source = doc.metadata.get("source", None)
            if file_source not in files:
                files.append(file_source)
                metadata.append({
                    "file": file_source,
                    "also_called": doc.metadata.get("also_called", None),
                    "related_topic": doc.metadata.get("related_topic", None),
                    "url": doc.metadata.get("url", None)
                })

        return metadata

    def _prepare_context(self, search_results) -> str:
        """
        Prepare the context string by joining the content of the retrieved documents.

        Args:
            search_results (list): List of search results from the database.

        Returns:
            str: A single string containing the joined context.
        """
        return "\n\n---\n\n".join([doc.page_content.replace('\n', ' ').replace('  ', ' ') for doc, _score in search_results])

    def _generate_response(self, context: str, query: str, max_attempts: int) -> str:
        """
        Generate a response using the model, with a specified number of attempts.

        Args:
            context (str): The context string prepared from search results.
            query (str): The user's query.
            max_attempts (int): Maximum number of attempts to get a response from the model.

        Returns:
            str: The generated response from the model, or None if no response was generated.
        """
        attempts = 0
        while attempts < max_attempts:
            response = self.model.predict(context, query)
            if response:
                return response
            attempts += 1
        return None