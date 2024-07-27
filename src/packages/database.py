from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import unicodedata
from pathlib import Path
from tqdm import tqdm


class DocumentDatabase:
    def __init__(self, chroma_path: str, embedding_function, create_new: bool = False, 
                 files_directory: str | None = None, metadata_dict: dict | None = None, 
                 batch_size: int | None = None) -> None:
        """
        Initialize the DocumentDatabase instance.

        Args:
            chroma_path (str): The path to the Chroma database directory.
            embedding_function: The function used to generate embeddings for the documents.
            create_new (bool, optional): Flag indicating whether to create a new database. Defaults to False.
            files_directory (str | None, optional): The directory containing markdown files to load into the database. Required if create_new is True. Defaults to None.
            metadata_dict (dict | None, optional): A dictionary containing metadata for the documents. Required if create_new is True. Defaults to None.
            batch_size (int | None, optional): The number of documents to process in each batch. Defaults to None.

        Raises:
            Exception: If the database could not be created or loaded.
        """
        self.chroma_path = chroma_path
        self.embedding_function = embedding_function

        if create_new:
            self.database = self._create_database(files_directory, metadata_dict, batch_size)
        else:
            self.database = self._load_existing_database()

    def _create_database(self, files_directory: str, metadata_dict: dict, batch_size: int | None) -> Chroma:
        """
        Create a new Chroma database from markdown files in the specified directory.

        Args:
            files_directory (str): The directory containing markdown files to load into the database.
            metadata_dict (dict): A dictionary containing metadata for the documents.
            batch_size (int | None): The number of documents to process in each batch. If None, all documents are processed at once.

        Raises:
            Exception: If the database could not be created due to missing or incorrect arguments.
        """
        try:
            loader = DirectoryLoader(files_directory, glob="*.md")
            documents = loader.load()
            chunks = self._split_text_into_chunks(documents, metadata_dict)
            return self._load_chunks_into_database(chunks, batch_size)
        except Exception as e:
            raise Exception("The database could not be created. Please complete all the arguments.") from e

    def _load_existing_database(self) -> Chroma:
        """
        Load an existing Chroma database.

        This method attempts to load an existing Chroma database from the specified
        directory using the provided embedding function. If the database cannot be
        loaded, an exception is raised.

        Raises:
            Exception: If the database could not be loaded, an exception is raised
                    with a message indicating the failure.
        """
        try:
            return Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        except Exception as e:
            raise Exception("The database could not be loaded. Please create a new instance") from e

    def _split_text_into_chunks(self, documents: list[Document], metadata_dict: dict) -> list:
        """
        Split the provided documents into smaller chunks and enrich their metadata.

        This method uses a RecursiveCharacterTextSplitter to divide the documents into
        smaller chunks based on specified chunk size and overlap. It also enriches the
        metadata of each chunk with additional information from the provided metadata dictionary.

        Args:
            documents (list[Document]): A list of Document objects to be split into chunks.
            metadata_dict (dict): A dictionary containing metadata for the documents. The keys
                                should be the document titles, and the values should be dictionaries
                                with keys 'also_called', 'related_topic', and 'url'.

        Returns:
            list: A list of chunks with enriched metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2300,
            chunk_overlap=10,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            path = Path(chunk.metadata['source'])
            title = unicodedata.normalize('NFC', path.stem)

            chunk.metadata['also_called'] = '[' + ', '.join(f'"{alias}"' for alias in metadata_dict[title]['also_called']) + ']'
            chunk.metadata['related_topic'] = '[' + ', '.join(f'"{topic}"' for topic in metadata_dict[title]['related_topic']) + ']'
            chunk.metadata['url'] = metadata_dict[title]['url']

        return chunks

    def _load_chunks_into_database(self, chunks: list, batch_size: int | None = None, max_attempts: int = 5) -> Chroma:
        """
        Load chunks of documents into the Chroma database.

        This method loads the provided chunks of documents into the Chroma database. If a batch size is specified,
        the documents are loaded in batches. Otherwise, all documents are loaded at once. The method also includes
        a retry mechanism to handle potential failures during the loading process.

        Args:
            chunks (list): A list of document chunks to be loaded into the database.
            batch_size (int | None, optional): The number of documents to process in each batch. If None, all documents are processed at once. Defaults to None.
            max_attempts (int, optional): The maximum number of attempts to retry loading documents in case of failure. Defaults to 5.

        Raises:
            Exception: If the documents could not be loaded after the specified number of attempts.
        """
        if batch_size is None:
            db = Chroma.from_documents(chunks, self.embedding_function, persist_directory=self.chroma_path)
        else:
            db = Chroma.from_documents([chunks[0]], self.embedding_function, persist_directory=self.chroma_path)
            total_chunks = len(chunks)
            for i in tqdm(range(1, total_chunks, batch_size), desc="Loading documents"):
                subset = chunks[i:i + batch_size]
                self._attempt_to_add_documents(db, subset, max_attempts)
        return db

    def _attempt_to_add_documents(self, db, documents: list, max_attempts: int) -> None:
        """
        Attempt to add documents to the Chroma database with a retry mechanism.

        This method tries to add the provided list of documents to the Chroma database.
        If the operation fails, it will retry up to a specified number of attempts.

        Args:
            db: The Chroma database instance where the documents will be added.
            documents (list): A list of document chunks to be added to the database.
            max_attempts (int): The maximum number of attempts to retry adding documents in case of failure.

        Raises:
            Exception: If the documents could not be added after the specified number of attempts.
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                Chroma.add_documents(self=db, documents=documents)
                break
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise Exception(f"The function failed after {max_attempts} attempts.") from e

    def search(self, query: str, top_k: int):
        """
        Perform a similarity search on the Chroma database.

        This method takes a query string and performs a similarity search on the Chroma database,
        returning the top-k most relevant documents along with their relevance scores.

        Args:
            query (str): The search query string.
            top_k (int): The number of top relevant documents to return.

        Returns:
            list: A list of tuples, where each tuple contains a document and its relevance score.
        """
        return self.database.similarity_search_with_relevance_scores(query, k=top_k)