import unittest
import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.abspath(os.path.join('..')))
from packages.database import DocumentDatabase

class TestDocumentDatabase(unittest.TestCase):
    def setUp(self):
        self.chroma_path = "test_path"
        self.embedding_function = Mock()

    @patch('packages.database.Chroma')
    def test_load_existing_database(self, mock_chroma):
        database = DocumentDatabase(self.chroma_path, self.embedding_function)
        mock_chroma.assert_called_once_with(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        self.assertIsNotNone(database.database)

    @patch('packages.database.DirectoryLoader')
    @patch('packages.database.Chroma')
    def test_create_new_database(self, mock_chroma, mock_loader):
        files_directory = "test_files"
        metadata_dict = {"test_doc": {"also_called": [], "related_topic": [], "url": ""}}
        mock_loader.return_value.load.return_value = [Mock(page_content="Test content", metadata={"source": "test_doc.md"})]
        
        database = DocumentDatabase(self.chroma_path, self.embedding_function, create_new=True, 
                                    files_directory=files_directory, metadata_dict=metadata_dict)
        
        mock_loader.assert_called_once_with(files_directory, glob="*.md")
        mock_chroma.from_documents.assert_called()
        self.assertIsNotNone(database.database)

    def test_search_returns_results_in_correct_format(self):
        database = DocumentDatabase(self.chroma_path, self.embedding_function)
        database.database = Mock()
        mock_doc = Mock()
        mock_results = [(mock_doc, 0.9)]
        database.database.similarity_search_with_relevance_scores.return_value = mock_results
        
        results = database.search("test query", 1)
        
        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)
        self.assertIs(results[0][0], mock_doc)
        self.assertIsInstance(results[0][1], float)

    def test_search_with_no_results(self):
        database = DocumentDatabase(self.chroma_path, self.embedding_function)
        database.database = Mock()
        database.database.similarity_search_with_relevance_scores.return_value = []
        
        results = database.search("test query", 5)
        
        self.assertEqual(results, [])
        database.database.similarity_search_with_relevance_scores.assert_called_once_with("test query", k=5)

if __name__ == '__main__':
    unittest.main()