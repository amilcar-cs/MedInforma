import unittest
import os 
import sys
from unittest.mock import Mock, patch
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

sys.path.append(os.path.abspath(os.path.join('..')))
from packages.model import MedicalAssistantModel

class TestMedicalAssistantModel(unittest.TestCase):
    @patch('packages.model.ChatVertexAI')
    @patch('packages.model.VertexAIEmbeddings')
    def test_invalid_credentials(self, mock_embeddings, mock_chat):
        mock_chat.side_effect = Exception("Invalid credentials")
        mock_embeddings.side_effect = Exception("Invalid credentials")
        
        with self.assertRaises(Exception) as context:
            MedicalAssistantModel("invalid_credentials", "embedding_model", "chat_model")
        
        self.assertTrue("Invalid credentials" in str(context.exception))

    @patch('packages.model.ChatVertexAI')
    @patch('packages.model.VertexAIEmbeddings')
    def test_non_existent_embedding_model(self, mock_embeddings, mock_chat):
        mock_embeddings.side_effect = Exception("Model not found")
        mock_chat.return_value = Mock()
        
        with self.assertRaises(Exception) as context:
            MedicalAssistantModel("valid_credentials", "non_existent_embedding_model", "chat_model")
        
        self.assertTrue("Model not found" in str(context.exception))

    @patch('packages.model.ChatVertexAI')
    @patch('packages.model.VertexAIEmbeddings')
    def test_non_existent_chat_model(self, mock_embeddings, mock_chat):
        mock_embeddings.return_value = Mock()
        mock_chat.side_effect = Exception("Model not found")
        
        with self.assertRaises(Exception) as context:
            MedicalAssistantModel("valid_credentials", "embedding_model", "non_existent_chat_model")
        
        self.assertTrue("Model not found" in str(context.exception))

    @patch('packages.model.ChatVertexAI')
    @patch('packages.model.VertexAIEmbeddings')
    def test_empty_context(self, mock_embeddings, mock_chat):
        mock_model = MedicalAssistantModel("valid_credentials", "embedding_model", "chat_model")
        mock_model.chat_chain = Mock()
        mock_model.chat_chain.invoke.return_value = Mock(content="No puedo responder sin contexto.")

        result = mock_model.predict("", "¿Cuál es el tratamiento para la hipertensión?")
        self.assertEqual(result, "No puedo responder sin contexto.")

    @patch('packages.model.ChatVertexAI')
    @patch('packages.model.VertexAIEmbeddings')
    def test_empty_query(self, mock_embeddings, mock_chat):
        mock_model = MedicalAssistantModel("valid_credentials", "embedding_model", "chat_model")
        mock_model.chat_chain = Mock()
        mock_model.chat_chain.invoke.return_value = Mock(content="No se ha proporcionado ninguna pregunta.")

        result = mock_model.predict("Contexto sobre hipertensión", "")
        self.assertEqual(result, "No se ha proporcionado ninguna pregunta.")