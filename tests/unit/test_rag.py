import unittest
import os 
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.abspath(os.path.join('..')))
from packages.rag import Assistant

class TestAssistant(unittest.TestCase):
    def setUp(self):
        self.database_mock = Mock()
        self.model_mock = Mock()
        self.assistant = Assistant(self.database_mock, self.model_mock)

    def test_generate_response_success_first_attempt(self):
        self.model_mock.predict.return_value = "Generated response"
        response = self.assistant._generate_response("context", "query", 3)
        self.assertEqual(response, "Generated response")
        self.model_mock.predict.assert_called_once_with("context", "query")

    def test_generate_response_success_multiple_attempts(self):
        self.model_mock.predict.side_effect = [None, None, "Generated response"]
        response = self.assistant._generate_response("context", "query", 3)
        self.assertEqual(response, "Generated response")
        self.assertEqual(self.model_mock.predict.call_count, 3)

    def test_generate_response_failure_max_attempts_reached(self):
        self.model_mock.predict.return_value = None
        response = self.assistant._generate_response("context", "query", 3)
        self.assertIsNone(response)
        self.assertEqual(self.model_mock.predict.call_count, 3)

    def test_generate_response_success_last_attempt(self):
        self.model_mock.predict.side_effect = [None, None, None, None, "Generated response"]
        response = self.assistant._generate_response("context", "query", 5)
        self.assertEqual(response, "Generated response")
        self.assertEqual(self.model_mock.predict.call_count, 5)

    def test_generate_response_empty_string_response(self):
        self.model_mock.predict.return_value = ""
        response = self.assistant._generate_response("context", "query", 3)
        self.assertIsNone(response)
        self.assertEqual(self.model_mock.predict.call_count, 3)

if __name__ == '__main__':
    unittest.main()