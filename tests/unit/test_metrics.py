import unittest
import os 
import sys
from unittest.mock import Mock, patch
import pandas as pd
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join('..')))
from packages.metrics import Evaluator

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.credentials = Mock()
        self.model_name = "test_model"
        self.embedding_model_name = "test_embedding_model"
        with patch('packages.metrics.ChatVertexAI'), patch('packages.metrics.VertexAIEmbeddings'):
            self.evaluator = Evaluator(self.credentials, self.model_name, self.embedding_model_name)

    @patch('packages.metrics.evaluate')
    def test_evaluate_single_item_success(self, mock_evaluate):
        # Test successful evaluation
        mock_result = Mock()
        mock_result.to_pandas.return_value = pd.DataFrame({'metric': ['test_metric'], 'score': [0.9]})
        mock_evaluate.return_value = mock_result

        dataset = Dataset.from_dict({
            "question": ["Test question"],
            "ground_truth": ["Test ground truth"],
            "answer": ["Test answer"],
            "contexts": ["Test context"]
        })

        result = self.evaluator._evaluate_single_item(dataset, max_attempts=1)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result['metric'][0], 'test_metric')
        self.assertEqual(result['score'][0], 0.9)

    @patch('packages.metrics.evaluate')
    def test_evaluate_single_item_retry_success(self, mock_evaluate):
        # Test successful evaluation after one failure
        mock_evaluate.side_effect = [Exception("Test error"), Mock(to_pandas=Mock(return_value=pd.DataFrame({'metric': ['test_metric'], 'score': [0.8]})))]

        dataset = Dataset.from_dict({
            "question": ["Test question"],
            "ground_truth": ["Test ground truth"],
            "answer": ["Test answer"],
            "contexts": ["Test context"]
        })

        result = self.evaluator._evaluate_single_item(dataset, max_attempts=2)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result['metric'][0], 'test_metric')
        self.assertEqual(result['score'][0], 0.8)

    @patch('packages.metrics.evaluate')
    def test_evaluate_single_item_all_attempts_fail(self, mock_evaluate):
        # Test when all evaluation attempts fail
        mock_evaluate.side_effect = Exception("Test error")

        dataset = Dataset.from_dict({
            "question": ["Test question"],
            "ground_truth": ["Test ground truth"],
            "answer": ["Test answer"],
            "contexts": ["Test context"]
        })

        result = self.evaluator._evaluate_single_item(dataset, max_attempts=3)
        
        self.assertIsNone(result)
        self.assertEqual(mock_evaluate.call_count, 3)

    @patch('packages.metrics.evaluate')
    def test_evaluate_single_item_empty_dataset(self, mock_evaluate):
        # Test with an empty dataset
        dataset = Dataset.from_dict({
            "question": [],
            "ground_truth": [],
            "answer": [],
            "contexts": []
        })

        result = self.evaluator._evaluate_single_item(dataset, max_attempts=1)
        
        self.assertIsNone(result)
        mock_evaluate.assert_not_called()

    @patch('packages.metrics.evaluate')
    def test_evaluate_single_item_custom_metrics(self, mock_evaluate):
        # Test with custom metrics
        mock_result = Mock()
        mock_result.to_pandas.return_value = pd.DataFrame({'custom_metric': ['test_custom_metric'], 'score': [0.95]})
        mock_evaluate.return_value = mock_result

        dataset = Dataset.from_dict({
            "question": ["Test question"],
            "ground_truth": ["Test ground truth"],
            "answer": ["Test answer"],
            "contexts": ["Test context"]
        })

        custom_metrics = [Mock()]
        self.evaluator.metrics = custom_metrics

        result = self.evaluator._evaluate_single_item(dataset, max_attempts=1)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result['custom_metric'][0], 'test_custom_metric')
        self.assertEqual(result['score'][0], 0.95)
        mock_evaluate.assert_called_once_with(
            dataset,
            metrics=custom_metrics,
            is_async=False,
            raise_exceptions=False,
            llm=self.evaluator.language_model,
            embeddings=self.evaluator.embedding_model
        )

if __name__ == '__main__':
    unittest.main()