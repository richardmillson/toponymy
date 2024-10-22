import unittest

import anthropic
import cohere
import llama_cpp
import openai
from httpx import LocalProtocolError

from toponymy.llm_wrappers import (
    AnthropicWrapper,
    CohereWrapper,
    LlamaCppWrapper,
    OpenAIWrapper,
)


class TestAnthropicWrapper(unittest.TestCase):
    def test_init(self):
        AnthropicWrapper("API_KEY")


class TestCohereWrapper(unittest.TestCase):
    def test_init(self):
        # CohereWrapper.__init__ will fail when it tries to validate the given model without a valid API key.
        with self.assertRaises(cohere.UnauthorizedError) as context:
            CohereWrapper("API_KEY")


class TestLlamaCppWrapper(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError) as context:
            LlamaCppWrapper(model_path="non_existent_path")


class TestOpenAIWrapper(unittest.TestCase):
    def test_init(self):
        OpenAIWrapper("API_KEY")


if __name__ == "__main__":
    unittest.main()
