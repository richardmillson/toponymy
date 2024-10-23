import json
import logging
import string
from abc import ABC, abstractmethod
from pathlib import Path

import tokenizers
import transformers

logging.basicConfig()
logger = logging.getLogger(__name__)


class BaseLlmWrapper(ABC):
    """Wraps an LLM to provide a consistent interface."""

    @abstractmethod
    def generate_topic_name(self, prompt: str, temperature: float) -> str:
        """Generates a topic name from a prompt."""
        pass

    @abstractmethod
    def generate_topic_cluster_names(
        self, prompt: str, old_names: list[str], temperature: float
    ) -> list[str]:
        """Generates new topic names for multiple topic names that were similar."""
        pass

    @property
    @abstractmethod
    def llm_instruction_base_layer() -> str:
        """The instruction for generating topic names for the base layer."""
        pass

    @property
    @abstractmethod
    def llm_instruction_intermediate_layer() -> str:
        """The instruction for generating topic names for intermediate layers."""
        pass

    @property
    @abstractmethod
    def llm_instruction_remedy() -> str:
        """The instruction for improving a topic name that had issues."""
        pass


try:

    import llama_cpp

    class LlamaCppWrapper(BaseLlmWrapper):

        def __init__(self, model_path, **kwargs):
            if not Path(model_path).exists():
                raise ValueError(f"Model path '{model_path}' doesn't exist.")
            self.model_path = model_path
            self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)
            for arg, val in kwargs.items():
                setattr(self, arg, val)

        def generate_topic_name(self, prompt, temperature=0.8):
            topic_name = self.llm(prompt, temperature=temperature)["choices"][0]["text"]
            if "\n" in topic_name:
                topic_name = topic_name.lstrip("\n ")
                topic_name = topic_name.split("\n")[0]
            topic_name = string.capwords(
                topic_name.strip(string.punctuation + string.whitespace)
            )
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm(prompt, temperature=temperature)
                topic_name_info_text = topic_name_info_raw["choices"][0]["text"]
                topic_name_info = json.loads(topic_name_info_text)
                result = []
                for old_name, name_mapping in zip(old_names, topic_name_info):
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values()[0]))
                    else:
                        result.append(old_name)

                return result
            except:
                return old_names

        @property
        def llm_instruction_base_layer(self):
            return "\nThe short distinguising topic name is:\n"

        @property
        def llm_instruction_intermediate_layer(self):
            return "\nThe short topic name that encompasses the sub-topics is:\n"

        @property
        def llm_instruction_remedy(self):
            return "\nA better and more specific name that still captures the topic of these article titles is:\n"

except ImportError:
    pass

try:

    import cohere

    class CohereWrapper(BaseLlmWrapper):

        def __init__(self, API_KEY, model="command-r-08-2024", local_tokenizer=None):
            self.llm = cohere.Client(api_key=API_KEY)

            try:
                self.llm.models.get(model)
            except cohere.errors.not_found_error.NotFoundError:
                models = [x.name for x in self.llm.models.list().models]
                msg = f"Model '{model}' not found, try one of {models}"
                raise ValueError(msg)
            self.model = model

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                ).text
                topic_name_info = json.loads(topic_name_info_raw)
                topic_name = topic_name_info["topic_name"]
            except:
                logger.warn(
                    (
                        "Failed to generate topic name with Cohere."
                        "\nException:\n%s\nPrompt:\n%s\nResponse:\n%s"
                    ),
                    e,
                    prompt,
                    topic_name_info_text,
                )
                topic_name = ""
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.text
                topic_name_info = json.loads(topic_name_info_text)
            except Exception as e:
                logger.warn(
                    (
                        "Failed to generate topic cluster names with Cohere."
                        "\nException:\n%s\nPrompt:\n%s\nResponse:\n%s"
                    ),
                    e,
                    prompt,
                    topic_name_info_text,
                )
                return old_names

            result = []
            for old_name, name_mapping in zip(old_names, topic_name_info):
                try:
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values())[0])
                    else:
                        logger.warn(
                            "Old name '%s' does not match the new name '%s'",
                            old_name,
                            list(name_mapping.keys())[0],
                        )
                        # Use old_name?
                        result.append(list(name_mapping.values())[0])
                except:
                    result.append(old_name)

            return result

        @property
        def llm_instruction_base_layer(self):
            return """
You are to give a brief (five to ten word) name describing this group.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

        @property
        def llm_instruction_intermediate_layer(self):
            return """
You are to give a brief (three to five word) name describing this group of papers.
The topic should be the most specific topic that encompasses the breadth of sub-topics, with a focus on the major sub-topics.
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

        @property
        def llm_instruction_remedy(self):
            return """
You are to give a brief (three to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be in JSON formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

except ImportError:
    pass

try:

    import anthropic

    class AnthropicWrapper(BaseLlmWrapper):

        def __init__(
            self, API_KEY, model="claude-3-haiku-20240307", local_tokenizer=None
        ):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.model = model

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = json.loads(topic_name_info_text)
                topic_name = topic_name_info["topic_name"]
            except:
                topic_name = ""
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = json.loads(topic_name_info_text)
                result = []
                for old_name, name_mapping in zip(old_names, topic_name_info):
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values()[0]))
                    else:
                        result.append(old_name)

                return result
            except:
                return old_names

        @property
        def llm_instruction_base_layer(self):
            return """
You are to give a brief (five to ten word) name describing this group.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

        @property
        def llm_instruction_intermediate_layer(self):
            return """
You are to give a brief (three to five word) name describing this group of papers.
The topic should be the most specific topic that encompasses the breadth of sub-topics, with a focus on the major sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

        @property
        def llm_instruction_remedy(self):
            return """
You are to give a brief (five to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

except ImportError:
    pass

try:

    import openai

    class OpenAIWrapper(BaseLlmWrapper):

        def __init__(self, API_KEY, model="gpt-4o-mini"):
            self.llm = openai.OpenAI(api_key=API_KEY)
            self.model = model

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                topic_name_info_text = topic_name_info_raw.choices[0].message.content

                topic_name_info = json.loads(topic_name_info_text)
                topic_name = topic_name_info["topic_name"]
                logger.info(topic_name_info)
            except Exception as e:
                topic_name = ""
                logger.warn(
                    (
                        "Failed to generate topic name with OpenAI."
                        "\nException:\n%s\nPrompt:\n%s\nResponse:\n%s"
                    ),
                    e,
                    prompt,
                    topic_name_info_text,
                )
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                topic_name_info_text = topic_name_info_raw.choices[0].message.content
                topic_name_info = json.loads(topic_name_info_text)
                result = []
                for old_name, name_mapping in zip(old_names, topic_name_info):
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values()[0]))
                    else:
                        result.append(old_name)

                return result
            except:
                return old_names

        @property
        def llm_instruction_base_layer(self):
            return """
You are to give a brief (five to ten word) name describing this group.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response must be **ONLY** JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

        @property
        def llm_instruction_intermediate_layer(self):
            return """
You are to give a brief (three to five word) name describing this group of papers.
The topic should be the most specific topic that encompasses the breadth of sub-topics, with a focus on the major sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

        @property
        def llm_instruction_remedy(self):
            return """
You are to give a brief (five to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""

except ImportError:
    pass
