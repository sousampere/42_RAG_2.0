
from abc import ABC, abstractmethod

from transformers import pipeline

from .data_models import MinimalSearchResults


class LLMAugmenter(ABC):
    """
    Class used to augment retrieved response
    and generate LLM answer from a MinimalSearchResults.
    """
    @abstractmethod
    def load_model(
        self,
        model_name: str,
            device: str) -> None:
        """
        Loads the model.
        """
        pass

    @abstractmethod
    def answer(self, search_results: MinimalSearchResults) -> str:
        """
        Answer the query in the MinimalSearchResults using
        the previously loaded model.
        """
        pass


class LLM(LLMAugmenter):
    def load_model(self, model_name: str, device: str) -> None:
        """
        Loads the model
        """
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map=device
        )
        return None

    def answer(self, search_results: MinimalSearchResults) -> str:
        """
        Answer to the user's query using the available sources.
        """
        # Gather sources
        sources = []
        for result in search_results.retrieved_sources:
            with open(result.file_path, 'r') as f:
                file_content = f.read()
                sources.append(file_content[
                    result.first_character_index:result.last_character_index
                    ])
        print(sources[0])
        exit()

        chat_template = "You are a helpful assistant that gives answers on "
        "a user's query, by looking at the given sources.\n\n"
        "SOURCES :\n\n" + ''.join(search_results.retrieved_sources)
        messages = [
            {'role': 'user', 'content': 'What is 2+2 ? /no_think'}
        ]
        output = self._pipe(messages)
        print(output[0]["generated_text"][-1]["content"])
        return ''