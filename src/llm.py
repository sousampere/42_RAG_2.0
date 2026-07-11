
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
        sources: list[str] = []
        for s in search_results.retrieved_sources:
            with open(s.file_path, 'r') as f:
                f_content = f.read()
                sources.append(f_content[
                    s.first_character_index:s.last_character_index])

        # Prepare message
        messages = [
            {
                'role': 'system',
                'content': 'You will answer the user\'s query using '
                           'the provided sources. The answer will be '
                           ' concise. The sources are : \n\n'
                           f'{'\n\n'.join(sources)}'
            },
            {
                'role': 'user',
                'content': f'{search_results.question} /no_think'
            }
        ]

        # Generate LLM answer
        print(messages)
        for s in search_results.retrieved_sources:
            print(s.file_path)
        output = self._pipe(messages)

        # Return LLM generated output
        return str(output[0]["generated_text"][-1]["content"])
