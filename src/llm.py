
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