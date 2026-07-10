#!/usr/bin/python3

import fire
from colorama import Fore

from .retriever import BM25sRetriever


class RagCLI:
    """
    RAG CLI
    """
    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        """
        Start indexing .py and .md files located in the ./data/raw/ folder.

        Files will be split into chunks of <max_chunk_size>
        characters at maximum.
        """
        print(f"[RAG] Indexing with {max_chunk_size} chunk size...")

        # Index
        retriever = BM25sRetriever()
        retriever.index(max_chunk_size, overlap=5/100)

        print(f"[RAG] ✅ {Fore.GREEN}Data indexed successfully !")
        return None

    @staticmethod
    def search(
        query: str,
            k: int = 5) -> None:
        """Search the most relevant files in the dataset for a given query.

        Args:
            query (str): String used as reference for the research
            k (int, optional): Number of results to retrieve. Defaults to 5.
        """
        print(f"Retrieving best top {k} best results for \"{query}\"...")
        return None

    @staticmethod
    def search_dataset(
        dataset_path: str,
        save_directory: str,
            k: int = 5) -> None:
        """Apply a search operation on a dataset of questions, and exporting
        the results in a given <save_directory> directory.

        Args:
            dataset_path (str): Path to the dataset containing queries.
            save_directory (str): Path to save the output file.
            k (int, optional): Number of retrieved sources. Defaults to 5.
        """
        print(f"Executing search on dataset located at {dataset_path} and "
              f"saving the result in {save_directory} with {k} sources "
              "for each query...")
        return None

    @staticmethod
    def answer(
        query: str,
            k: int = 5) -> None:
        """Generate a response to your query by giving the search result to
        an LLM, therefore augmenting it with a tailored response.

        Args:
            query (str): The query to get a response to.
            k (int, optional): Number of retrieved sources to give to the LLM.
            Defaults to 5.
        """
        print(f"Using LLM to answer your query \"{query}\" with "
              f"{k} sources in context...")
        return None

    @staticmethod
    def answer_dataset(
        student_search_results_path: str,
            save_directory: str) -> None:
        """Generates answers for the given <student_search_results_path> file
        containing the output of a previous "search_dataset" action.

        Args:
            student_search_results_path (str): File containing search results
            save_directory (str): Output with the LLM responses
        """
        print("Generating custom responses for the "
              f"search results in {student_search_results_path} and saving "
              f"the result in {save_directory}.")

    @staticmethod
    def evaluate(
        student_search_results_path: str,
            dataset_path: str) -> None:
        """Compare retrieved results with ground truth data.

        Args:
            student_search_results_path (str): Path to JSON file
            exported when running search on the dataset.
            dataset_path (str): Ground-truth dataset.
        """
        print("Evaluating your results")


if __name__ == '__main__':
    fire.Fire(RagCLI)
