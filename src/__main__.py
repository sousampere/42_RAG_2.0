#!/usr/bin/python3

import fire
from colorama import Fore
import json
from sympy import Q
from tqdm import tqdm
import os
import logging

from .rag_processor import RagProcessor, RagProcessorError
from .llm import LLM
from .data_models import MinimalSearchResults, StudentSearchResults
from .retriever import BM25sRetriever, RetrieverError

# Disable huggingface warnings
os.environ['HF_HUB_DISABLE_WARNINGS'] = '1'
logging.getLogger('transformers').setLevel(logging.ERROR)


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
        rag = RagProcessor()

        try:
            rag.index(max_chunk_size=max_chunk_size)
        except RagProcessorError as e:
            print(f'[RAG] ❌ {Fore.RED}{e}{Fore.RESET}')
            exit()

        print(f"[RAG] ✅ {Fore.GREEN}Data indexed successfully !{Fore.RESET}")

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
        # Load RagProcessor
        rag = RagProcessor()

        # Start searching using RagProcessor
        try:
            results = rag.search(
                query=query,
                k=k
            )
        except RagProcessorError as e:
            print(f'[RAG] ❌ {Fore.RED}{e}{Fore.RESET}')
            exit()

        # Print results
        for row in results.retrieved_sources:
            print(f'{Fore.RESET + row.file_path} {Fore.YELLOW}['
                  f'{row.first_character_index}:{row.last_character_index}]')

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
        rag = RagProcessor()

        try:
            dataset_results = rag.search_dataset(
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k
            )
        except RagProcessorError as e:
            print(f'[RAG] ❌ {Fore.RED}{e}{Fore.RESET}')
            exit()

        # Export search results
        search_results = dataset_results.model_dump()
        try:
            with open(save_directory, 'w') as f:
                json.dump(search_results, f)
        except (PermissionError):
            print(f'[RAG] ❌ {Fore.RED}Could not save the output.{Fore.RESET}')
            exit()

        print(f"[RAG] ✅ {Fore.GREEN}Dataset successfully processed !{Fore.RESET}")

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
        # Get search results for the query
        retriever = BM25sRetriever()
        # Try loading retriever
        try:
            retriever.load()
        except RetrieverError:
            print(f"[RAG] ❌ {Fore.RED}Couldn't load the previous index. "
                  "Please indexate the data first.")
            exit()
        # Retrieve
        results = retriever.retrieve(
            query=query,
            k=k,
            run_manager=None
        )

        # Load LLM
        llm = LLM()
        llm.load_model(model_name='Qwen/Qwen3-0.6B', device='auto')

        # Print response
        print(llm.answer(results))

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
    # Launch CLI
    fire.Fire(RagCLI, serialize='')
