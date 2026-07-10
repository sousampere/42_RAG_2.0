#!/usr/bin/python3

import fire
from colorama import Fore
import json
from tqdm import tqdm

from .data_models import MinimalSearchResults, StudentSearchResults

from .retriever import BM25sRetriever, RetrieverError


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
        # Invalid data protection
        if max_chunk_size <= 199:
            print(f"[RAG] ❌ {Fore.RED}max_chunk_size must be at least 200.")
            return None

        print(f"[RAG] Indexing with {max_chunk_size} chunk size...")

        # Index
        retriever = BM25sRetriever()
        retriever.index(max_chunk_size, overlap=5/100)
        retriever.export()

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
        # Invalid data protection
        if k <= 0:
            print(f"[RAG] ❌ {Fore.RED}Invalid number of sources (must be >= 1).")
            return None

        # Load retriever
        retriever = BM25sRetriever()
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
        for row in results.retrieved_sources:
            print(f'{Fore.RESET + row.file_path} {Fore.YELLOW}[{row.first_character_index}:{row.last_character_index}]')

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
        # Load retriever
        retriever = BM25sRetriever()
        try:
            retriever.load()
        except RetrieverError:
            print(f"[RAG] ❌ {Fore.RED}Couldn't load the previous index. "
                  "Please indexate the data first.")
            exit()

        # Import dataset
        with open(dataset_path, 'r') as f:
            json_data = json.load(f)
        dataset = json_data['rag_questions']

        # Start gathering sources for each question
        dataset_results = StudentSearchResults(
            search_results=[],
            k=k
        )
        with tqdm(total=len(dataset)) as pbar:
            for question in dataset:
                # Retrieve
                results = retriever.retrieve(
                    question['question'],
                    k=k,
                    run_manager=None
                )
                # Add results to dataset_results
                dataset_results.search_results.append(
                    MinimalSearchResults(
                        question_id=question['question_id'],
                        question=question['question'],
                        retrieved_sources=results.retrieved_sources
                    )
                )
                pbar.update(1)

        # Export search results
        search_results = dataset_results.model_dump()
        with open(save_directory, 'w') as f:
            json.dump(search_results, f)

        print(f"[RAG] ✅ {Fore.GREEN}Dataset successfully processed !")

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
