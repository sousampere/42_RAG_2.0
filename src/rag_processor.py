

from abc import ABC, abstractmethod
import json
from tqdm import tqdm

from .llm import LLM
from .retriever import BM25sRetriever, RetrieverError
from .data_models import MinimalSearchResults, StudentSearchResults, \
    StudentSearchResultsAndAnswers


class RagProcessorError(Exception):
    pass


class AbstractRagProcessor(ABC):
    @abstractmethod
    def index(self, max_chunk_size: int) -> bool:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> MinimalSearchResults:
        pass

    @abstractmethod
    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
            k: int) -> StudentSearchResults:
        pass

    @abstractmethod
    def answer(self, query: str, k: int) -> str:
        pass

    @abstractmethod
    def answer_dataset(
        self,
        student_search_results_path: str,
            save_directory: str) -> StudentSearchResultsAndAnswers:
        pass

    @abstractmethod
    def evaluate(
        self,
        student_search_result_path: str,
            dataset_path: str) -> None:
        pass


class RagProcessor(AbstractRagProcessor):
    def index(self, max_chunk_size: int) -> bool:
        # Invalid data protection
        if max_chunk_size <= 199:
            raise RagProcessorError("max_chunk_size must be at least 200.")

        # Index
        retriever = BM25sRetriever()
        retriever.index(max_chunk_size, overlap=5/100)
        try:
            retriever.export()
        except (FileNotFoundError, PermissionError):
            raise RagProcessorError("Could not save the index.")

        return True

    def search(self, query: str, k: int) -> MinimalSearchResults:
        # Invalid data protection
        if k <= 0:
            raise RagProcessorError('Invalid number of '
                                    'sources (must be >= 1).')

        # Load retriever
        retriever = BM25sRetriever()
        try:
            retriever.load()
        except RetrieverError:
            raise RagProcessorError("Couldn't load the previous "
                                    "index. Please indexate the data first.")

        # Retrieve
        results = retriever.retrieve(
            query=query,
            k=k,
            run_manager=None
        )

        return results

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str, k: int) -> StudentSearchResults:
        # Load retriever
        retriever = BM25sRetriever()
        try:
            retriever.load()
        except RetrieverError:
            raise RagProcessorError('Couldn\'t load the previous index. '
                                    'Please indexate the data first.')

        # Import dataset
        try:
            with open(dataset_path, 'r') as f:
                json_data = json.load(f)
            dataset = json_data['rag_questions']
        except (PermissionError, FileNotFoundError):
            raise RagProcessorError('Couldn\'t load the given dataset.')

        # Start gathering sources for each question
        dataset_results = StudentSearchResults(
            search_results=[],
            k=k
        )
        with tqdm(total=len(dataset),
                  colour='green', ascii=" ▖▘▝▞▚▋█") as pbar:
            pbar.set_description(f'Processing {len(dataset)} questions')
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

        return dataset_results

    def answer(self, query: str, k: int) -> str:
        # Gather search results with RagProcessor
        results = self.search(
            query=query,
            k=k
        )

        # Load LLM
        llm = LLM()
        llm.load_model(
            model_name='Qwen/Qwen3-0.6B',
            device='auto'
        )

        # Process data into LLM to get the query's answer
        answer = str(llm.answer(results))

        # Return LLM answer
        return answer

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> StudentSearchResultsAndAnswers:
        return StudentSearchResultsAndAnswers

    def evaluate(self,
                 student_search_result_path: str,
                 dataset_path: str) -> None:
        return None
