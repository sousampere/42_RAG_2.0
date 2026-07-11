

from abc import ABC, abstractmethod
import json
from tqdm import tqdm

from .llm import LLM
from .retriever import BM25sRetriever, RetrieverError
from .data_models import MinimalSearchResults, StudentSearchResults, \
    StudentSearchResultsAndAnswers


class RagProcessorError(Exception):
    """
    Errors related to the RagProcessor
    """
    pass


class AbstractRagProcessor(ABC):
    """
    Abstract class for a RagProcessor
    """
    @abstractmethod
    def index(self, max_chunk_size: int) -> bool:
        """
        Method used to index data
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> MinimalSearchResults:
        """
        Method used to retrieve data from a query
        """
        pass

    @abstractmethod
    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
            k: int) -> StudentSearchResults:
        """
        Method used to process a whole dataset of questions
        """
        pass

    @abstractmethod
    def answer(self, query: str, k: int) -> str:
        """
        Method used to answer a query using sources
        """
        pass

    @abstractmethod
    def answer_dataset(
        self,
        student_search_results_path: str,
            save_directory: str) -> StudentSearchResultsAndAnswers:
        """
        Method to answer a dataset of questions
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        student_search_result_path: str,
            dataset_path: str) -> None:
        """
        Method used to evaluate the performance of the retriever
        """
        pass


class RagProcessor(AbstractRagProcessor):
    """
    RagProcessor class that orchestrate the RAG

    It's the core of the program. It has multiple methods :
    index, search, search_dataset, answer, answer_dataset
    and evaluate. All these methods are used to interract
    with the documentation in data/raw/*.
    """
    def index(self, max_chunk_size: int) -> bool:
        """
        Index the data in data/raw and export the retriever
        for further use.
        """
        # Invalid data protection
        if max_chunk_size <= 199:
            raise RagProcessorError("max_chunk_size must be at least 200.")

        # Index
        retriever = BM25sRetriever()
        retriever.index(max_chunk_size, overlap=15/100)
        try:
            retriever.export()
        except (FileNotFoundError, PermissionError):
            raise RagProcessorError("Could not save the index.")

        return True

    def search(self, query: str, k: int) -> MinimalSearchResults:
        """
        Search the data for a query. Returns the search result
        as a MinimalSearchResults object.
        """
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
        """
        Searches for sources for all questions in the given
        dataset_path. Returns results as a StudentSearchResults object.
        """
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
                # Make sure there is no more than 10 sources
                results.retrieved_sources = results.retrieved_sources[:10]
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
        """
        Answer a given query by retrieveing the best docs
        from the indexed data, and giving it to an LLM to
        make it create a custom answer to the query.
        """
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
        """
        Answer all question with sources from a previously
        processed dataset of queries.
        """
        return StudentSearchResultsAndAnswers

    def evaluate(self,
                 student_search_result_path: str,
                 dataset_path: str) -> None:
        """
        Evaluate the results of the RAG with ground truth
        datas to give a ratio of performances.
        """
        return None
