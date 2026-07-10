
from abc import ABC, abstractmethod
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import bm25s
import json

from .data_models import MinimalSearchResults, MinimalSource
from uuid import uuid4


class RetrieverError(Exception):
    pass


class NoPreviousIndexError(RetrieverError):
    pass


class AbstractDocumentLoader(ABC):
    """
    Common interface for implementing a document loader.

    The document loader loads a list of Document using
    the _load_documents(...) function.
    """
    @abstractmethod
    def _load_documents(
        self,
        chunk_size: int,
        overlap: float,
            main_directory: str) -> list[Document]:
        """
        Loads documents.
        """
        pass


class MarkdownPythonDocumentLoader(AbstractDocumentLoader):
    """
    Document loader implementation that loads markdown and
    python documents with different parameters.
    """
    def _load_documents_in_given_format(
            self,
            chunk_size: int,
            overlap: float,
            extension: str,
            language: Language,
            main_directory: str = './data/raw'
            ) -> list[Document]:
        """
        Method to load Documents from a given file format
        """
        # Load files
        loader = DirectoryLoader(main_directory,
                                 glob=f"**/*.{extension}",
                                 loader_cls=TextLoader)
        documents = loader.load()

        # Chunk files
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * overlap),
            add_start_index=True)
        chunks = text_splitter.split_documents(documents)

        # Add 'end_index' key in metadata
        for doc in chunks:
            doc.metadata['end_index'] = \
                doc.metadata['start_index'] + len(doc.page_content)

        return chunks

    def _load_documents(
            self,
            chunk_size: int,
            overlap: float,
            main_directory: str = './data/raw') -> list[Document]:
        """
        Method used to load markdown and python documents, chunk them
        and combine them into one single document type
        """
        md_files = self._load_documents_in_given_format(
            chunk_size,
            overlap,
            'md',
            Language.MARKDOWN,
            main_directory
        )
        py_files = self._load_documents_in_given_format(
            chunk_size,
            overlap,
            'py',
            Language.PYTHON,
            main_directory
        )

        return md_files + py_files


class BM25sRetriever(BaseRetriever):
    """
    Lanchain Retriever using BM25s library.
    """
    _corpus: list[str] | None = None
    _retriever: bm25s.BM25 | None = None
    _documents: list[Document] | None = None

    def index(self, chunk_size: int, overlap: float) -> bm25s.BM25:
        """
        Index the data in ./data/raw/* and creates Document objects out of it.
        """
        # Load chunked documents
        doc_loader = MarkdownPythonDocumentLoader()
        self._documents = doc_loader._load_documents(chunk_size, overlap)

        # Convert documents into a list of text that will be processed
        self._corpus = [doc.page_content for doc in self._documents]

        # Tokenize corpus
        tokenized_corpus = bm25s.tokenize(self._corpus)

        # Start indexing
        self._retriever = bm25s.BM25()
        self._retriever.index(tokenized_corpus)

        return self._retriever

    def retrieve(
            self,
            query: str,
            run_manager: CallbackManagerForRetrieverRun | None,
            k: int = 5
            ) -> MinimalSearchResults:
        """
        Alternative function to _get_relevant_documents
        to get documents relevant to a query, in a list
        of MinimalSource objects.
        """
        # Get documents
        documents = self._get_relevant_documents(
            query,
            k,
            run_manager=run_manager
        )

        # Create MinimalSource objects
        sources: list[MinimalSource] = []
        for doc in documents:
            sources.append(
                MinimalSource(
                    file_path=doc.metadata['source'],
                    first_character_index=doc.metadata['start_index'],
                    last_character_index=doc.metadata['end_index'],
                )
            )

        # Create MinimalSearchResult object to return
        search_results = MinimalSearchResults(
            question_id=str(uuid4()),
            question=query,
            retrieved_sources=sources
        )

        return search_results

    def _get_relevant_documents(
            self,
            query: str,
            k: int = 5,
            *,
            run_manager: CallbackManagerForRetrieverRun | None
            ) -> list[Document]:
        """
        Retrieve the best sources from a given query.
        """
        # Verify index has been done first
        if self._retriever is None:
            raise NoPreviousIndexError('Tried to retrieve without '
                                       'indexing first.')
        if self._corpus is None:
            raise RetrieverError('Tried to retrieve with no corpus.')
        if self._documents is None:
            raise RetrieverError('Tried to retrieve with no documents.')

        # Tokenize query and then retrieve sources
        tokenized_query = bm25s.tokenize(query)
        retrieved = self._retriever.retrieve(tokenized_query,
                                             self._corpus,
                                             k=5)

        # Turn sources into Document objects
        documents = []
        for retrieved in retrieved[0][0]:
            for chunk in self._documents:
                if chunk.page_content == retrieved:
                    documents.append(chunk)

        # Return list of Document containing sources
        return documents

    def export(self, path: str = './data/processed') -> None:
        """
        Export the index to a folder.
        """
        # Verify data
        if self._retriever is None or self._corpus is None \
           or self._documents is None:
            raise RetrieverError('Failed to save an invalid retriever.')

        try:
            # Export index
            self._retriever.save(f'{path}/index')

            # Export corpus
            with open(f'{path}/index/corpus.json', 'w') as f:
                json.dump(self._corpus, f)

            # Export Documents
            docs = [doc.model_dump() for doc in self._documents]
            with open(f'{path}/index/documents.json', 'w') as f:
                json.dump(docs, f)
        except (PermissionError):
            raise RetrieverError("Couldn't export your retriever.")

    def load(self, path: str = './data/processed') -> None:
        """
        Loads a previous index that has been exported.
        """
        try:
            # Load retriever
            self._retriever = bm25s.BM25.load(f'{path}/index')

            # Load corpus
            with open(f'{path}/index/corpus.json', 'r') as c:
                self._corpus: list[str] = json.load(c)

            # Load documents
            with open(f'{path}/index/documents.json', 'r') as d:
                loaded_data = json.load(d)

                self._documents: list[Document] = [
                    Document(**doc_dict) for doc_dict in loaded_data
                ]
        except FileNotFoundError:
            raise RetrieverError("Couldn't load a previous retriever.")


if __name__ == '__main__':

    # # Index
    # print('init retriever')
    # retriever = BM25sRetriever()
    # print('index...')
    # retriever.index(chunk_size=2000, overlap=5/100)
    # retriever.export()

    # Load retriever
    retriever = BM25sRetriever()
    retriever.load()

    query = 'How to setup an OpenAI server ?'
    print("retrieve for", query, '...')
    results = retriever.retrieve(query, k=5, run_manager=None)
    print(results)
