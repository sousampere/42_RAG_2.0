
import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    Class corresponding to a minimal source, containing :

    - file_path: str -> Path of the file
    - first_character_index: int -> Index of the first character of the file
    - last_character_index: int -> Index of the last character
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    Class corresponding to an unanswered question, containing :

    - question_id: str -> UUID randomly generated
    - question: str -> The question itself
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Class corresponding to an answered question, containing:

    - sources: list[MinimalSource] -> List of sources used
        to answer a question
    - answer: str -> The answer itself
    """
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    Class representing a list of RAG questions, containing:

    - rag_question: list[AnsweredQuestion | UnansweredQuestion] -> List of
        questions in the dataset
    """
    rag_question: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Class representing a search result, containing:

    - question_id: str -> The UUID of the question it's answering
    - question: str -> The question itself
    - retrieved_sources: list[MinimalSource] -> List of sources that may solve
        this question
    """
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Class representing a minimal answer, containing:

    - answer: str -> The answer to the question

    From MinimalSearchResult inheritance:

    - question_id: str -> The UUID of the question it's answering
    - question: str -> The question itself
    - retrieved_sources: list[MinimalSource] -> List of sources that
        may solve this question
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Class representing multiple search results, containing:

    - search_results: list[MinimalSearchResults] -> The search results
    - k: int -> The number of result for each search result
    """
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswers(BaseModel):
    """
    Class representing multiple search results with answers, containing:

    - search_results: list[MinimalAnswer] -> List of minimal answers
    - k: int -> The number of result for each search result
    """
    # Ignoring mypy error due to subject requiring this exact code
    search_results: list[MinimalAnswer]
    k: int
