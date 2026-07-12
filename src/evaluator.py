
from pydantic import BaseModel, ValidationError
import json
from tqdm import tqdm

from .data_models import StudentSearchResults, RagDataset


class EvaluationError(Exception):
    pass


class EvaluationResult(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    questions_evaluated: int


class Evaluator:
    def evaluate(self,
                 student_search_result_path: str,
                 dataset_path: str,
                 k: int) -> EvaluationResult:
        """
        Evaluate the results of the RAG with ground truth
        datas to give a ratio of performances.
        """
        # Load student results
        try:
            with open(student_search_result_path, 'r') as f:
                data = json.load(f)
            search_results = StudentSearchResults.model_validate(data)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            raise EvaluationError('Unable to read the '
                                  'search_results properly.')
        except ValidationError:
            raise EvaluationError('Your student_search_result_path '
                                  'file is corrupted. Cannot read it.')

        # Load ground-truth dataset
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            ground_truth = RagDataset.model_validate(data)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            raise EvaluationError('Unable to read the '
                                  'ground truth dataset properly.')
        except ValidationError:
            raise EvaluationError('Your ground truth dataset '
                                  'file is corrupted. Cannot read it.')

        # Calculate results
        with tqdm(total=4, desc="Calculating racall@k") as pbar:
            k1 = self.get_recall_at_k(search_results, ground_truth, 1)
            pbar.update(1)
            k3 = self.get_recall_at_k(search_results, ground_truth, 3)
            pbar.update(1)
            k5 = self.get_recall_at_k(search_results, ground_truth, 5)
            pbar.update(1)
            k10 = self.get_recall_at_k(search_results, ground_truth, 10)
            pbar.update(1)

        try:
            results = EvaluationResult(
                recall_at_1=round(k1, 2),
                recall_at_3=round(k3, 2),
                recall_at_5=round(k5, 2),
                recall_at_10=round(k10, 2),
                questions_evaluated=len(search_results.search_results)
            )
        except ValidationError:
            raise EvaluationError('Could not evaluate the resuls.')

        return results

    @staticmethod
    def get_recall_at_k(
        search_results: StudentSearchResults,
        ground_truth: RagDataset,
            k: int) -> float:
        """
        Calculate the recall@k score over the whole dataset
        compared to the ground-truth dataset.

        Return a ratio of good retrieving (overlap of 0.05% at least)
        """
        score = 0
        # Compare each question with the ground truth
        for question in search_results.search_results:
            for truth_data in ground_truth.rag_questions:
                if truth_data.question_id == question.question_id:
                    # Corresponding data found. Comparing sources
                    for q in question.retrieved_sources[:k]:
                        if q.file_path == truth_data.sources[0].file_path:
                            # Source found. Calculating overlap
                            truth_src = truth_data.sources[0]

                            # Calulate overlap
                            intersection = max(
                                0,
                                min(
                                    truth_src.last_character_index,
                                    q.last_character_index
                                ) - max(
                                    truth_src.first_character_index,
                                    q.first_character_index))
                            union = (
                                truth_src.last_character_index - truth_src.
                                first_character_index
                                ) + (q.last_character_index - q.
                                     first_character_index) - intersection
                            overlap = intersection / union

                            if overlap >= 0.05:
                                score += 1
                                break
        if score == 0:
            return 0

        return score / len(search_results.search_results)
