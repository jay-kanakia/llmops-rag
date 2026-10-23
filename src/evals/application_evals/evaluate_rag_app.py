from deepeval.metrics import (
    GEval,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams
from deepeval.metrics.g_eval import Rubric
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig,DisplayConfig
from deepeval.evaluate import evaluate
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

model = 'gpt-5.4-mini'

# defining in-built metrics
recall = ContextualRecallMetric(model=model)
precision = ContextualPrecisionMetric(model=model)
context_relevancy = ContextualRelevancyMetric(model=model)
answer_relevancy = AnswerRelevancyMetric(model=model)
faithfullness = FaithfulnessMetric(model=model)

# define the custom metrics
answer_correctness = GEval(
    name="answer correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare
    the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0,5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has minor differences"),
            Rubric(score_range=(10,10), expected_outcome=r"100% correct")],
    model=model
)

simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read the 'actual output' first and then the 'input'",
        "Check whether the response is simple and easy to understand or not",
        "Make sure the response has least number of technical jargons and is student friendly"
    ],
    rubric=[
        Rubric(score_range=(0,3), expected_outcome="Too Difficult"),
        Rubric(score_range=(4,7), expected_outcome="Slightly difficult"),
        Rubric(score_range=(8,9), expected_outcome="moderately simple"),
        Rubric(score_range=(10,10), expected_outcome="very simple")
    ],
    model=model
)

# define the dataset path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_PATH = ROOT_DIR/"data"/"evaluation"/"eval_dataset"/"evaluation_dataset_final.json"

if DATASET_PATH.exists():
    dataset = EvaluationDataset()

    dataset.add_test_cases_from_json_file(
        file_path=DATASET_PATH,
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        retrieval_context_key_name="retrieval_context"
    )

# store the test case in a list
test_case = dataset.test_cases

# defining config
EVALUATION_RESULT_PATH = ROOT_DIR/"reports"/"evaluation_results"
EVALUTAION_REPORT_PATH = ROOT_DIR/"reports"/"evaluation_report"

display_config = DisplayConfig(
            verbose_mode=True,
            results_folder=EVALUATION_RESULT_PATH.as_posix(),
            file_type='md',
            file_output_dir=EVALUTAION_REPORT_PATH.as_posix()
)

async_config=AsyncConfig(
        throttle_value=3, max_concurrent=5
    )

# evaluate the dataset
evaluate(
    test_cases=test_case,
    metrics=[
        recall,precision,context_relevancy,answer_relevancy,answer_correctness,simple_explanation,faithfullness
    ],
    async_config=async_config,
    display_config=display_config
)